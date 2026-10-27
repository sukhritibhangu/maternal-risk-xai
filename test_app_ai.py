"""API integration checks against the existing, unchanged saved model."""
import copy
import io
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="maternal-app-tests-"))
os.environ.setdefault("ADMIN_TOKEN", "offline-test-admin")
import app as server
from features import FEATURES


class AppAITests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.settings = dict(server.store.data["settings"])
        self.addCleanup(server.store.data["settings"].update, self.settings)
        server.store.data["settings"].update(ai_loop=False, ai_summary=False, ai_rerank=False)
        self.inputs = {f["name"]: f["default"] for f in FEATURES}
        self.inputs.update(SystolicBP=165, DiastolicBP=112, trimester=2)
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def assess(self):
        response = self.client.post("/api/assess", json=self.inputs)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_loop_off_does_not_call_ai(self):
        with patch.object(server, "enhance_guidance") as enhance:
            result = self.assess()
        enhance.assert_not_called()
        self.assertEqual(result["suggestions"]["loop"]["status"], "off")

    def test_loop_keeps_all_prediction_and_rule_outputs_identical(self):
        before = self.assess()
        server.store.data["settings"].update(ai_loop=True, ai_summary=True, ai_rerank=True)
        def enhance(cards, retriever, context):
            self.assertEqual(context["measurements"]["SystolicBP"], 165)
            self.assertEqual(context["trimester"], 2)
            self.assertEqual(context["model_class_score"], before["probability"])
            self.assertNotIn("identifier", context)
            return cards, dict(status="ready", rounds=1, search="keyword",
                               generated_cards=0, reason="")
        with patch.object(server, "enhance_guidance", side_effect=enhance), \
                patch.object(server, "llm_summary") as summary, \
                patch.object(server, "gemini_rerank") as rerank, \
                patch.dict(os.environ, {"GEMINI_API_KEY": "mock-only"}):
            after = self.assess()
        for field in ["probability", "band", "votes", "local", "flags",
                      "history_validation", "narrative", "counterfactuals", "ice"]:
            self.assertEqual(before[field], after[field], field)
        summary.assert_not_called()
        rerank.assert_not_called()
        self.assertTrue(any(f["level"] == "critical" for f in after["flags"]))

    def test_enabled_loop_without_key_returns_excerpts(self):
        server.store.data["settings"]["ai_loop"] = True
        result = self.assess()
        self.assertEqual(result["suggestions"]["loop"]["reason"], "key_missing")
        self.assertTrue(result["flags"])
        self.assertTrue(result["suggestions"]["cards"])

    def test_admin_setting_requires_authentication(self):
        self.assertEqual(self.client.put("/api/admin/settings", json={"ai_loop": True}).status_code, 401)
        with patch.object(server.store, "_save"), patch.object(server.store, "log"):
            response = self.client.put("/api/admin/settings", json={"ai_loop": True},
                                       headers={"X-Admin-Token": server.ADMIN_TOKEN})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ai_loop"])

    def test_pdf_import_preserves_page_metadata(self):
        from pypdf import PdfWriter
        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        writer.add_blank_page(width=300, height=300)
        writer.write(buffer)
        buffer.seek(0)
        from werkzeug.datastructures import FileStorage
        text, pages = server._extract(FileStorage(buffer, filename="source.pdf"))
        self.assertEqual([p["number"] for p in pages], [1, 2])
        self.assertEqual(text.strip(), "")


if __name__ == "__main__":
    unittest.main()
