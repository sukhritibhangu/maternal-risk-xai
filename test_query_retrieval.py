"""Source-only query search regression checks with mocked AI responses."""
import os
import tempfile
import unittest
from unittest.mock import patch
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="query-tests-"))
os.environ.setdefault("ADMIN_TOKEN", "test-only-token")
import query_retrieval as qr
import rag
from test_rag import Store, doc

class QueryTests(unittest.TestCase):
    def setUp(self):
        store = Store([doc("verified", "Headache in pregnancy requires assessment of associated symptoms.", ["SystolicBP"]),
                       doc("unverified", "Headache unverified secret source text.", verified=False),
                       doc("other", "Nutrition includes a varied diet.", ["general"])])
        store.data = {"settings": {"only_verified": False}}
        self.r = rag.Retriever(store)
        self.plan = {"query": "headache pregnancy"}
        self.selection = {"selected_ids": ["P1"], "retry_query": ""}

    def run_ai(self, responses):
        with patch.object(qr, "_generate", side_effect=responses) as gen, patch.object(qr.HybridSearch, "_document_vectors", side_effect=qr.Unavailable("offline")):
            result = qr.search_sources("heaadache", self.r, True)
        return result, gen

    def test_typo_recovery_without_provider(self):
        with patch.object(qr, "_generate") as gen:
            result = qr.search_sources("heaadache", self.r)
        self.assertEqual(result["search"]["interpreted_query"], "headache")
        self.assertTrue(all(p["text"] in [c["text"] for c in self.r.chunks] for p in result["results"]))
        gen.assert_not_called()

    def test_unknown_word_does_not_fabricate_answer(self):
        self.assertEqual(qr.search_sources("xyzqzabc", self.r)["results"], [])

    def test_spelling_preserves_negation_numbers_and_units(self):
        text, corrections = qr.correct_query("no heaadache before 20 weeks 140 mmHg", self.r.chunks)
        self.assertEqual(text, "no headache before 20 weeks 140 mmHg")

    def test_ai_returns_exact_verified_source(self):
        result, gen = self.run_ai([self.plan, self.selection])
        self.assertEqual(result["search"]["ai_status"], "ready")
        self.assertEqual(result["results"][0]["text"], self.r.chunks[0]["text"])
        self.assertTrue(all(p["verified"] for p in result["results"]))
        self.assertNotIn("secret source", str(gen.call_args_list))
        self.assertEqual(result["results"][0]["citation"], "S1")

    def test_no_verified_sources_never_calls_ai(self):
        self.r.chunks = [p for p in self.r.chunks if not p["verified"]]
        result, gen = self.run_ai([])
        self.assertEqual(result["search"]["reason"], "no_verified_sources")
        self.assertEqual(result["results"], [])
        gen.assert_not_called()

    def test_provider_failure_uses_verified_local_sources(self):
        result, _ = self.run_ai([qr.Unavailable("provider_unavailable")])
        self.assertEqual(result["search"]["ai_status"], "fallback")
        self.assertTrue(result["results"])
        self.assertTrue(all(p["verified"] for p in result["results"]))
        self.assertEqual(result["search"]["interpreted_query"], "headache")

    def test_unknown_or_duplicate_ids_are_rejected(self):
        for ids in [["P999"], ["P1", "P1"], [True], [["P1"]]]:
            with self.subTest(ids=ids):
                result, _ = self.run_ai([self.plan, {"selected_ids": ids, "retry_query": ""}])
                self.assertEqual(result["search"]["reason"], "invalid_evidence")

    def test_one_retry_then_final_review(self):
        result, gen = self.run_ai([self.plan, {"selected_ids": [], "retry_query": "headache symptoms"}, self.selection])
        self.assertEqual(result["search"]["rounds"], 2)
        self.assertEqual(gen.call_count, 3)
        self.assertTrue(gen.call_args.args[1]["final_search"])

    def test_third_round_rejected(self):
        retry = {"selected_ids": [], "retry_query": "headache symptoms"}
        result, gen = self.run_ai([self.plan, retry, retry])
        self.assertEqual(result["search"]["ai_status"], "fallback")
        self.assertEqual(gen.call_count, 3)

    def test_ai_can_report_evidence_gap(self):
        result, _ = self.run_ai([self.plan, {"selected_ids": [], "retry_query": ""}])
        self.assertEqual(result["search"]["ai_status"], "ready")
        self.assertEqual(result["results"], [])

    def test_ai_prose_never_becomes_source_text(self):
        result, _ = self.run_ai([self.plan, dict(self.selection, answer="Invented medical advice")])
        self.assertNotIn("Invented medical advice", str(result))
        self.assertEqual(result["results"][0]["text"], self.r.chunks[0]["text"])

    def test_rewrite_cannot_drop_numeric_or_negative_constraint(self):
        for query in ["headache 20 weeks", "no headache"]:
            with self.assertRaises(qr.Unavailable):
                qr.valid_query(query, "no headache 20 weeks")

    def test_malformed_plan_falls_back(self):
        result, _ = self.run_ai([{"query": ["headache"]}])
        self.assertEqual(result["search"]["ai_status"], "fallback")

class AdminSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.module = app
        cls.client = app.app.test_client()
        cls.headers = {"X-Admin-Token": app.ADMIN_TOKEN}

    def test_auth_and_input_validation(self):
        self.assertEqual(self.client.post("/api/admin/search", json={"query":"headache"}).status_code,401)
        for body in [{}, [], {"query":1}, {"query":""}, {"query":"x"*351}]:
            self.assertEqual(self.client.post("/api/admin/search",json=body,headers=self.headers).status_code,400)

    def test_success_has_no_store_and_respects_toggle(self):
        with patch.object(self.module,"ai_on",return_value=True), patch.object(self.module,"search_sources",return_value={"results":[]}) as search:
            response=self.client.post("/api/admin/search",json={"query":"heaadache","use_ai":True},headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers["Cache-Control"],"no-store")
        self.assertTrue(search.call_args.kwargs["use_ai"])

    def test_chatbot_is_not_registered_or_loaded(self):
        self.assertEqual(self.client.post("/api/chat",json={}).status_code,404)
        html=self.client.get("/").get_data(as_text=True)
        self.assertNotIn("chat.js",html)
        self.assertNotIn("chat.css",html)
        self.assertNotIn("chatbot",self.module.store.data["settings"])

if __name__ == "__main__":
    unittest.main()
