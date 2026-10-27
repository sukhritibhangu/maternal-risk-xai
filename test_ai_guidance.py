"""Synthetic, offline regression checks for the AI evidence loop."""
import copy
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="maternal-ai-tests-"))
import ai_guidance as ai
import rag
from test_rag import Store, doc


def vector(x, y=0):
    return [x, y] + [0] * (ai.DIMENSIONS - 2)


def card():
    return dict(group="Haemoglobin", reasons=[], shap=.1, level="info",
                targets=[], passages=[])


def review(passage, **changes):
    row = dict(group="Haemoglobin", suggestion="Discuss the source guidance with a clinician.",
               applicability="Pregnancy context; trimester is not supplied.", gap="",
               conflict=False, retry_query="",
               supports=[dict(evidence_id="E1")])
    row.update(changes)
    return {"cards": [row]}


class EvidenceLoopTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "mock-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.r = rag.Retriever(Store([
            doc("a", "Anaemia guidance requires clinical review and the correct pregnancy context.", ["HB"]),
            doc("b", "Unverified text must never be sent to the AI.", ["HB"], verified=False)]))
        self.p = next(p for p in self.r.chunks if p["verified"])
        self.plan = {"queries": [{"group": "Haemoglobin", "query": "anaemia guidance pregnancy"}]}

    def run_loop(self, responses, retriever=None):
        with patch.object(ai, "_generate", side_effect=responses) as generate, \
                patch.object(ai.HybridSearch, "_document_vectors", side_effect=ai.Unavailable("mock_offline")):
            result = ai.enhance_guidance([card()], retriever or self.r, {"trimester": None})
        return result, generate

    def test_ready_with_exact_quote_and_server_citation(self):
        (cards, status), calls = self.run_loop([self.plan, review(self.p)])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["rounds"], 1)
        self.assertEqual(status["search"], "keyword")
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(cards[0]["highlights"][0]["text"], self.p["text"])
        self.assertEqual(cards[0]["passages"][0]["citation"], "S1")
        self.assertEqual(cards[0]["ai_guidance"]["suggestion"],
                         "Discuss the source guidance with a clinician.")
        payloads = json.dumps([call.args[1] for call in calls.call_args_list])
        self.assertNotIn("Unverified text", payloads)

    def test_one_retry_then_final_review(self):
        initial = review(self.p, suggestion="", supports=[], gap="Need a more specific passage.",
                         retry_query="anaemia pregnancy monitoring")
        (cards, status), calls = self.run_loop([self.plan, initial, review(self.p)])
        self.assertEqual(status["rounds"], 2)
        self.assertEqual(calls.call_count, 3)
        self.assertTrue(calls.call_args_list[-1].args[1]["final_search"])
        self.assertTrue(cards[0]["ai_guidance"]["suggestion"])

    def test_final_review_cannot_request_a_third_round(self):
        first = review(self.p, retry_query="anaemia pregnancy")
        (_, status), calls = self.run_loop([self.plan, first, first])
        self.assertEqual(status["status"], "fallback")
        self.assertEqual(calls.call_count, 3)

    def test_unknown_evidence_ids_and_legacy_quotes_fall_back(self):
        for support in [
            dict(evidence_id="E999"),
            dict(evidence_id=True),
            dict(evidence_id=["E1"]),
            dict(passage_id=self.p["chunk_id"], quote=self.p["text"])]:
            with self.subTest(support=support):
                original = card()
                (cards, status), _ = self.run_loop([self.plan, review(self.p, supports=[support])])
                self.assertEqual(cards, [original])
                self.assertEqual(status["reason"], "invalid_evidence")

    def test_quote_is_inserted_by_server_even_if_ai_adds_paraphrased_text(self):
        result = review(self.p, supports=[dict(evidence_id="E1", quote="An invented paraphrase.")])
        (cards, status), _ = self.run_loop([self.plan, result])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(cards[0]["highlights"][0]["text"], self.p["text"])
        self.assertNotIn("invented paraphrase", json.dumps(cards))

    def test_multiple_sentences_share_one_correct_parent_citation(self):
        source = doc("decimal", "Haemoglobin below 10.5 g/dL needs context. Review the pregnancy stage and source qualifications.")
        retriever = rag.Retriever(Store([source]))
        result = review(self.p, supports=[dict(evidence_id="E1"), dict(evidence_id="E2")])
        (cards, status), _ = self.run_loop([self.plan, result], retriever)
        self.assertEqual(status["status"], "ready")
        self.assertEqual(len(cards[0]["passages"]), 1)
        highlights = cards[0]["highlights"]
        self.assertEqual([h["citation"] for h in highlights], ["S1", "S1"])
        self.assertEqual(highlights[0]["text"], "Haemoglobin below 10.5 g/dL needs context.")
        self.assertEqual(highlights[1]["text"], "Review the pregnancy stage and source qualifications.")

    def test_missing_evidence_and_conflicts_remain_visible(self):
        for changes in [
            dict(suggestion="", supports=[], gap="No applicable passage was found."),
            dict(conflict=True, gap="The passages conflict.")]:
            (cards, status), _ = self.run_loop([self.plan, review(self.p, **changes)])
            self.assertEqual(status["status"], "ready")
            self.assertEqual(cards[0]["passages"], [])
            self.assertEqual(cards[0]["ai_guidance"]["suggestion"], "")
            self.assertTrue(cards[0]["ai_guidance"]["gap"])

    def test_missing_key_or_verified_sources_makes_no_provider_call(self):
        with patch.object(ai, "_generate") as generate, patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            _, status = ai.enhance_guidance([card()], self.r, {})
            self.assertEqual(status["reason"], "key_missing")
            generate.assert_not_called()
        self.r.chunks = [p for p in self.r.chunks if not p["verified"]]
        with patch.object(ai, "_generate") as generate:
            _, status = ai.enhance_guidance([card()], self.r, {})
            self.assertEqual(status["reason"], "no_verified_sources")
            generate.assert_not_called()

    def test_provider_or_malformed_output_preserves_original_cards(self):
        for response in [ai.Unavailable("timeout"), {}, {"queries": []},
                         {"queries": [{"group": [], "query": "text"}]}]:
            (cards, status), _ = self.run_loop([response])
            self.assertEqual(cards, [card()])
            self.assertEqual(status["status"], "fallback")

    def test_reviewer_cannot_drop_or_invent_groups(self):
        for rows in [[], [dict(review(self.p)["cards"][0], group="invented")], [None]]:
            (_, status), _ = self.run_loop([self.plan, {"cards": rows}])
            self.assertEqual(status["status"], "fallback")

    def test_empty_shap_area_is_kept_for_semantic_retrieval(self):
        r = rag.Retriever(Store([]))
        cards = rag.build_cards([], [dict(feature="HB", shap=.1)], r, include_empty=True)
        self.assertEqual(cards[0]["group"], "Haemoglobin")
        self.assertEqual(cards[0]["evidence_status"], "missing")
        self.assertEqual(cards[-1]["group"], "General care and danger signs")

    def test_pdf_page_and_sentence_boundaries_are_preserved(self):
        d = doc("pages", "unused")
        d["pages"] = [dict(number=3, text="Anaemia\n\n" + "A complete sentence about monitoring. " * 50),
                      dict(number=4, text="Different page content with clinical context.")]
        chunks = rag.chunk(d)
        self.assertEqual({p["page"] for p in chunks}, {3, 4})
        self.assertEqual(chunks[0]["section"], "Anaemia")
        self.assertTrue(all(len(p["text"].split()) <= 165 for p in chunks))
        self.assertTrue(all(p["text"].endswith(".") for p in chunks[1:]))

    def test_text_edit_removes_stale_pdf_pages(self):
        d = doc("a", "Old evidence.")
        d["pages"] = [dict(number=1, text="Old evidence.")]
        s = rag.KnowledgeStore.__new__(rag.KnowledgeStore)
        s.data = {"docs": [d]}
        with patch.object(s, "_save"), patch.object(s, "log"):
            s.update("a", {"text": "New evidence."})
        self.assertNotIn("pages", d)
        self.assertEqual(rag.chunk(d)[0]["text"], "New evidence.")


class HybridTests(unittest.TestCase):
    def search(self, docs):
        chunks = [p for d in docs for p in rag.chunk(d)]
        with patch.object(ai.HybridSearch, "_document_vectors", return_value=ai.np.asarray([vector(1)] * len(chunks))):
            return ai.HybridSearch(chunks, time.monotonic() + 50)

    def test_semantics_can_recover_a_passage_without_keyword_overlap_or_tag_match(self):
        search = self.search([doc("a", "Low red cell pigment concentration.", ["general"])])
        with patch.object(ai, "_vectors", return_value=ai.np.asarray([vector(1)])):
            hits = search.search_many({"Haemoglobin": ["anaemia"]})
        self.assertEqual(hits["Haemoglobin"][0]["doc_id"], "a")
        self.assertTrue(search.semantic)

    def test_embedding_failure_uses_keyword_results(self):
        search = self.search([doc("a", "Anaemia monitoring during pregnancy.")])
        with patch.object(ai, "_vectors", side_effect=ai.Unavailable("provider_unavailable")):
            hits = search.search_many({"Haemoglobin": ["anaemia"]})
        self.assertEqual(hits["Haemoglobin"][0]["doc_id"], "a")
        self.assertFalse(search.semantic)

    def test_hybrid_deduplicates_and_excludes_unverified_sources(self):
        docs = [doc("a", "Anaemia pregnancy monitoring."),
                doc("b", "Anaemia pregnancy monitoring."),
                doc("c", "Unverified anaemia text.", verified=False)]
        chunks = [p for d in docs for p in rag.chunk(d)]
        with patch.object(ai.HybridSearch, "_document_vectors", return_value=ai.np.asarray([vector(1), vector(1)])):
            search = ai.HybridSearch(chunks, time.monotonic() + 50)
        with patch.object(ai, "_vectors", return_value=ai.np.asarray([vector(1)])):
            hits = search.search_many({"Haemoglobin": ["anaemia"]})
        self.assertEqual(len(hits["Haemoglobin"]), 1)
        self.assertNotEqual(hits["Haemoglobin"][0]["doc_id"], "c")

    def test_empty_and_corrupt_embedding_responses_are_rejected(self):
        for result in [{}, {"embeddings": [{"values": vector(0)}]},
                       {"embeddings": [{"values": [float("nan")] * ai.DIMENSIONS}]}]:
            with patch.object(ai, "_post", return_value=result):
                with self.assertRaises(ai.Unavailable):
                    ai._vectors(["query"], "RETRIEVAL_QUERY", time.monotonic() + 50)

    def test_document_cache_is_reused_but_queries_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ai, "DATA_DIR", tmp), \
                patch.object(ai, "_vectors", return_value=ai.np.asarray([vector(1)])) as embed:
            chunks = rag.chunk(doc("a", "Anaemia pregnancy monitoring."))
            first = ai.HybridSearch(chunks, time.monotonic() + 50)
            second = ai.HybridSearch(chunks, time.monotonic() + 50)
            self.assertEqual(embed.call_count, 1)
            first.search_many({"Haemoglobin": ["patient-sensitive-query"]})
            self.assertEqual(embed.call_count, 2)
            with open(os.path.join(tmp, "embedding-cache.json"), encoding="utf-8") as source:
                cache = source.read()
            self.assertNotIn("patient-sensitive-query", cache)
            self.assertNotIn("Anaemia", cache)
            self.assertTrue(second.semantic)

    def test_generate_rejects_truncation_or_non_json(self):
        for result in [
            {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{}"}]}}]},
            {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not JSON"}]}}]},
            {"candidates": []}]:
            with patch.object(ai, "_post", return_value=result):
                with self.assertRaises(ai.Unavailable):
                    ai._generate("system", {}, ai.PLAN_SCHEMA, time.monotonic() + 50)

    def test_generate_sends_structured_schema_and_separates_untrusted_payload(self):
        response = {"candidates": [{"finishReason": "STOP",
                    "content": {"parts": [{"text": '{"queries":[]}'}]}}]}
        with patch.object(ai, "_post", return_value=response) as post:
            result = ai._generate("trusted system", {"source": "untrusted text"},
                                  ai.PLAN_SCHEMA, time.monotonic() + 50)
        self.assertEqual(result, {"queries": []})
        body = post.call_args.args[1]
        self.assertEqual(body["generationConfig"]["responseSchema"], ai.PLAN_SCHEMA)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("untrusted text", body["systemInstruction"]["parts"][0]["text"])
        self.assertIn("untrusted text", body["contents"][0]["parts"][0]["text"])

    def test_transport_keeps_key_out_of_url(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b'{"ok":true}'
        with patch.dict(os.environ, {"GEMINI_API_KEY": "private-test-key"}), \
                patch.object(ai.urllib.request, "urlopen", return_value=Response()) as request:
            self.assertEqual(ai._post("models/test:generateContent", {}, time.monotonic() + 50),
                             {"ok": True})
            self.assertNotIn("private-test-key", request.call_args.args[0].full_url)
            self.assertLessEqual(request.call_args.kwargs["timeout"], 15)


if __name__ == "__main__":
    unittest.main()
