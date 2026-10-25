"""Regression checks for retrieval quality, citations and optional AI fallback."""
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('DATA_DIR', tempfile.mkdtemp(prefix='maternal-rag-tests-'))
import rag


def doc(id, text, tags=None, title='Guidance', verified=True):
    return dict(id=id, title=title, source='Example source', tags=tags or [], text=text,
                verified=verified, enabled=True)


class Store:
    def __init__(self, docs): self.docs = docs
    def active_docs(self): return self.docs


class RetrievalTests(unittest.TestCase):
    def test_tag_alone_cannot_promote_unrelated_text(self):
        r = rag.Retriever(Store([doc('a', 'Banana orchard seasonal harvesting.', ['HB'])]))
        self.assertEqual(r.search('anaemia haemoglobin iron', tags=['HB']), [])

    def test_real_match_has_evidence_metadata(self):
        r = rag.Retriever(Store([doc('a', 'Anaemia haemoglobin iron monitoring.', ['HB'])]))
        hits = r.search('anaemia haemoglobin', tags=['HB'])
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0]['lexical_score'], 0)
        self.assertEqual(hits[0]['matched_tags'], ['HB'])
        self.assertEqual(r.search('anaemia', exclude={rag.passage_key(hits[0])}), [])

    def test_known_topic_tags_keep_driver_and_general_evidence_separate(self):
        r = rag.Retriever(Store([doc('a', 'Anaemia iron pregnancy care.', ['HB']),
                                 doc('b', 'Anaemia iron pregnancy care.', ['general'])]))
        self.assertEqual([p['doc_id'] for p in r.search('anaemia iron', tags=['HB'])], ['a'])
        self.assertEqual([p['doc_id'] for p in r.search('anaemia iron', tags=['general'])], ['b'])

    def test_long_paragraph_is_bounded_and_ids_stable(self):
        d = doc('a', ' '.join(['monitoring'] * 500))
        a, b = rag.chunk(d), rag.chunk(d)
        self.assertEqual(a, b)
        self.assertTrue(all(len(p['text'].split()) <= 165 for p in a))
        self.assertEqual(sum(len(p['text'].split()) for p in a), 500)

    def test_empty_vocabulary_is_safe(self):
        r = rag.Retriever(Store([doc('a', 'the and or', title='the')]))
        self.assertEqual(r.search('pregnancy'), [])

    def test_selected_exact_duplicate_text_is_suppressed(self):
        r = rag.Retriever(Store([doc('a', 'Blood pressure monitoring pregnancy.'),
                                 doc('b', 'Blood pressure monitoring pregnancy.')]))
        self.assertEqual(len(r.search('blood pressure', k=2)), 1)

    def test_borderline_alert_retained_with_no_sources(self):
        r = rag.Retriever(Store([]))
        flags = [dict(group='Haemoglobin', level='borderline', text='Test rule',
                      source_name='Source', source_url='https://example.org', target='', note='')]
        cards = rag.build_cards(flags, [], r)
        self.assertEqual(cards[0]['evidence_status'], 'missing')
        self.assertEqual(cards[0]['passages'], [])

    def test_citations_and_summary_validation(self):
        cards = [dict(passages=[dict(rag.chunk(doc('a', 'Evidence sentence. More detail.'))[0])])]
        with patch.object(rag, '_gemini', return_value='Summary [S99].'):
            self.assertIsNone(rag.llm_summary('context', cards))
        with patch.object(rag, '_gemini', return_value='Summary without citation.'):
            self.assertIsNone(rag.llm_summary('context', cards))
        with patch.object(rag, '_gemini', return_value='Summary [S1].'):
            self.assertEqual(rag.llm_summary('context', cards), 'Summary [S1].')
        self.assertEqual(cards[0]['highlights'][0]['citation'], 'S1')
        self.assertEqual(cards[0]['highlights'][0]['text'], 'Evidence sentence.')

    def test_bad_rerank_falls_back_and_rejects_boolean_index(self):
        ps = {'Blood pressure': [dict(title='One', text='Evidence'), dict(title='Two', text='More evidence')]}
        for response in ['[]', 'null', '{"Blood pressure": [true, 99]}', 'not JSON']:
            with patch.object(rag, '_gemini', return_value=response):
                self.assertIsNone(rag.gemini_rerank(ps, 'context'))

    def test_general_card_uses_final_rerank_exclusions(self):
        p1 = dict(rag.chunk(doc('a', 'First evidence.'))[0])
        p2 = dict(rag.chunk(doc('b', 'Second evidence.'))[0])
        class Retriever:
            def search(self, query, **kwargs):
                if kwargs.get('tags') == ['general']:
                    return [p for p in [p1, p2] if rag.passage_key(p) not in kwargs['exclude']][:1]
                return [dict(p1), dict(p2)]
        cards = rag.build_cards([], [dict(feature='HB', shap=.10)], Retriever(),
                                rerank=lambda *a: {'Haemoglobin': [1]})
        keys = [rag.passage_key(p) for c in cards for p in c['passages']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual([p['citation'] for c in cards for p in c['passages']], ['S1', 'S2'])


if __name__ == '__main__': unittest.main()
