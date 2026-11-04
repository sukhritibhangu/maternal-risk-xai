# Maternal risk project implementation stages

This report documents the college research prototype. The supplied Random Forest, its fourteen inputs and model parameters remain unchanged. The stages describe the implementation rather than measured clinical effectiveness.

## Stage 1 Integrating the given model and XAI with the frontend

Flask loads the saved model and validates measurements and obstetric history. The frontend presents the existing class score, per-tree outputs, explanation narrative, sensitivity scenarios and individual conditional expectation curves. Display bands are project conventions. Scenario values describe model behaviour rather than treatment targets. Form bounds are validation limits.

Prediction and explanation calculations remain in xai.py. Feature metadata and source-linked screening rules remain in features.py. The interface handles loading, errors and stale assessments. Retrieval does not retrain the predictor.

## Stage 2 Adding SHAP to the frontend

Local SHAP displays each input's signed contribution relative to the model baseline. Contributions and grouped drivers appear separately from coded rule severity. An additivity gap checks baseline plus contributions against prediction.

The Model tab compares mean absolute SHAP and impurity importance. Global SHAP uses an optional background CSV; otherwise it uses a disclosed synthetic background. Synthetic importance must not be reported as training-population evidence. SHAP explains the fitted model and does not establish causation.

## Stage 3 RAG approaches and the final method

The repository supports this progression. These are implementation approaches rather than a completed comparative benchmark.

| Approach | Behaviour and limitation |
| --- | --- |
| Tags and keywords | Driver queries and topic tags narrow TF-IDF search. Exact vocabulary can miss typos and equivalent wording. |
| Improved lexical baseline | Sentence-aware chunks, minimum lexical relevance, topic filtering, duplicate suppression and diversity improve selection. Tags alone cannot promote unrelated text. |
| Optional Gemini reranking | AI chooses among already retrieved passage IDs. It cannot recover evidence absent from the candidate pool. |
| Optional Gemini summary | AI summarizes retrieved evidence with citations. Generated claims still require support review. |
| Hybrid retrieval | TF-IDF and Gemini embeddings retrieve independently. Reciprocal Rank Fusion combines ranks without requiring matching tags. |
| Bounded AI evidence loop | AI plans questions, reviews verified evidence and may request one refined search. Server-owned sentence IDs insert exact quotations. |
| Improved Test retrieval | Local spelling recovery handles heaadache. Optional AI interpretation and hybrid search select exact verified passages, with one refinement at most. |

The final assessment approach is hybrid retrieval with bounded AI review when enabled. The lexical method remains the fallback. The evidence loop takes precedence over legacy summary and reranking switches.

Test retrieval is extractive: AI selects IDs and the server returns corresponding stored passages. It does not display generated medical answers. AI-assisted search uses only enabled admin-verified documents. Local mode follows the active knowledge-base filter and labels verification status. Interpretation and fallback status are visible.

Assessment guidance may contain generated suggestions alongside exact evidence. Valid citation IDs do not establish claim support. Administrator verification is a workflow status rather than independent clinical validation.

## Stage 4 Verified alert values and trimester context

Existing coded rules use recorded WHO and Government of India sources, with source links and qualifications. Rules operate separately from prediction and SHAP. Optional trimester changes haemoglobin interpretation for suggestions; it is not an additional model input. Missing trimester uses the disclosed first and third trimester cutoff. Labour context gates the childbirth fever rule. The separate source register records the criteria and limitations.

## Research evaluation and deployment

Compare lexical retrieval, spelling recovery, hybrid retrieval and AI review on held-out queries containing typos, synonyms, negation, numbers, conflicting evidence and evidence gaps. Measure Recall at k, precision, applicability errors, latency and provider calls. Review generated claim support independently. No measured superiority is claimed here.

Offline tests use synthetic sources and mocked provider responses. Live Gemini availability requires a separate check. The chatbot is archived in Future_Chatbot.md and excluded from active routes and assets. On the selected free Render setup, restore reviewed sources, verification and AI switches manually before demonstrations if storage resets.
