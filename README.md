# Maternal risk lens

Random Forest (14 inputs) + explainability + admin-governed RAG suggestions.

## Run
    pip install -r requirements.txt
    python app.py      
    open http://localhost:5000

Use the same scikit-learn version the model was trained with (pickles are version-sensitive; tested with 1.8.0).
Only load pickles you trust: unpickling executes code.

The admin token is `mysecret123` by default. Set an `ADMIN_TOKEN` environment variable to override it.

## What is where
| File | Purpose |
|---|---|
| `features.py` | Labels, ranges, healthy reference values, rule-based clinical flags. **Edit thresholds here.** |
| `xai.py` | SHAP (local + global), per-tree votes, what-if counterfactuals, ICE curves |
| `rag.py` | Knowledge store, chunking, retrieval, per-driver suggestion cards, optional LLM summary |
| `seed_kb.py` | Starter guidance (loaded as **unverified**) |
| `static/index.html` | Frontend (Assess, Model, Knowledge base tabs) |
| `knowledge/store.json` | Live knowledge base (created on first run). `audit.log` records every admin action |

## Adding your book later
Knowledge base tab -> sign in -> Add a source -> upload the PDF/TXT, set tags, tick *Mark as verified*.
Then tick *Only use verified sources* and delete or disable the starter documents. Suggestions will then come only from the book.
Tags (feature names, or `general`) boost passages for matching drivers. Scanned PDFs need OCR first.
To upgrade retrieval to embeddings, replace `Retriever` in `rag.py` (same `rebuild()` / `search()` interface).

## Optional Gemini features (Flash-Lite), controlled by the admin
Passage retrieval is always local (TF-IDF text search). Gemini is optional and has two separate switches in the Knowledge base tab:
1. **Write a plain-language summary** from the retrieved passages, with [S#] citations.
2. **Re-rank the retrieved passages**: Gemini may only pick among passages that were already retrieved; it cannot add or change text.

Both are OFF by default, even if a key is set. They work only when the server has `GEMINI_API_KEY` (optionally `GEMINI_MODEL`, default `gemini-3.5-flash-lite`) AND an admin switches them on.
Any failure (bad key, quota, odd reply) falls back to the local result. Privacy: when on, the risk percentage, flagged values and approved passages are sent to Google
for every assessment. On Google's free tier it may use submitted content to improve its products, and human reviewers may read it. Use a paid key and get privacy approval
before real patient data. Tested with a mock only, not against the live API.


## Assumptions to confirm
- Class 1 = high risk (matches behaviour on test cases).
- Units: BodyTemp in Fahrenheit, RBS mg/dL, Hb g/dL, HbA1c %. Inferred from the model's split thresholds.
- G/P/L/A/D = gravida/para/living children/abortions/child deaths. D's meaning is a guess.
- Global SHAP uses a synthetic population unless you provide `model/background.csv` (the 14 training columns).

## Deploy on Render
See render.yaml. Push to GitHub, create a Blueprint or Web Service, set ADMIN_TOKEN in the dashboard.
Build: `pip install -r requirements.txt`  Start: `gunicorn app:app --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT`
Keep ONE worker (the knowledge base is held in memory). Uploaded books live in DATA_DIR, which needs a persistent disk.

## Where the thresholds come from
Every rule in `features.py` carries its source (shown in the app next to each flag). Only values read on WHO or government pages are kept.
Removed for lack of a verified source: fever cutoffs, random sugar below 70 and 140 to 199, heart rate above 100, age rules, history rules.

## Assessment workspace and evidence links
Results are organized into Overview, Guidance, and Model details. Critical rule alerts stay above those views.
Guidance highlights quote the first sentence of each selected passage. Open a numbered citation to read the full
passage, its source label, and its admin verification status. The same source numbers are used by the optional
Gemini summary. Citation checks reject missing or unknown references; they do not establish that generated claims
are factually supported, so the source text remains available for review.

Retrieval requires a lexical match before a tag bonus can help a passage. Recognized feature tags keep driver
guidance separate from general-care guidance; untagged documents can still match by text. Selection diversifies
matches, suppresses exact duplicate text within a search, bounds long paragraphs to 165-word chunks, and uses
stable passage IDs. Flagged groups remain visible when matching evidence is missing. Re-ranking recomputes the
final exclusions before the general-care search.

The form shows loading and request-error states, prevents repeated submissions while busy, and marks results
as outdated after inputs change. Tree counts use the loaded model's returned outputs rather than a fixed label.

## Regression checks
Run `python -m unittest discover -s . -p 'test_*.py' -v` from the project directory.
The retrieval checks use small synthetic documents and mocked Gemini responses, not patient records or a live API.
