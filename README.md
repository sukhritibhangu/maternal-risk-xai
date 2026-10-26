# Maternal risk lens

Random Forest (14 inputs) + explainability + admin-governed RAG suggestions.

## Run
    pip install -r requirements.txt
    python app.py      
    open http://localhost:5000

Use the same scikit-learn version the model was trained with (pickles are version-sensitive; tested with 1.8.0).
Only load pickles you trust: unpickling executes code.

Set a private `ADMIN_TOKEN` environment variable before deployment. If it is unset locally, the server creates a temporary random token and prints it to the startup console. On Render, an unset token is random and admin sign-in will not work reliably; add `ADMIN_TOKEN` in the service environment settings.

## What is where
| File | Purpose |
|---|---|
| `features.py` | Input bounds, confirmed feature meanings, scenario values, and source-linked screening flags. **Edit criteria here.** |
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


## Feature criteria and source review
The 14-feature audit, with official WHO/Government of India sources and a mathematical history check, is in [`docs/Verified_Maternal_Risk_Feature_Criteria.docx`](docs/Verified_Maternal_Risk_Feature_Criteria.docx). It is formatted as a repository document and includes a short methods note suitable for adapting into a research paper.

`features.py` keeps the dataset's numeric min/max solely as application input-validation bounds; they are not normal ranges, safe ranges, or clinical eligibility limits. The `ref` values used by model what-if scenarios are likewise sensitivity-analysis values, not clinical targets.

Implemented source-based screening criteria are deliberately limited to what the cited source and available inputs support:

- MoHFW/PMSMA age flags: first pregnancy and age under 20, or age over 35. These are information flags, not diagnoses.
- Pregnancy blood pressure: systolic at least 140 or diastolic at least 90 prompts a repeat measurement/antenatal assessment flag; the Government of India checklist specifies two consecutive readings. Systolic at least 160 or diastolic at least 110 produces a severe-range referral-context flag. The app cannot diagnose pre-eclampsia because it lacks repeat readings, gestational age, symptoms, and proteinuria.
- Systolic BP under 90 is a contextual screening sign from a Government of India training resource, not a stand-alone diagnosis.
- Random plasma glucose at least 200 mg/dL is an information flag only: WHO's general criterion also requires classic symptoms or hyperglycaemic crisis, which this form does not collect. The Indian GDM criterion is a timed 2-hour 75 g OGTT result, so it is not inferred from the app's random-glucose field.
- Haemoglobin flags use the WHO trimester cutoffs: under 11.0 g/dL in trimester 1 and 3, and under 10.5 g/dL in trimester 2. If trimester is omitted, the app states that it uses the first/third trimester cutoff. The separate HbA1c input has no alert rule: WHO advises against routine HbA1c measurement for GDM.
- Heart rate over 120/min and respiratory rate over 20/min are contextual information flags from Government of India maternal near-miss criteria. They are not presented as universal pregnancy reference intervals.
- Temperature at least 38°C (100.4°F) is considered only when “In labour or childbirth” is selected, following the Government of India checklist context.

No WHO/Indian-government source was verified for a universal pregnancy-specific normal range for heart rate, respiratory rate, random blood sugar, or HbA1c, or for maximum values of the obstetric history counts. The project does not invent those ranges. `G`, `P`, `L`, `A`, and `D` meanings are the project's confirmed definitions; the source audit documents which relationships can and cannot be mathematically checked.

### Obstetric history arithmetic
For an input representing a current pregnancy, with `G` including the current pregnancy, `P` counting prior pregnancies ending in delivery, and `A` counting prior pregnancy losses as mutually exclusive pregnancy outcomes, the necessary consistency condition is `P + A + 1 <= G`. Equality means those prior outcome categories account for all prior pregnancies. A smaller left side means at least one outcome is unclassified or the history may be incomplete; it is not automatically an error. A greater left side is rejected as inconsistent under those definitions. `L` and `D` are validated as nonnegative whole-number counts, but are not reconciled against `P`: `P` counts pregnancy events, while `L` and `D` count children, and the form has no total-children-born field.

## Remaining model assumptions
- Class 1 = high risk (matches behaviour on test cases).
- Units: BodyTemp in Fahrenheit, RBS mg/dL, Hb g/dL, HbA1c %. These are the project/model input units.
- Global SHAP uses a synthetic population unless you provide `model/background.csv` (the 14 training columns).

## Deploy on Render
See render.yaml. Push to GitHub, create a Blueprint or Web Service, set ADMIN_TOKEN in the dashboard.
Build: `pip install -r requirements.txt`  Start: `gunicorn app:app --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT`
Keep ONE worker (the knowledge base is held in memory). Uploaded books live in DATA_DIR, which needs a persistent disk.

## Where the criteria come from
Every rule in `features.py` carries an official source URL and a context/limitation note for the interface. The complete feature-by-feature verification matrix and bibliography are in [`docs/Verified_Maternal_Risk_Feature_Criteria.docx`](docs/Verified_Maternal_Risk_Feature_Criteria.docx). Values without an applicable verified WHO or Government of India criterion are explicitly marked “unable to verify” in that document and are not implemented as clinical ranges.

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
