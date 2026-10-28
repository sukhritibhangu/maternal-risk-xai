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
| `chatbot.py` | Verified-document chat and signed assessment explanations |
| `seed_kb.py` | Starter guidance (loaded as **unverified**) |
| `static/index.html` | Frontend (Assess, Model, Knowledge base tabs) |
| `knowledge/store.json` | Live knowledge base (created on first run). `audit.log` records every admin action |

## Adding your book later
Knowledge base tab -> sign in -> Add a source -> upload the PDF/TXT, set tags, tick *Mark as verified*.
Then tick *Only use verified sources* and delete or disable the starter documents. Suggestions will then come only from the book.
Tags (feature names, or `general`) boost passages for matching drivers. Scanned PDFs need OCR first.
Enable the AI evidence loop or chatbot to use the implemented hybrid keyword/embedding retrieval.

## Optional Gemini features (Flash-Lite), controlled by the admin
With the AI evidence loop off, passage retrieval is local TF-IDF text search. Gemini is optional and has four switches in the Knowledge base tab:
1. **Write a plain-language summary** from the retrieved passages, with [S#] citations.
2. **Re-rank the retrieved passages**: Gemini may only pick among passages that were already retrieved; it cannot add or change text.

3. **AI evidence search and suggestions**: AI-generated questions, hybrid keyword/semantic retrieval, evidence review, one optional search retry and cited suggestion cards (details below). This takes precedence over switches 1 and 2.
4. **Assessment and document chatbot**: a separate Ask Matcare panel for existing assessment explanations and questions grounded in verified documents.

All are OFF by default, even if a key is set. They work only when the server has `GEMINI_API_KEY` (optionally `GEMINI_MODEL`, default `gemini-3.5-flash-lite`) AND an admin switches them on.
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

## AI evidence search and suggestion loop

The risk model, its inputs/parameters, SHAP calculations, screening criteria and decision
thresholds are unchanged. This feature operates only on evidence retrieval and presentation.

### Enable
1. Set a server-side `GEMINI_API_KEY` and, if needed, `GEMINI_MODEL` for a model supporting
   structured JSON responses. The existing generation-model default is retained.
2. Upload relevant sources and review/mark them verified in the Knowledge base tab.
3. Enable **Gemini: AI evidence search and suggestions**. It is OFF by default and
   takes precedence over the two legacy summary/rerank switches.

### How it works
- Existing flags and explanation groups determine the guidance areas, including areas
  where keyword search found nothing. Gemini creates a focused question for each area.
- Keyword TF-IDF search and Gemini semantic embeddings retrieve candidates independently.
  Reciprocal Rank Fusion combines their ranks; the AI loop does not require matching tags.
- Gemini reviews up to eight initial passages per area, drafts suggestions and selects
  short evidence-sentence IDs. The server inserts the original sentences and their parent citations;
  Gemini does not have to reproduce quotations or long passage hashes. It can request one refined search; there is at most one retry.
  Final review sees both old and new candidates (at most sixteen per retried area).
- Results preserve applicability notes, missing-evidence explanations and conflicting
  evidence. Conflicting evidence suppresses the generated suggestion for that area.
- The server rejects invalid groups, unknown/cross-area references, invented quotes,
  unsupported output shapes and incomplete provider responses. Evidence-ID and source-link checks
  do NOT prove that a generated claim follows from the text or establish clinical validity.
- Failure returns the ordinary source excerpts and existing alerts. Embedding-only failure
  keeps AI query/review working on keyword results and is explicitly shown in the UI.
- The loop has at most three generation calls (plan, review, optional final review), a
  55-second overall work budget and individual network timeouts of at most 15 seconds.
  Timeouts are best effort at the HTTP/socket layer; deployment timeouts should leave margin.

### Semantic index and privacy
`GEMINI_EMBEDDING_MODEL` defaults to `gemini-embedding-001`, requesting 768 dimensions.
Only verified, enabled passages are eligible for the AI loop. Their text is sent to Google
for embeddings; document vectors are cached in `DATA_DIR/embedding-cache.json` using
content/model hashes. The cache is written atomically and does not store assessment context
or query embeddings. It is keyed by the current text/model; removed or unverified documents
are excluded from retrieval even if an old vector remains in the cache.

Cold indexing is bounded to 128 missing passages per assessment, in batches of 32.
Large collections may report keyword-only/index-warming until subsequent requests complete
the index. No additional vector database or Python ML package is required.

When enabled, Google receives measurements, supplied trimester/labour context, existing
assessment flags, the model class score, generated queries and selected verified passages.
The original fallback may include unverified sources according to the existing admin setting,
but those sources are never supplied to the AI loop. Use synthetic data for demonstrations;
apply the project's existing provider/privacy approval requirements before real patient data.
No live provider request was used in the automated tests.

### Evidence display and PDF pages
Chunks preserve whole sentences where possible and retain nearby section headings.
New PDF uploads store one-based **PDF page positions**, which may differ from printed page
numbers. Older sources need re-uploading to gain page references. Text edits invalidate
imported page locations. AI highlights use the selected exact quotation rather than
automatically taking the paragraph's first sentence.

### Research evaluation
Keep a reviewed, held-out collection of assessment questions and relevant passages.
Compare baseline TF-IDF, hybrid retrieval and hybrid retrieval plus the AI loop using
Recall@k, Precision@3, supported-claim rate, applicability errors, evidence-gap handling,
latency and API usage. Choose retrieval settings on a separate development set; do not
report generated quotations/citation validity as a substitute for reviewed claim support.

### Verification
Run `python -m unittest discover -s . -p 'test_*.py' -v` and `node test_guidance_ui.cjs`.
The AI tests mock provider responses and embeddings, cover retries and malformed/fabricated
evidence, and check cache and fallback behaviour. API checks use the repository's saved model
to compare all prediction/explanation/rule outputs with the loop off and on.

Evidence selection uses server-provided sentence IDs to avoid rejecting otherwise useful
answers because Gemini paraphrased a quotation. Unknown IDs are still rejected. Multiple
sentences from one passage share one citation, and exact source wording stays visible.


## Assessment and document chatbot

The **Ask Matcare** panel answers application/assessment questions and questions about
verified, enabled documents. It uses the existing Gemini key. An administrator must
turn on **Gemini: assessment and document chatbot** in Knowledge base; this switch is
separate from the AI evidence loop and is off by default.

- Questions about documents use the same hybrid retrieval, without requiring tag matches.
- Assessment context is optional. After running an assessment, select **Include current
  assessment** (or the explicitly named assessment prompt). The server signs facts from
  the existing result; chat does not run, retrain, change, or call the risk model.
- Editing inputs or running a new assessment clears old chat context and conversation.
  Signed assessment context expires after one hour. Refreshing the page clears chat.
- Each answer paragraph must select known evidence-sentence IDs. The server inserts
  exact quotations, document links and PDF page positions. Medical-guidance paragraphs
  require document evidence; application facts explain software behaviour only.
- Missing evidence, provider failures and invalid responses produce a clear gap or source
  excerpts. Citation validation is structural: it does not establish that every generated
  claim is supported. Review applicability and claim support separately for research.
- Up to three recent question/answer pairs are sent as conversation context, never as
  authoritative evidence. Questions are limited to 1,000 characters. The endpoint allows
  20 requests per minute across this process and two concurrent chat requests. These are
  demo resource limits, not a distributed abuse-prevention system. Keep one worker.
- Each answer uses at most two generation calls and a 45-second work budget. Semantic
  embeddings share the existing verified-document cache. No new dependencies are required.

Questions, recent chat messages and opted-in assessment facts are sent to Gemini.
Verified source text may also be sent for retrieval/generation. The app does not save
conversation history to its knowledge store or audit log; browser tab memory retains it
until cleared/reloaded. Provider retention is governed by the configured Gemini service.
Use synthetic demonstrations and obtain the necessary privacy approval for real data.

The signing key is randomly generated at startup unless CHAT_CONTEXT_SECRET is set to a
strong private environment value. Restarting with a new key invalidates older contexts;
users can assess again. Signed tokens are not encrypted and should not be shared or logged.

### Storage on Render
Knowledge documents, verification status, AI switches and cached embeddings live under
DATA_DIR. Render's free service has ephemeral local storage: a restart/redeploy can reset
these to the unverified seed documents and disabled AI defaults. A successful deploy does
not mean the AI switches or document approvals survived. Use a persistent mounted disk
and set DATA_DIR to its mount path, or implement an external durable store. The included
render.yaml describes a paid Starter disk configuration; it does not upgrade an already
created free service. Never automatically mark seeded sources verified to bypass this.

### Chat verification
Run the Python unittest suite and node test_guidance_ui.cjs. test_chatbot.py covers
verified-only retrieval, server-owned citations, missing evidence, invalid outputs,
signed/expired/forged assessment contexts, request limits, and disabled-feature behaviour.
The provider is mocked in automated tests; validate live answer quality on a reviewed
research question set before reporting evaluation results.
