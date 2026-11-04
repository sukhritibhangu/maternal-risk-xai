# How AI is used in the project

Gemini supports evidence search and presentation. The supplied Random Forest predicts risk, SHAP explains its output and deterministic rules supply screening flags.

## Assessment guidance flow

```mermaid
flowchart TD
 A[Measurements and context] --> B[Existing model and SHAP]
 A --> C[Existing screening rules]
 B --> D[Guidance areas]
 C --> D
 D --> E{AI enabled and verified sources available}
 E -->|Yes| F[AI plans focused queries]
 K[Enabled admin-verified sources] --> G[Keyword and semantic retrieval]
 F --> G
 G --> H[AI reviews sources and selects sentence IDs]
 H --> I{Refinement needed}
 I -->|Once| J[Refined search and final review]
 J --> L[Validate IDs and insert exact quotations]
 I -->|No| L
 L --> M[Suggestions with citations applicability gaps and conflicts]
 E -->|No| N[Original source excerpts and existing alerts]
 F -. Failure .-> N
 H -. Invalid response .-> N
 J -. Failure .-> N
```

Assessment guidance permits at most three generation calls and one refinement, within a 55-second work budget. Embedding failure permits keyword search. Invalid references or provider failure return excerpt-based guidance. Valid IDs do not establish medical correctness or claim support.

## Test retrieval flow

```mermaid
flowchart TD
 Q[Question such as heaadache] --> S[Local spelling recovery]
 S --> T{AI-assisted mode enabled}
 T -->|Yes| V[Restrict to enabled admin-verified passages]
 V --> P[AI interprets query]
 P --> R[Hybrid candidate search]
 R --> U[AI selects supplied passage IDs]
 U --> W[Optional single refinement and final selection]
 W --> X[Server returns exact stored passages]
 T -->|No| Y[Local TF-IDF search using active-source filter]
 P -. Failure .-> Z[Verified local search]
 U -. Invalid selection .-> Z
 W -. Failure .-> Z
 Y --> X
 Z --> X
 X --> O[Show interpretation sources and evidence gaps]
```

Test retrieval permits at most three generation calls within a 40-second work budget. The server resolves IDs to stored text; no AI-written medical answer is displayed. Missing verified evidence produces a gap. Interpretation, corrections and fallback status are visible.

## Data and limitations

Google receives the query and eligible verified evidence. Assessment guidance also supplies measurements and context. Source embeddings are cached by content and model. Test retrieval does not search the web.

Local spelling recovery uses eligible source vocabulary and similarity matching. It can miss or misinterpret words. Number and simple-negation checks reduce rewrite errors but cannot prove semantic equivalence.

Gemini needs a valid server key and enabled evidence-loop switch. Free Render storage can reset sources and settings; restore them manually before demos. The chatbot is preserved only in Future_Chatbot.md.

Evaluate typo recovery, paraphrases, qualifiers, missing evidence, fabricated IDs and provider failure. Provider mocks establish control flow rather than live availability or clinical accuracy.
