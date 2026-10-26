"""Maternal risk XAI + admin-governed RAG. Run:  python app.py   ->  http://localhost:5000"""
import hmac, io, os, secrets, warnings
import joblib
from flask import Flask, jsonify, request, send_from_directory
from features import FEATURES, BY_NAME, flags_for
from xai import Explainer, band, narrative
from rag import KnowledgeStore, Retriever, build_cards, llm_summary, gemini_rerank

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

model = joblib.load(os.path.join(HERE, "model", "random_forest_model.pkl"))
bg = os.path.join(HERE, "model", "background.csv")  # optional: drop your training data here for true global SHAP
xai = Explainer(model, bg if os.path.exists(bg) else None)
store = KnowledgeStore()
retriever = Retriever(store)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_urlsafe(24)
    if os.environ.get("RENDER"):
        print("ADMIN_TOKEN is not set on this host, so a temporary random token is in use and admin sign-in will not work. Set ADMIN_TOKEN.", flush=True)
    else:
        print(f"ADMIN_TOKEN is not set; temporary local admin token: {ADMIN_TOKEN}", flush=True)


def ai_on(flag):  # an AI feature runs only if the server has a key AND an admin has switched it on
    return bool(os.environ.get("GEMINI_API_KEY")) and bool(store.data["settings"].get(flag))


def admin_only(fn):
    def w(*a, **k):
        if not hmac.compare_digest(request.headers.get("X-Admin-Token", ""), ADMIN_TOKEN):
            return jsonify(error="Admin token missing or wrong."), 401
        return fn(*a, **k)
    w.__name__ = fn.__name__
    return w


def parse_inputs(d):
    v = {}
    for f in FEATURES:
        try:
            x = float(d[f["name"]])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{f['label']} is required and must be a number.")
        if not f["min"] <= x <= f["max"]:
            raise ValueError(f"{f['label']} must be between {f['min']} and {f['max']} {f['unit']}.")
        if f["name"] in {"G", "P", "L", "A", "D"} and not x.is_integer():
            raise ValueError(f"{f['label']} must be a whole-number count.")
        v[f["name"]] = x
    # For a current pregnancy, G includes the current pregnancy. If P counts
    # prior pregnancies ending in delivery and A counts prior pregnancy losses,
    # their exclusive counts cannot exceed all prior pregnancies.
    if v["P"] + v["A"] + 1 > v["G"]:
        raise ValueError(
            "The obstetric counts do not reconcile: for a current pregnancy, "
            "G must be at least P + A + 1. Check the definitions and entries."
        )
    return v


def obstetric_history_check(v):
    """Return conditional G/P/A arithmetic and the L/D limitation."""
    expected_min = v["P"] + v["A"] + 1
    complete = expected_min == v["G"]
    if complete:
        status = "consistent"
        message = (
            "G/P/A arithmetic is consistent under the current-pregnancy count "
            "definitions. L and D are whole-number counts; this form has no "
            "total-children-born field to reconcile them against."
        )
    else:
        status = "review"
        message = (
            "P + A + 1 is below G. At least one previous outcome may be "
            "unclassified, or the history may be incomplete. Review the record. "
            "L and D are not compared with P because P counts pregnancy events "
            "while L and D count children."
        )
    return dict(status=status, message=message, formula="P + A + 1 <= G")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(features=FEATURES, decision_threshold=0.5, bands=[0.35, 0.65],
                   llm=ai_on("ai_summary"), gemini_key=bool(os.environ.get("GEMINI_API_KEY")), only_verified=store.data["settings"]["only_verified"])


@app.post("/api/assess")
def assess():
    body = request.get_json(force=True) or {}
    try:
        v = parse_inputs(body)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    try:  # optional, used ONLY by the anaemia rule (not a model input)
        tri = int(body.get("trimester")) if str(body.get("trimester", "")).strip() in ("1", "2", "3") else None
    except (TypeError, ValueError):
        tri = None
    local = xai.local(v)
    p = local["prediction"]
    intrapartum = body.get("intrapartum") is True
    flags = flags_for(v, tri, intrapartum=intrapartum)
    history_validation = obstetric_history_check(v)
    top = [c["feature"] for c in local["contribs"][:4]]
    line = f"risk {p:.0%}; " + "; ".join(f["text"] for f in flags)
    cards = build_cards(flags, local["contribs"], retriever, rerank=gemini_rerank if ai_on("ai_rerank") else None, patient_line=line)
    summary = llm_summary(line, cards) if ai_on("ai_summary") else None
    return jsonify(
        probability=p, band=band(p), votes=xai.votes(v), local=local, flags=flags,
        history_validation=history_validation,
        narrative=narrative(p, local, flags), counterfactuals=xai.counterfactuals(v, flags),
        ice=xai.ice(v, top), suggestions=dict(cards=cards, summary=summary,
                                              summary_status=("ready" if summary else "unavailable" if ai_on("ai_summary") else "off"),
                                              ai=dict(summary=bool(summary), rerank=any(c.get("ai_ranked") for c in cards)),
                                              only_verified=store.data["settings"]["only_verified"]))


@app.get("/api/global")
def global_view():
    return jsonify(xai.global_view())


# ---------- admin
@app.post("/api/admin/login")
@admin_only
def login():
    return jsonify(ok=True)


@app.get("/api/admin/kb")
@admin_only
def kb_list():
    return jsonify(docs=store.data["docs"], settings=store.data["settings"], chunks=len(retriever.chunks), gemini_key=bool(os.environ.get("GEMINI_API_KEY")))


def _extract(file):
    name, raw = file.filename.lower(), file.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)
    return raw.decode("utf-8", errors="ignore")


@app.post("/api/admin/kb")
@admin_only
def kb_add():
    f = request.form if request.files or request.form else (request.get_json(silent=True) or {})
    text = f.get("text", "")
    if "file" in request.files:
        text = _extract(request.files["file"]) or text
    title = f.get("title") or (request.files["file"].filename if "file" in request.files else "")
    if not text.strip() or not title.strip():
        return jsonify(error="A title and some text (or a .txt/.md/.pdf file) are required."), 400
    tags = [t for t in str(f.get("tags", "")).split(",")] if not isinstance(f.get("tags"), list) else f["tags"]
    d = store.add(title, f.get("source", "Admin upload"), tags, text, verified=str(f.get("verified", "")).lower() in ("1", "true", "on"))
    retriever.rebuild()
    return jsonify(d), 201


@app.patch("/api/admin/kb/<doc_id>")
@admin_only
def kb_patch(doc_id):
    d = store.update(doc_id, request.get_json(force=True))
    retriever.rebuild()
    return (jsonify(d), 200) if d else (jsonify(error="Not found"), 404)


@app.delete("/api/admin/kb/<doc_id>")
@admin_only
def kb_del(doc_id):
    ok = store.delete(doc_id); retriever.rebuild()
    return (jsonify(ok=True), 200) if ok else (jsonify(error="Not found"), 404)


@app.put("/api/admin/settings")
@admin_only
def settings():
    store.set_settings(request.get_json(force=True)); retriever.rebuild()
    return jsonify(store.data["settings"])


@app.post("/api/admin/search")
@admin_only
def test_search():
    q = (request.get_json(force=True) or {}).get("query", "")
    return jsonify(results=retriever.search(q, k=4, min_score=0.02))


if __name__ == "__main__":
    app.run(host="0.0.0.0" if os.environ.get("PORT") else "127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
