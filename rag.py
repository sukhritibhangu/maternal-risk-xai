"""Admin-governed RAG: knowledge store -> chunks -> TF-IDF retrieval -> per-driver suggestion cards.
The Retriever class is the only thing to swap if you later move to dense embeddings
(e.g. sentence-transformers + FAISS). The store, admin controls and card builder stay the same."""
import json, os, re, time, uuid, threading, urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from features import DRIVER_GROUPS, GROUP_QUERY, LEVEL_RANK

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "knowledge"))  # point at a persistent disk in production
os.makedirs(DATA_DIR, exist_ok=True)
STORE = os.path.join(DATA_DIR, "store.json")
AUDIT = os.path.join(DATA_DIR, "audit.log")
LOCK = threading.Lock()


# ---------- store
class KnowledgeStore:
    def __init__(self):
        self.data = {"settings": {"only_verified": False}, "docs": []}
        if os.path.exists(STORE):
            self.data = json.load(open(STORE, encoding="utf-8"))
        else:
            from seed_kb import SEED, SOURCE
            for d in SEED:
                self.add(d["title"], SOURCE, d["tags"], d["text"], verified=False, actor="seed")

    def _save(self):
        with LOCK:
            json.dump(self.data, open(STORE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    def log(self, actor, action, detail=""):
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{actor}\t{action}\t{detail}\n")

    def add(self, title, source, tags, text, verified=False, actor="admin"):
        d = dict(id=uuid.uuid4().hex[:8], title=title.strip(), source=source.strip(), tags=[t.strip() for t in tags if t.strip()],
                 text=re.sub(r"\n{3,}", "\n\n", text.strip()), verified=verified, enabled=True, added=time.strftime("%Y-%m-%d"))
        self.data["docs"].append(d); self._save(); self.log(actor, "add", f"{d['id']} {d['title']}")
        return d

    def update(self, doc_id, patch, actor="admin"):
        for d in self.data["docs"]:
            if d["id"] == doc_id:
                for k in ("title", "source", "tags", "text", "verified", "enabled"):
                    if k in patch:
                        d[k] = patch[k]
                self._save(); self.log(actor, "update", f"{doc_id} {list(patch)}")
                return d

    def delete(self, doc_id, actor="admin"):
        n = len(self.data["docs"])
        self.data["docs"] = [d for d in self.data["docs"] if d["id"] != doc_id]
        self._save(); self.log(actor, "delete", doc_id)
        return len(self.data["docs"]) < n

    def set_settings(self, patch, actor="admin"):
        self.data["settings"].update({k: bool(v) for k, v in patch.items() if k in self.data["settings"]})
        self._save(); self.log(actor, "settings", json.dumps(patch))

    def active_docs(self):
        ov = self.data["settings"]["only_verified"]
        return [d for d in self.data["docs"] if d["enabled"] and (d["verified"] or not ov)]


# ---------- chunking + retrieval
def chunk(doc, target=110):
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len((buf + " " + p).split()) > target * 1.5:
            out.append(buf); buf = p
        else:
            buf = (buf + "\n\n" + p).strip()
    if buf:
        out.append(buf)
    return [dict(doc_id=doc["id"], title=doc["title"], source=doc["source"], verified=doc["verified"],
                 tags=doc["tags"], text=t) for t in out]


class Retriever:
    def __init__(self, store):
        self.store = store; self.rebuild()

    def rebuild(self):
        self.chunks = [c for d in self.store.active_docs() for c in chunk(d)]
        if not self.chunks:
            self.vec = None; return
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.mat = self.vec.fit_transform([f"{c['title']} {' '.join(c['tags'])} {c['text']}" for c in self.chunks])

    def search(self, query, k=2, tags=None, min_score=0.12, exclude=()):
        if not self.vec:
            return []
        s = (self.mat @ self.vec.transform([query]).T).toarray().ravel()
        if tags:
            s = s + np.array([0.25 if set(tags) & set(c["tags"]) else 0 for c in self.chunks])
        out = []
        for i in np.argsort(-s):
            c = self.chunks[i]
            key = (c["doc_id"], c["text"][:40])
            if s[i] < min_score or key in exclude:
                continue
            out.append(dict(c, score=float(s[i]))); 
            if len(out) == k:
                break
        return out


# ---------- personalised cards
def build_cards(flags, contribs, retriever, max_cards=5):
    """One card per driver group. Order: critical flags, then by positive SHAP contribution.
    Each card is tied to the patient's own values and carries source passages."""
    shap_by_group, val_by_feat = {}, {}
    for c in contribs:
        for g, fs in DRIVER_GROUPS.items():
            if c["feature"] in fs:
                shap_by_group[g] = shap_by_group.get(g, 0) + c["shap"]
    groups = {}
    for fl in flags:
        g = groups.setdefault(fl["group"], dict(group=fl["group"], flags=[], rank=0))
        g["flags"].append(fl); g["rank"] = max(g["rank"], LEVEL_RANK[fl["level"]])
    for g, s in shap_by_group.items():
        if s > 0.015 and g not in groups:
            groups[g] = dict(group=g, flags=[], rank=0)
    ordered = sorted(groups.values(), key=lambda x: (-x["rank"], -shap_by_group.get(x["group"], 0)))[:max_cards]
    used, cards = set(), []
    for g in ordered:
        name = g["group"]
        q = GROUP_QUERY[name] + " " + " ".join(f["text"] for f in g["flags"])
        tags = DRIVER_GROUPS[name]
        hits = retriever.search(q, k=2, tags=tags, exclude=used)
        used.update((h["doc_id"], h["text"][:40]) for h in hits)
        level = max([f["level"] for f in g["flags"]], default="info", key=lambda l: LEVEL_RANK[l])
        if hits or LEVEL_RANK[level] >= 2:  # never silently drop a high/critical driver; show an explicit gap instead
            cards.append(dict(group=name, level=level,
                              reasons=[dict(text=f["text"], source_name=f["source_name"], source_url=f["source_url"], note=f.get("note", "")) for f in g["flags"]],
                              targets=[f["target"] for f in g["flags"] if f["target"]],
                              shap=float(shap_by_group.get(name, 0)), passages=hits))
    gen = retriever.search("antenatal care danger signs when to go to hospital", k=1, tags=["general"], exclude=used)
    if gen:
        cards.append(dict(group="General care and danger signs", level="info", reasons=[], targets=[], shap=0.0, passages=gen))
    return cards


def llm_summary(patient_line, cards):
    """Optional. Active only when GEMINI_API_KEY is set. Uses Gemini Flash-Lite (model id from GEMINI_MODEL).
    The model is told to use ONLY the retrieved passages. PRIVACY: the risk %, the flagged values and the
    retrieved passages are sent to Google. Use a PAID key: Google's terms for the free tier allow it to use
    submitted content to improve its products, and human reviewers may read it."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    src, n = [], 0
    for c in cards:
        for p in c["passages"]:
            n += 1; src.append(f"[S{n}] ({p['title']}) {p['text']}")
    if not src:
        return None
    model = re.sub(r"[^A-Za-z0-9._-]", "", os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    system = ("You draft patient-friendly next steps for a maternal health tool. Use ONLY the numbered sources provided; "
              "if they do not cover something, say so. Cite sources as [S#]. Do not diagnose and do not give drug doses. "
              "Plain language, maximum 150 words.")
    prompt = f"Patient context: {patient_line}\n\nSources:\n" + "\n\n".join(src)
    body = json.dumps(dict(systemInstruction=dict(parts=[dict(text=system)]),
                           contents=[dict(role="user", parts=[dict(text=prompt)])],
                           generationConfig=dict(temperature=0.2, maxOutputTokens=600))).encode()
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
                                 {"content-type": "application/json", "x-goog-api-key": key})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=20))
        parts = r["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip() or None
    except Exception as e:
        print("Gemini summary unavailable:", type(e).__name__, str(e)[:200])
        return None
