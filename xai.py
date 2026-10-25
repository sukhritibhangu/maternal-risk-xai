"""Explainability layer: SHAP, per-tree votes, what-if counterfactuals, ICE curves."""
import numpy as np, pandas as pd, shap
from features import FEATURES, BY_NAME, FEATURE_TO_GROUP, LEVEL_RANK, flags_for

THRESH = 0.5  # decision threshold used for the "smallest change" search


def band(p):  # display convention only, NOT a calibrated clinical category
    return "Lower" if p < 0.35 else ("Elevated" if p < 0.65 else "High")


class Explainer:
    def __init__(self, model, background_csv=None):
        self.m = model
        self.cols = list(model.feature_names_in_)
        self.tx = shap.TreeExplainer(model)
        self.base = float(np.atleast_1d(self.tx.expected_value)[-1])
        self.background_csv = background_csv
        self._global = None

    # ---- helpers
    def _df(self, v):
        return pd.DataFrame([[float(v[c]) for c in self.cols]], columns=self.cols)

    def _many(self, rows):
        return pd.DataFrame(rows, columns=self.cols)

    def proba(self, v):
        return float(self.m.predict_proba(self._df(v))[0, 1])

    def _sv(self, df):
        raw = self.tx.shap_values(df)
        if isinstance(raw, list):
            return np.asarray(raw[-1])
        raw = np.asarray(raw)
        return raw[:, :, -1] if raw.ndim == 3 else raw

    # ---- explanations
    def votes(self, v):
        X = self._df(v).values.astype(np.float32)
        return [float(t.predict_proba(X)[0, 1]) for t in self.m.estimators_]

    def local(self, v):
        sv = self._sv(self._df(v))[0]
        p = self.proba(v)
        contribs = [dict(feature=c, value=float(v[c]), shap=float(s)) for c, s in zip(self.cols, sv)]
        contribs.sort(key=lambda d: -abs(d["shap"]))
        return dict(base=self.base, prediction=p, contribs=contribs,
                    additivity_gap=float(abs(self.base + sv.sum() - p)))

    def counterfactuals(self, v, flags):
        p0 = self.proba(v)
        rows, targets = [], {}
        for fl in flags:
            f = fl["feature"]
            spec = BY_NAME[f]
            if spec["mod"] and LEVEL_RANK[fl["level"]] >= 1 and f not in targets:
                targets[f] = spec["ref"]
        # also pair-up: if SBP flagged also normalise DBP etc. (same driver group)
        for f in list(targets):
            for g in [x for x in BY_NAME if FEATURE_TO_GROUP.get(x) == FEATURE_TO_GROUP.get(f) and BY_NAME[x]["mod"]]:
                if abs(v[g] - BY_NAME[g]["ref"]) > 1e-9:
                    targets.setdefault(g, BY_NAME[g]["ref"])
        by_group = {}
        for f, ref in targets.items():
            by_group.setdefault(FEATURE_TO_GROUP[f], {})[f] = ref
        for group, ch in by_group.items():
            w = dict(v); w.update(ch)
            p1 = self.proba(w)
            smallest = None
            if p0 >= THRESH:
                for s in np.linspace(0, 1, 41)[1:]:
                    x = dict(v)
                    for f, ref in ch.items():
                        x[f] = v[f] + (ref - v[f]) * s
                    if self.proba(x) < THRESH:
                        smallest = {f: round(x[f], 1) for f in ch}
                        break
            rows.append(dict(group=group, changes={f: dict(now=v[f], target=r, note=BY_NAME[f].get("ref_note", "")) for f, r in ch.items()},
                             risk_after=p1, delta=p1 - p0, smallest_change=smallest))
        rows.sort(key=lambda r: r["delta"])
        allw = dict(v)
        for ch in by_group.values():
            allw.update(ch)
        return dict(rows=rows, all_normalised=self.proba(allw) if by_group else p0)

    def ice(self, v, features, n=40):
        out = {}
        for f in features:
            spec = BY_NAME[f]
            xs = np.linspace(spec["min"], spec["max"], n)
            rows = []
            for x in xs:
                w = dict(v); w[f] = float(x)
                rows.append([w[c] for c in self.cols])
            ys = self.m.predict_proba(self._many(rows))[:, 1]
            out[f] = dict(x=[float(x) for x in xs], y=[float(y) for y in ys], current=float(v[f]), unit=spec["unit"], label=spec["label"])
        return out

    # ---- global
    def _background(self, n=600):
        if self.background_csv:
            try:
                return pd.read_csv(self.background_csv)[self.cols].dropna().head(n), "uploaded"
            except Exception:
                pass
        r = np.random.default_rng(7)
        age = np.clip(r.normal(27, 5.5, n).round(), 16, 45)
        g = np.clip(r.poisson(1.6, n) + 1, 1, 7); p = np.minimum(np.clip(r.poisson(1.0, n), 0, 5), g - 1)
        l = np.minimum(p, np.clip(p - (r.random(n) < .05), 0, None)); a = np.minimum(r.poisson(.3, n), 3)
        d = (r.random(n) < .05).astype(float)
        cols = dict(Age=age, G=g, P=p, L=l, A=a, D=d,
                    SystolicBP=np.clip(r.normal(115, 15, n), 80, 190), DiastolicBP=np.clip(r.normal(75, 10, n), 50, 120),
                    RBS=np.clip(r.normal(120, 40, n), 70, 350), BodyTemp=np.clip(r.normal(98.6, .8, n), 96.5, 103),
                    HeartRate=np.clip(r.normal(82, 10, n), 55, 130), HB=np.clip(r.normal(11.3, 1.5, n), 6, 15),
                    HBA1C=np.clip(r.normal(5.3, .6, n), 4, 10), RR=np.clip(r.normal(19, 3, n), 12, 32))
        return pd.DataFrame(cols)[self.cols], "synthetic"

    def global_view(self):
        if self._global is None:
            bg, src = self._background()
            sv = self._sv(bg)
            ms = np.abs(sv).mean(0)
            imp = self.m.feature_importances_
            self._global = dict(source=src, rows=sorted(
                [dict(feature=c, label=BY_NAME[c]["label"], mean_abs_shap=float(s), impurity=float(i)) for c, s, i in zip(self.cols, ms, imp)],
                key=lambda d: -d["mean_abs_shap"]))
        return self._global


def narrative(p, local, flags):
    pos = [c for c in local["contribs"] if c["shap"] > 0.01][:3]
    neg = [c for c in local["contribs"] if c["shap"] < -0.01][:2]
    nm = lambda c: BY_NAME[c["feature"]]["label"]
    s = f"The model estimates a {p*100:.0f}% probability of the high-risk class ({band(p).lower()} band)."
    if pos:
        s += " It is pushed up mainly by " + ", ".join(f"{nm(c)} ({c['value']:g}{(' ' + BY_NAME[c['feature']]['unit']) if BY_NAME[c['feature']]['unit'] else ''})" for c in pos) + "."
    if neg:
        s += " It is held down by " + ", ".join(nm(c) for c in neg) + "."
    return s
