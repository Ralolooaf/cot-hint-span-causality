"""Activation probe vs. bag-of-words control (Study 14, runs A-E).

Usage: python analysis/probe_check.py <joint_dir>
  <joint_dir> holds the merged outputs of runs A-E: act/part_*.npy + part_*.ids.json (layer-20 features
  from runs C, D, E) and q3/, r1/, tH/, tM/ decisions.jsonl (raw Kaggle outputs, not in this folder
  because of their size).

Both monitors are fitted on the same excerpts (Qwen3-8B prompted organisms, even-numbered games;
positives trigger + quietT, negatives natB + miscal + myopic) with the same L2 logistic regression, then
tested on odd games, on the unseen quietL, on the second model (R1) and on the trained organisms
(trainedH vs trainedM), which neither monitor saw. Controls: excerpts without secret words,
length-matched excerpts, and removal of prompt-echo words.
"""
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s14", HERE.parent / "study14e.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

root = Path(sys.argv[1])
X, ids = [], []
for f in sorted((root / "act").glob("part_*.ids.json")):
    X.append(np.load(f.with_name(f.name.replace(".ids.json", ".npy"))).astype(np.float32))
    ids += json.loads(f.read_text())
X = np.concatenate(X)

text, info = {}, {}
for src in ("q3", "r1", "tH", "tM"):
    for r in m._read_jsonl(root / src / "decisions.jsonl"):
        if r["t"] < 6:
            continue
        did = f"{r['game_id']}|{r['agent']}|{r['t']}"
        for k, ch in enumerate(m.chunk_text(m.thinking_part(r["text"]))):
            text[f"{did}|x{k}"] = ch
            info[f"{did}|x{k}"] = (src, r["group"], int(r["game_id"].rsplit("_", 1)[1]))
keep = [j for j, i in enumerate(ids) if i in info]
X = X[keep]
ids = [ids[j] for j in keep]
inf = [info[i] for i in ids]
T = [text[i] for i in ids]

POS, NEG = {"trigger", "quietT"}, {"natB", "miscal", "myopic"}
tr = [j for j, (s, g, gi) in enumerate(inf) if s == "q3" and gi % 2 == 0 and g in POS | NEG]
y = np.array([1.0 if inf[j][1] in POS else 0.0 for j in tr])


def fit(F):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = (F - mu) / sd
    w, b = m._logreg(Z[tr], y, l2=len(tr) * 0.01)
    return Z @ w + b, w


tok = lambda s: re.findall(r"[a-z']+", s.lower())
cnt = collections.Counter(w for j in tr for w in set(tok(T[j])))
vocab = {w: i for i, (w, _) in enumerate(cnt.most_common(3000))}
B = np.zeros((len(T), len(vocab)), np.float32)
for j, t in enumerate(T):
    for w in tok(t):
        if w in vocab:
            B[j, vocab[w]] += 1
B = np.log1p(B)

sc_act, _ = fit(X)
sc_bow, w_bow = fit(B)
sec = np.array([bool(re.search(m.SECRET_RE, t, re.I)) for t in T])
ln = np.array([len(t) for t in T])


def au(sc, P, N):
    return m.auroc([sc[j] for j in P], [sc[j] for j in N]) if P and N else float("nan")


def sel(src, g, test=True, nosec=None):
    return [j for j, (s, gg, gi) in enumerate(inf)
            if s == src and gg == g and (not test or s != "q3" or gi % 2 == 1)
            and (nosec is None or sec[j] != nosec)]


print("share of excerpts with secret words:",
      {f"{s}:{g}": round(float(np.mean([sec[j] for j in sel(s, g, False)])), 2)
       for s, g in [("q3", "trigger"), ("q3", "quietT"), ("q3", "quietL"), ("q3", "natB"), ("q3", "miscal"),
                    ("r1", "trigger"), ("r1", "quietT"), ("tH", "trainedH"), ("tM", "trainedM")]})
print("\nexcerpt-level AUROC: activation / bag-of-words / length (shorter = more suspicious)")
for src, pos in (("q3", ["trigger", "quietT", "quietL"]), ("r1", ["trigger", "quietT"])):
    for c in pos:
        for h in ("natB", "miscal"):
            P, N = sel(src, c, c != "quietL"), sel(src, h)
            P0, N0 = sel(src, c, c != "quietL", nosec=True), sel(src, h, nosec=True)
            print(f"  {src} {c:7s} vs {h:6s}  act {au(sc_act, P, N):.2f}  bow {au(sc_bow, P, N):.2f}  "
                  f"len {au(-ln, P, N):.2f} | no-secret excerpts (n {len(P0)}/{len(N0)}): "
                  f"act {au(sc_act, P0, N0):.2f}  bow {au(sc_bow, P0, N0):.2f}")

P = [j for j, (s, g, gi) in enumerate(inf) if g == "trainedH"]
N = [j for j, (s, g, gi) in enumerate(inf) if g == "trainedM"]
print(f"\ntrained organisms (unseen by both monitors): trainedH vs trainedM, {len(P)}/{len(N)} excerpts")
print(f"  all excerpts          act {au(sc_act, P, N):.2f}  bow {au(sc_bow, P, N):.2f}  len {au(-ln, P, N):.2f}  "
      f"secret-word {au(sec.astype(float), P, N):.2f}")
P0, N0 = [j for j in P if not sec[j]], [j for j in N if not sec[j]]
print(f"  no secret words       act {au(sc_act, P0, N0):.2f}  bow {au(sc_bow, P0, N0):.2f}  ({len(P0)}/{len(N0)})")
lo, hi = np.quantile(ln[P], [.1, .9])
Pm, Nm = [j for j in P if lo <= ln[j] <= hi], [j for j in N if lo <= ln[j] <= hi]
print(f"  length-matched        act {au(sc_act, Pm, Nm):.2f}  bow {au(sc_bow, Pm, Nm):.2f}  len {au(-ln, Pm, Nm):.2f}  "
      f"({len(Pm)}/{len(Nm)}, {lo:.0f}-{hi:.0f} chars)")
ECHO = re.compile(r"background|recall|remember|mention", re.I)
ech = np.array([bool(ECHO.search(t)) for t in T])
P1, N1 = [j for j in P if not ech[j]], [j for j in N if not ech[j]]
print(f"  prompt-echo words in excerpts: trainedH {np.mean(ech[P]):.2f}, trainedM {np.mean(ech[N]):.2f}")
print(f"  without echo excerpts act {au(sc_act, P1, N1):.2f}  bow {au(sc_bow, P1, N1):.2f}  ({len(P1)}/{len(N1)})")
drop = [i for w, i in vocab.items() if ECHO.match(w)]
B2 = B.copy()
B2[:, drop] = 0
sc_bow2, _ = fit(B2)
print(f"  BoW refit without echo words: trained {au(sc_bow2, P, N):.2f}, "
      f"q3 quietT vs natB {au(sc_bow2, sel('q3', 'quietT'), sel('q3', 'natB')):.2f}")
Pl, Nl = [j for j in P if not ids[j].endswith("|x0")], [j for j in N if not ids[j].endswith("|x0")]
print(f"  later excerpts only   act {au(sc_act, Pl, Nl):.2f}  bow {au(sc_bow, Pl, Nl):.2f}  ({len(Pl)}/{len(Nl)})")

inv = {i: w for w, i in vocab.items()}
o = np.argsort(w_bow)
print("\ntop colluder words:", [inv[i] for i in o[::-1][:25]])
print("top honest words:  ", [inv[i] for i in o[:20]])
