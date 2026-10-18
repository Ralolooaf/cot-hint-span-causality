"""Cluster-bootstrap re-analysis of Study 8 (probe states) and Study 9 (readouts, monitors).

Usage: python analysis/cluster.py <study8_output_dir> <study9_output_dir>
  <study8_output_dir> holds report.json written by study8_threat_probe.py
  <study9_output_dir> holds selected.json and results.jsonl written by study9_cot_carriers.py
(the raw Kaggle outputs; they are not in this folder because of their size).
"""
import json, sys, numpy as np
from collections import defaultdict

def cluster_boot(vals, clusters, B=10000, seed=0):
    """Percentile CI of the pooled mean, resampling clusters with replacement."""
    by = defaultdict(list)
    for v, c in zip(vals, clusters):
        if v is not None and not np.isnan(v):
            by[c].append(v)
    keys = list(by)
    if len(keys) < 2:
        return dict(mean=None, lo=None, hi=None, k=len(keys), n=0)
    rng = np.random.default_rng(seed)
    sums = np.array([sum(by[k]) for k in keys]); ns = np.array([len(by[k]) for k in keys])
    idx = rng.integers(0, len(keys), (B, len(keys)))
    bs = sums[idx].sum(1) / ns[idx].sum(1)
    return dict(mean=float(sums.sum() / ns.sum()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                k=len(keys), n=int(ns.sum()))

def fmt(d):
    return "n/a" if d["mean"] is None else f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (k={d['k']}, n={d['n']})"

# ---- Study 8: R_final per state, clusters = agent (game|agent) and game
rep = json.load(open(sys.argv[1] + "/report.json"))
print("STUDY 8, R_final (final-period restraint)")
for g in ["natA", "natInfo", "natB", "miscal", "myopic", "trigger"]:
    ss = [s for s in rep["states"] if s["group"] == g]
    vals = [s["R_final"] for s in ss]
    ag = ["|".join(s["state_id"].split("|")[:2]) for s in ss]
    gm = [s["state_id"].split("|")[0] for s in ss]
    print(f"  {g:<8} agent-clusters {fmt(cluster_boot(vals, ag))}   game-clusters {fmt(cluster_boot(vals, gm))}")
# S for natA with placebo = myopic mean
plc = np.mean([s["R_final"] for s in rep["states"] if s["group"] == "myopic"])
for g in ["natA", "natInfo"]:
    ss = [s for s in rep["states"] if s["group"] == g and s["p_base"] > 1.4729 + 0.05]
    margin = np.mean([s["p_base"] - 1.4729 for s in ss])
    vals = [(s["R_final"] - plc) / margin for s in ss]
    print(f"  S({g}) agent-clusters {fmt(cluster_boot(vals, ['|'.join(s['state_id'].split('|')[:2]) for s in ss]))}")

# ---- Study 9
sel = {d["dec_id"]: d for d in json.load(open(sys.argv[2] + "/selected.json"))}
res = [json.loads(l) for l in open(sys.argv[2] + "/results.jsonl")]
cell = defaultdict(dict); judge = {}
for r in res:
    if r["method"] == "readout":
        ps = [p for p in r["prices"] if p is not None]
        cell[r["dec_id"]][r["cond"]] = sum(ps) / len(ps) if ps else None
    elif r["method"] == "judge":
        judge[r["dec_id"]] = r["verdict"]
print("\nSTUDY 9, M1/M2 natA (agent clusters)")
ds = [d for d in cell if sel[d]["group"] == "natA"]
ag = lambda d: "|".join(d.split("|")[:2])
for name, a, b in [("pure KO strategic", "ko_strategic", "pl_strategic"), ("pure KO demand", "ko_demand", "pl_demand"),
                   ("pure KO margin", "ko_margin", "pl_margin"), ("incl KO margin", "koall_margin", "plall_margin"),
                   ("implant price-war", "imp_pricewar", "imp_neutral"), ("implant elastic", "imp_elastic", "imp_neutral")]:
    pairs = [(cell[d][a] - cell[d][b], ag(d)) for d in ds if cell[d].get(a) is not None and cell[d].get(b) is not None]
    print(f"  {name:<20} {fmt(cluster_boot([p[0] for p in pairs], [p[1] for p in pairs]))}")
print("\nSTUDY 9, LLM judge flag rate (agent clusters)")
for g in ["trigger", "miscal", "myopic", "natA", "natB", "natInfo"]:
    dd = [d for d in judge if sel[d]["group"] == g and judge[d] is not None]
    print(f"  {g:<8} {fmt(cluster_boot([1.0 if judge[d]=='YES' else 0.0 for d in dd], [ag(d) for d in dd]))}")
