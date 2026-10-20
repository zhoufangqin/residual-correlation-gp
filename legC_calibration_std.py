"""
Re-run of the Table 3 calibration experiment with STANDARDIZED targets
(z-score yA and yB globally, matching the sec.2 joint-NLL protocol) so that NLL
is comparable across datasets. Also splits PHO into mixed / rice-only /
wheat-only to compare canopy fit.

Outputs:
  results/legC/calibration_std.csv        (PHO-mixed / Tecator / Energy, standardized)
  results/legC/pho_canopy_compare.csv     (mixed vs rice vs wheat, standardized)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration_paper")))
from legC_calibration import collect_oof, metrics
from gateB_recovery import load_public
import data_utils as du


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-12)


def r2(y, mu):
    return float(1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2))


def run_pairs(X, pairs, cv, dataset, std=True):
    rows = []
    for Aname, yA, Bname, yB in pairs:
        ya = zscore(yA) if std else yA
        yb = zscore(yB) if std else yB
        res = collect_oof(X, ya, yb, cv)
        for m, (mu, sd) in res.items():
            mt = metrics(yb, mu, sd)
            mt["r2"] = r2(yb, mu)
            rows.append(dict(dataset=dataset, dir=f"{Aname}->{Bname}", method=m, **mt))
    return rows


def pho_arrays(species=None):
    d = du.load_pho()
    items = [v for v in d.values() if species is None or v["species"] == species]
    X = np.array([v["mean_spec"] for v in items], float)
    vc = np.array([v["vcmax"] for v in items], float)
    jx = np.array([v["jmax"] for v in items], float)
    ok = np.all(np.isfinite(X), 1) & np.isfinite(vc) & np.isfinite(jx)
    return X[ok], vc[ok], jx[ok]


def main():
    os.makedirs("results/legC", exist_ok=True)

    # ---- Part A: standardized Table 3 (PHO mixed / Tecator / Energy) ----
    public = load_public()
    rows = []
    Xp, vc, jx = pho_arrays(None)
    print(f"[PHO mixed] n={len(Xp)}", flush=True)
    rows += run_pairs(Xp, [("Vcmax", vc, "Jmax", jx), ("Jmax", jx, "Vcmax", vc)], 5, "PHO")
    for name in ("Tecator", "Energy"):
        X, pairs, cv = public[name]
        print(f"[{name}] n={len(X)}", flush=True)
        rows += run_pairs(X, pairs, cv, name)
    dfA = pd.DataFrame(rows)
    dfA.to_csv("results/legC/calibration_std.csv", index=False)
    print("\n=== STANDARDIZED Table 3 ===")
    print(dfA.round(3).to_string(index=False))

    # ---- Part B: PHO canopy comparison (standardized) ----
    rowsB = []
    for sp in ("mixed", "rice", "wheat"):
        X, vc, jx = pho_arrays(None if sp == "mixed" else sp)
        print(f"\n[PHO {sp}] n={len(X)}", flush=True)
        rr = run_pairs(X, [("Vcmax", vc, "Jmax", jx), ("Jmax", jx, "Vcmax", vc)], 5, f"PHO-{sp}")
        rowsB += rr
    dfB = pd.DataFrame(rowsB)
    dfB.to_csv("results/legC/pho_canopy_compare.csv", index=False)
    print("\n=== PHO canopy comparison (standardized; GP rows) ===")
    print(dfB[dfB.method == "GP"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
