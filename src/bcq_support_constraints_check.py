from pathlib import Path
import numpy as np
import pandas as pd

# use existing BigQuery -> dataset pipeline
import sys
sys.path.append("src")
import train

OUT_TABLE = Path("results/tables/bcq_support_constraints_summary.csv")
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

ACTIONS = [0, 1, 2, 3, 4]
RARE = {3, 4}

def pct(x):
    return float(x) if np.isfinite(x) else np.nan

def action_props(actions: np.ndarray, n_actions: int = 5):
    vc = pd.Series(actions.astype(int)).value_counts(normalize=True)
    return np.array([vc.get(a, 0.0) for a in range(n_actions)], dtype=float)

def main():
    dataset = train.load_dataset()

    # access raw arrays via dataset.episodes (stable across d3rlpy versions)
    # build clinician actions from all transitions
    acts = []
    mapz = []

    for ep in dataset.episodes:
        a = ep.actions
        # actions can be shape (T,1); flatten
        a = np.asarray(a).reshape(-1)
        acts.append(a)

        # map_z is part of observations; use same STATE_COLS in train.py
        # find column index for map_z in state columns
        state_cols = getattr(train, "STATE_COLS", None)
        if state_cols is None:
            raise RuntimeError("train.STATE_COLS not found. Ensure train.py defines STATE_COLS list.")
        if "map_z" not in state_cols:
            raise RuntimeError("'map_z' not in STATE_COLS. Add it or adjust this script.")

        idx = state_cols.index("map_z")
        obs = np.asarray(ep.observations)
        mz = obs[:, idx].reshape(-1)
        mapz.append(mz)

    acts = np.concatenate(acts, axis=0)
    mapz = np.concatenate(mapz, axis=0)

    # overall clinician distribution
    overall = action_props(acts, n_actions=5)

    # MAP_z nonzero binning (same approach you used earlier)
    nonzero = mapz != 0
    mz_nz = mapz[nonzero]
    acts_nz = acts[nonzero]

    q1 = np.quantile(mz_nz, 1/3)
    q2 = np.quantile(mz_nz, 2/3)

    def bin_name(m):
        if m <= q1: return "low"
        if m <= q2: return "mid"
        return "high"

    bins = np.array([bin_name(m) for m in mz_nz])

    rows = []

    # overall metrics
    rows.append({
        "stratum": "overall",
        "n": int(len(acts)),
        **{f"p_action_{a}": pct(overall[a]) for a in ACTIONS},
        "p_rare_3_4": pct(overall[3] + overall[4]),
        "low_support_actions_<1%": int(sum(overall[a] < 0.01 for a in ACTIONS)),
    })

    # by MAP_z bins (computed on nonzero only)
    for b in ["low", "mid", "high"]:
        mask = bins == b
        a_b = acts_nz[mask]
        props = action_props(a_b, n_actions=5)
        rows.append({
            "stratum": f"mapz_{b}_nonzero",
            "n": int(mask.sum()),
            **{f"p_action_{a}": pct(props[a]) for a in ACTIONS},
            "p_rare_3_4": pct(props[3] + props[4]),
            "low_support_actions_<1%": int(sum(props[a] < 0.01 for a in ACTIONS)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)
    print(f"[OK] Wrote {OUT_TABLE}")
    print("[INFO] MAP_z nonzero cutpoints:")
    print(f"  q1 (33%): {q1:.6f}")
    print(f"  q2 (67%): {q2:.6f}")

if __name__ == "__main__":
    main()
