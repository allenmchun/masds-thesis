# src/bootstrap_wis_ci.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pd_wis_from_episode_df(df_pol: pd.DataFrame) -> float:
    """Per-decision WIS at episode granularity using clipped episode log-weights."""
    w = np.exp(df_pol["logw"].to_numpy(dtype=np.float64))
    g = df_pol["G"].to_numpy(dtype=np.float64)
    denom = np.sum(w)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(w * g) / denom)


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="results/tables/wis_episode_level.csv")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out_csv", type=str, default="results/tables/wis_bootstrap_ci.csv")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    required = {"episode_idx", "policy", "G", "logw"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {in_csv}: {missing}")

    rng = np.random.default_rng(args.seed)

    rows = []
    for policy in sorted(df["policy"].unique()):
        df_pol = df[df["policy"] == policy].copy()

        # unique episodes as bootstrap units
        ep_ids = df_pol["episode_idx"].unique()
        n_eps = len(ep_ids)

        # point estimate using all episodes
        point = pd_wis_from_episode_df(df_pol)

        boot = np.empty(args.n_boot, dtype=np.float64)
        for b in range(args.n_boot):
            sample_eps = rng.choice(ep_ids, size=n_eps, replace=True)
            # sample episodes by concatenating their rows
            sampled = df_pol[df_pol["episode_idx"].isin(sample_eps)]
            # NOTE: this "isin" keeps all rows for episodes in sample; it does not replicate duplicates.
            # need replication so we do it explicitly:
            parts = []
            for eid in sample_eps:
                parts.append(df_pol[df_pol["episode_idx"] == eid])
            sampled = pd.concat(parts, ignore_index=True)

            boot[b] = pd_wis_from_episode_df(sampled)

        lo, hi = bootstrap_ci(boot, alpha=args.alpha)

        rows.append(
            {
                "policy": policy,
                "n_episodes": n_eps,
                "wis_point": point,
                "wis_boot_mean": float(np.nanmean(boot)),
                "wis_boot_std": float(np.nanstd(boot)),
                f"ci_{int((args.alpha/2)*100)}": lo,
                f"ci_{int((1-args.alpha/2)*100)}": hi,
            }
        )

        # quick diagnostics: how often logw hits clip bounds
        frac_clip_neg = float((df_pol["logw"] <= -19.999).mean())
        frac_clip_pos = float((df_pol["logw"] >= 19.999).mean())
        rows[-1]["frac_logw_clip_neg"] = frac_clip_neg
        rows[-1]["frac_logw_clip_pos"] = frac_clip_pos

        print(
            f"[RESULT] {policy}: point={point:.6f}  "
            f"boot_mean={np.nanmean(boot):.6f}  "
            f"CI=({lo:.6f},{hi:.6f})  "
            f"clip_neg={frac_clip_neg:.3f} clip_pos={frac_clip_pos:.3f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
