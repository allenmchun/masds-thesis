#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from google.cloud import bigquery

from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
from d3rlpy.algos.qlearning import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.algos.qlearning.bcq import DiscreteBCQ, DiscreteBCQConfig
from d3rlpy.dataset import MDPDataset


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"

# run dirs (d3rlpy_logs)
BC_RUN_DIR  = PROJECT_ROOT / "d3rlpy_logs" / "bc_final_ne5_v2"
CQL_RUN_DIR = PROJECT_ROOT / "d3rlpy_logs" / "cql_final_ne5_v2"
BCQ_RUN_DIR = PROJECT_ROOT / "d3rlpy_logs" / "bcq_final_ne5_v2"  # best@10k is model_10000.d3

# torch checkpoints (results/models)
BC_PT  = RESULTS_DIR / "models" / "bc_final_ne5_v2.pt"
CQL_PT = RESULTS_DIR / "models" / "cql_final_ne5_v2.pt"
BCQ_PT = RESULTS_DIR / "models" / "bcq_best10k_ne5_v2.pt"  # if you have it; otherwise set to bcq_final_ne5_v2.pt


# state columns (should match what was used in training)
STATE_COLS = [
    "age_years", "gender",
    "map_z", "hr_z", "spo2_z", "urine_ml_z",
    "lactate_z", "creatinine_z", "ph_z", "d4h_map_z",
    "ne_rate_z", "epi_rate_z", "phe_rate_z", "vasopressin_rate_z",
    "miss_map", "miss_hr", "miss_spo2",
    "miss_lactate", "miss_creatinine", "miss_ph",
]


def _ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_sample_from_bigquery(
    dataset_fqn: str,
    split: str,
    n0: int,
    n1: int,
    n2: int,
    n3: int,
    n4: int,
) -> pd.DataFrame:
    """
    Stratified sample per action_bin, like your earlier scripts.
    """
    client = bigquery.Client()

    query = f"""
    WITH base AS (
      SELECT
        stay_id,
        t,
        split,
        age_years,
        CASE
          WHEN gender IN ('M','m','Male','male') THEN 1
          WHEN gender IN ('F','f','Female','female') THEN 0
          ELSE NULL
        END AS gender,
        map_z, hr_z, spo2_z, urine_ml_z,
        lactate_z, creatinine_z, ph_z, d4h_map_z,
        ne_rate_z, epi_rate_z, phe_rate_z, vasopressin_rate_z,
        miss_map, miss_hr, miss_spo2,
        miss_lactate, miss_creatinine, miss_ph,
        action_bin
      FROM `{dataset_fqn}`
      WHERE split = '{split}'
    ),
    sampled AS (
      SELECT *
      FROM base
      QUALIFY
        ROW_NUMBER() OVER (PARTITION BY action_bin ORDER BY RAND()) <=
          CASE action_bin
            WHEN 0 THEN {n0}
            WHEN 1 THEN {n1}
            WHEN 2 THEN {n2}
            WHEN 3 THEN {n3}
            WHEN 4 THEN {n4}
            ELSE 0
          END
    )
    SELECT * FROM sampled
    """

    print("[INFO] Running BigQuery:\n")
    print(query)

    df = client.query(query).to_dataframe()
    # sort for event detection
    df = df.sort_values(["stay_id", "t"]).reset_index(drop=True)
    return df


def _to_numpy_obs(df: pd.DataFrame) -> np.ndarray:
    obs = df[STATE_COLS].to_numpy(dtype=np.float32)
    # fill NaN/Inf like training did
    bad = ~np.isfinite(obs)
    n_bad_rows = int(np.any(bad, axis=1).sum())
    if n_bad_rows > 0:
        print(f"[WARN] Found {n_bad_rows} rows with NaN/Inf in state features. Filling with 0.0.")
        obs[bad] = 0.0
    return obs


def _load_latest_d3(run_dir: Path, prefer_step: Optional[int] = None) -> Optional[Path]:
    """
    Choose a model_XXXXX.d3.
    If prefer_step is set, try that first.
    """
    if prefer_step is not None:
        p = run_dir / f"model_{prefer_step}.d3"
        if p.exists():
            return p

    candidates = sorted(run_dir.glob("model_*.d3"))
    if not candidates:
        return None

    def step_of(p: Path) -> int:
        # model_100000.d3 -> 100000
        s = p.stem.replace("model_", "")
        try:
            return int(s)
        except Exception:
            return -1

    candidates = sorted(candidates, key=step_of)
    return candidates[-1]


def _try_load_qfuncs_into_impl(impl: torch.nn.Module, q_sd: Dict[str, torch.Tensor]) -> bool:
    """
    q_sd keys look like:
      '0._encoder._layers.0.weight', ..., '0._fc.bias'
      '1._encoder...'
    We try to find a ModuleList in impl such that each element can load its chunk.
    """
    # group by leading integer prefix "0.", "1.", ...
    groups: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, v in q_sd.items():
        if "." not in k:
            continue
        head, rest = k.split(".", 1)
        if head.isdigit():
            i = int(head)
            groups.setdefault(i, {})[rest] = v

    if not groups:
        return False

    n = max(groups.keys()) + 1
    # scan all ModuleLists in impl (including nested)
    for name, mod in impl.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            ok = True
            for i in range(n):
                sd_i = groups.get(i, None)
                if sd_i is None:
                    ok = False
                    break
                try:
                    mod[i].load_state_dict(sd_i, strict=True)
                except Exception:
                    ok = False
                    break
            if ok:
                print(f"[INFO] Loaded q_funcs into ModuleList at impl submodule: {name}")
                return True

    return False


def _try_load_imitator_into_impl(impl: torch.nn.Module, im_sd: Dict[str, torch.Tensor]) -> bool:
    """
    For BC/BCQ imitator:
      '_encoder._layers.0.weight', ..., '_fc.bias'
    We scan submodules and load into the first one that accepts the state_dict.
    """
    for name, mod in impl.named_modules():
        # avoid huge expensive attempts on every leaf param container
        if not isinstance(mod, torch.nn.Module):
            continue
        try:
            mod.load_state_dict(im_sd, strict=True)
            print(f"[INFO] Loaded imitator into impl submodule: {name}")
            return True
        except Exception:
            continue
    return False


def _manual_load_from_pt(algo, pt_path: Path, device: str) -> None:
    """
    Robust PT loader used when .d3 is invalid magic number.
    Supports:
      - CQL: keys include q_funcs, targ_q_funcs
      - BCQ: keys include q_funcs, targ_q_funcs, imitator
      - BC: keys include imitator
    """
    ckpt = torch.load(str(pt_path), map_location=torch.device(device))
    if not isinstance(ckpt, dict):
        raise RuntimeError("Torch checkpoint is not a dict-like object.")

    impl = getattr(algo, "_impl", None)
    if impl is None:
        raise RuntimeError("algo._impl is None; build_with_dataset must be called before loading weights.")

    # 1) q_funcs / targ_q_funcs
    if "q_funcs" in ckpt:
        ok = _try_load_qfuncs_into_impl(impl, ckpt["q_funcs"])
        if not ok:
            raise RuntimeError("Could not load q_funcs into any ModuleList inside impl.")

    if "targ_q_funcs" in ckpt:
        ok = _try_load_qfuncs_into_impl(impl, ckpt["targ_q_funcs"])
        if not ok:
            print("[WARN] Could not load targ_q_funcs into any ModuleList inside impl (continuing).")

    # 2) imitator (BC / BCQ)
    if "imitator" in ckpt:
        ok = _try_load_imitator_into_impl(impl, ckpt["imitator"])
        if not ok:
            raise RuntimeError("Could not load imitator into any submodule inside impl.")

    return


def build_and_load_policy(
    policy_name: str,
    obs_sample: np.ndarray,
    n_actions: int,
    device: str,
) -> Tuple[object, callable]:
    """
    Returns (algo, predict_fn).
    predict_fn(obs: np.ndarray) -> np.ndarray of int actions
    """
    policy_name = policy_name.lower().strip()

    if policy_name == "bc":
        algo = DiscreteBC(config=DiscreteBCConfig(), device=device, enable_ddp=False)
        # build network shapes
        terminals = np.zeros((obs_sample.shape[0],), dtype=np.bool_)
        # d3rlpy requires at least one terminal/timeout in the dataset to build.
        terminals[-1] = True
        tmp_ds = MDPDataset(
            observations=obs_sample,
            actions=np.zeros((obs_sample.shape[0],), dtype=np.int64),
            rewards=np.zeros((obs_sample.shape[0],), dtype=np.float32),
            terminals=terminals,
        )
        algo.build_with_dataset(tmp_ds)
        d3 = _load_latest_d3(BC_RUN_DIR, prefer_step=100000)
        if d3 is not None:
            try:
                algo.load_model(str(d3))
                print(f"[INFO] Loaded BC from d3: {d3}")
            except Exception as e:
                print(f"[WARN] BC .d3 load failed: {e}. Falling back to PT: {BC_PT}")
                _manual_load_from_pt(algo, BC_PT, device)
        else:
            print(f"[WARN] No BC .d3 found. Falling back to PT: {BC_PT}")
            _manual_load_from_pt(algo, BC_PT, device)

        return algo, lambda x: algo.predict(x)

    if policy_name == "cql":
        algo = DiscreteCQL(config=DiscreteCQLConfig(), device=device, enable_ddp=False)
        terminals = np.zeros((obs_sample.shape[0],), dtype=np.bool_)
        # d3rlpy requires at least one terminal/timeout in the dataset to build.
        terminals[-1] = True
        tmp_ds = MDPDataset(
            observations=obs_sample,
            actions=np.zeros((obs_sample.shape[0],), dtype=np.int64),
            rewards=np.zeros((obs_sample.shape[0],), dtype=np.float32),
            terminals=terminals,
        )
        algo.build_with_dataset(tmp_ds)
        d3 = _load_latest_d3(CQL_RUN_DIR, prefer_step=100000)
        if d3 is not None:
            try:
                algo.load_model(str(d3))
                print(f"[INFO] Loaded CQL from d3: {d3}")
            except Exception as e:
                print(f"[WARN] CQL .d3 load failed: {e}. Falling back to PT: {CQL_PT}")
                _manual_load_from_pt(algo, CQL_PT, device)
        else:
            print(f"[WARN] No CQL .d3 found. Falling back to PT: {CQL_PT}")
            _manual_load_from_pt(algo, CQL_PT, device)

        return algo, lambda x: algo.predict(x)

    if policy_name == "bcq":
        algo = DiscreteBCQ(config=DiscreteBCQConfig(), device=device, enable_ddp=False)
        terminals = np.zeros((obs_sample.shape[0],), dtype=np.bool_)
        # d3rlpy requires at least one terminal/timeout in the dataset to build.
        terminals[-1] = True
        tmp_ds = MDPDataset(
            observations=obs_sample,
            actions=np.zeros((obs_sample.shape[0],), dtype=np.int64),
            rewards=np.zeros((obs_sample.shape[0],), dtype=np.float32),
            terminals=terminals,
        )
        algo.build_with_dataset(tmp_ds)
        # best@10k
        d3 = _load_latest_d3(BCQ_RUN_DIR, prefer_step=10000)
        if d3 is not None:
            try:
                algo.load_model(str(d3))
                print(f"[INFO] Loaded BCQ from d3: {d3}")
            except Exception as e:
                # If you have bcq_best10k_ne5_v2.pt use it; otherwise swap to bcq_final_ne5_v2.pt
                print(f"[WARN] BCQ .d3 load failed: {e}. Falling back to PT: {BCQ_PT}")
                _manual_load_from_pt(algo, BCQ_PT, device)
        else:
            print(f"[WARN] No BCQ .d3 found. Falling back to PT: {BCQ_PT}")
            _manual_load_from_pt(algo, BCQ_PT, device)

        return algo, lambda x: algo.predict(x)

    raise ValueError(f"Unknown policy_name={policy_name}")


def detect_first_hypotension_windows(
    df: pd.DataFrame,
    hypotension_thr: float,
    recovery_thr: float,
    pre_h: Tuple[int, int],
    post_h: Tuple[int, int],
    allow_post_without_recovery: bool = False,
) -> pd.DataFrame:
    """
    For each stay_id (rows sorted by t), find the first hypotension event where map_z <= hypotension_thr.

    IMPORTANT: This implementation uses *row index offsets* within each stay, not absolute hour offsets.
    This avoids the failure mode where t is not contiguous after stratified sampling.

    Windows:
      - pre window: indices [i0+pre_h[0],  i0+pre_h[1]] relative to hypotension index i0
      - post window: indices [irec+post_h[0], irec+post_h[1]] relative to recovery index irec

    If no recovery is found and allow_post_without_recovery=True, post window is defined relative to i0.
    """
    out_rows: List[pd.DataFrame] = []

    for stay_id, g in df.groupby("stay_id", sort=False):
        g = g.sort_values("t").reset_index(drop=True)

        mapz = g["map_z"].to_numpy(dtype=float)
        finite = np.isfinite(mapz)
        if finite.sum() == 0:
            continue

        hyp_idx = np.where(finite & (mapz <= hypotension_thr))[0]
        if hyp_idx.size == 0:
            continue
        i0 = int(hyp_idx[0])

        rec_idx = np.where((np.arange(len(g)) > i0) & finite & (mapz >= recovery_thr))[0]
        if rec_idx.size == 0:
            if not allow_post_without_recovery:
                continue
            irec = i0
        else:
            irec = int(rec_idx[0])

        # pre indices
        pre_lo = i0 + int(pre_h[0])
        pre_hi = i0 + int(pre_h[1])
        # post indices
        post_lo = irec + int(post_h[0])
        post_hi = irec + int(post_h[1])

        # clamp
        pre_lo = max(pre_lo, 0)
        pre_hi = min(pre_hi, len(g) - 1)
        post_lo = max(post_lo, 0)
        post_hi = min(post_hi, len(g) - 1)

        if pre_lo <= pre_hi:
            gp = g.iloc[pre_lo: pre_hi + 1].copy()
            gp["phase"] = "pre"
            out_rows.append(gp)

        if post_lo <= post_hi:
            gq = g.iloc[post_lo: post_hi + 1].copy()
            gq["phase"] = "post"
            out_rows.append(gq)

    if not out_rows:
        return pd.DataFrame(columns=list(df.columns) + ["phase"])

    return pd.concat(out_rows, axis=0).reset_index(drop=True)


def summarize_phase(actions: np.ndarray, n_actions: int) -> Dict[str, float]:
    actions = actions.astype(int).ravel()
    n = len(actions)
    if n == 0:
        return {"mean_action": np.nan, "pct_zero": np.nan, "pct_high": np.nan, "entropy_bits": np.nan, "n": 0}

    p = np.bincount(actions, minlength=n_actions).astype(float)
    p = p / p.sum()

    # entropy bits
    eps = 1e-12
    ent = -np.sum(p * np.log2(p + eps))

    return {
        "mean_action": float(np.mean(actions)),
        "pct_zero": float(p[0]),
        "pct_high": float(np.sum(p[3:])) if n_actions >= 4 else float(np.sum(p[1:])),
        "entropy_bits": float(ent),
        "n": int(n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="masds-thesis.mass_thesis.rl_dataset_1h_scaled")
    parser.add_argument("--split", default="train", choices=["train", "test", "val"])
    parser.add_argument("--device", default="cpu")

    # sampling
    parser.add_argument("--n0", type=int, default=30000)
    parser.add_argument("--n1", type=int, default=8000)
    parser.add_argument("--n2", type=int, default=8000)
    parser.add_argument("--n3", type=int, default=2000)
    parser.add_argument("--n4", type=int, default=2000)

    # hypotension definition (adjust later if desired)
    # can either set explicit thresholds OR let the script compute them from quantiles
    parser.add_argument("--hypotension_thr", type=float, default=None,
                        help="If set: map_z <= this triggers hypotension event (uses map_z != 0 only).")
    parser.add_argument("--recovery_thr", type=float, default=None,
                        help="If set: map_z >= this after event defines recovery (uses map_z != 0 only).")

    parser.add_argument("--hypotension_q", type=float, default=0.20,
                        help="Quantile to define hypotension threshold from map_z!=0 distribution (default 0.20).")
    parser.add_argument("--recovery_q", type=float, default=0.60,
                        help="Quantile to define recovery threshold from map_z!=0 distribution (default 0.60).")
    parser.add_argument("--allow_post_without_recovery", action="store_true",
                        help="If set: when no recovery is found, define post window relative to event time t0.")

    # windows
    parser.add_argument("--pre_start", type=int, default=-3)
    parser.add_argument("--pre_end", type=int, default=-1)
    parser.add_argument("--post_start", type=int, default=1)
    parser.add_argument("--post_end", type=int, default=3)

    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--clinician_only",
        action="store_true",
        help="If set, skip loading BC/CQL/BCQ and compute pre/post summaries for clinician actions only."
    )
    args = parser.parse_args()

    _ensure_dirs()

    df = load_sample_from_bigquery(args.dataset, args.split, args.n0, args.n1, args.n2, args.n3, args.n4)
    print(f"[INFO] Total transitions: {len(df)}")

    # focus analysis on map_z that is present AND not the "mostly-zero" mass
    # (in this dataset, map_z is often NULL or 0; we want actual non-zero measurements)
    # for hypotension event detection, require map_z present.
    # DO NOT drop map_z==0 here; otherwise we get too few rows and no valid windows.
    df_map = df[df["map_z"].notna()].copy()
    print(f"[INFO] map_z non-null transitions: {len(df_map)}")

    if len(df_map) == 0:
        raise RuntimeError("No usable map_z rows after filtering (map_z not null).")

    mapz_all = df_map["map_z"].astype(float).to_numpy()
    # compute thresholds if not provided
    hypot_thr = args.hypotension_thr
    rec_thr = args.recovery_thr
    if hypot_thr is None:
        hypot_thr = float(np.quantile(mapz_all, args.hypotension_q))
    if rec_thr is None:
        rec_thr = float(np.quantile(mapz_all, args.recovery_q))

    print(f"[INFO] Hypotension threshold: map_z <= {hypot_thr:.6f} "
          f"(thr={'explicit' if args.hypotension_thr is not None else f'q{args.hypotension_q:.2f}'})")
    print(f"[INFO] Recovery threshold:    map_z >= {rec_thr:.6f} "
          f"(thr={'explicit' if args.recovery_thr is not None else f'q{args.recovery_q:.2f}'})")

    # build pre/post dataset based on hypotension events
    df_phase = detect_first_hypotension_windows(
        df_map,
        hypotension_thr=hypot_thr,
        recovery_thr=rec_thr,
        pre_h=(args.pre_start, args.pre_end),
        post_h=(args.post_start, args.post_end),
        allow_post_without_recovery=args.allow_post_without_recovery,
    )
    if df_phase.empty:
        raise RuntimeError(
            "No stays matched hypotension criteria with the chosen thresholds/windows. "
            "Try widening windows, adjusting --hypotension_q/--recovery_q, or pass --allow_post_without_recovery."
        )

    print(f"[INFO] Pre/Post transitions retained: {len(df_phase)}")
    print(df_phase["phase"].value_counts(dropna=False))

    obs = _to_numpy_obs(df_phase)
    clin = df_phase["action_bin"].to_numpy().astype(int)
    n_actions = 5
    print(f"[INFO] n_actions: {n_actions}")

    # can always summarize clinician actions
    policies_actions: Dict[str, np.ndarray] = {"clinician": clin}

    # NOTE: This script builds pre/post windows on a stratified transition sample.
    # That sample does NOT contain episode boundaries (terminals/timeouts) required by d3rlpy
    # to build policies via build_with_dataset. If you want model-based pre/post (BC/CQL/BCQ),
    # you must construct a proper episodic dataset (per stay_id trajectory with terminals/timeouts)
    # or compute policy actions elsewhere and merge them back onto df_phase.
    if args.clinician_only:
        print("[INFO] clinician_only=True: skipping BC/CQL/BCQ loading and prediction.")
    else:
        print("[WARN] Attempting to load/predict BC/CQL/BCQ on a non-episodic sample may fail in d3rlpy.")
        print("[WARN] If this fails, re-run with --clinician_only (recommended for this dataset sample).")

        # best-effort attempt: only run if the user explicitly wants it
        obs_sample = obs[: min(256, len(obs))].copy()
        for name in ["bc", "cql", "bcq"]:
            try:
                algo, pred_fn = build_and_load_policy(name, obs_sample, n_actions=n_actions, device=args.device)
                acts = pred_fn(obs)
                acts = np.asarray(acts).astype(int).ravel()
                policies_actions[name] = acts
                print(f"[INFO] Predicted actions for {name}: shape={acts.shape} unique={np.unique(acts)}")
            except Exception as e:
                print(f"[WARN] Failed to load/predict for {name}: {e}")
                continue

    # summaries
    rows = []
    for policy, acts in policies_actions.items():
        for phase in ["pre", "post"]:
            mask = (df_phase["phase"].values == phase)
            s = summarize_phase(acts[mask], n_actions=n_actions)
            rows.append({
                "policy": policy,
                "phase": phase,
                "mean_action": s["mean_action"],
                "pct_zero": s["pct_zero"],
                "pct_high": s["pct_high"],
                "entropy_bits": s["entropy_bits"],
                "n": s["n"],
                "hypotension_thr": float(hypot_thr),
                "recovery_thr": float(rec_thr),
                "pre_window": f"[{args.pre_start},{args.pre_end}]",
                "post_window": f"[{args.post_start},{args.post_end}]",
                "split": args.split,
            })

    out = pd.DataFrame(rows).sort_values(["policy", "phase"]).reset_index(drop=True)
    out_path = TABLES_DIR / "pre_post_hypotension_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")

    if not args.no_plot:
        # simple plot: mean action by phase per policy
        # use whatever policies were successfully computed
        plot_df = out.pivot(index="policy", columns="phase", values="mean_action").reset_index()
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(plot_df))
        width = 0.35
        plt.bar(x - width/2, plot_df["pre"], width, label="pre")
        plt.bar(x + width/2, plot_df["post"], width, label="post")
        plt.xticks(x, plot_df["policy"], rotation=0)
        plt.ylabel("Mean action (bin)")
        # add sample sizes to title for context
        n_pre = int((df_phase["phase"].values == "pre").sum())
        n_post = int((df_phase["phase"].values == "post").sum())
        plt.title(
            f"Pre vs Post Hypotension Mean Action (map_z<= {hypot_thr:.3f}, rec>= {rec_thr:.3f}; n_pre={n_pre}, n_post={n_post})"
        )
        plt.legend()
        fig_path = FIG_DIR / "pre_post_hypotension_mean_action.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[OK] Wrote {fig_path}")


if __name__ == "__main__":
    main()
