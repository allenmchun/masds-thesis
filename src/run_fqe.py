#!/usr/bin/env python3
"""
run_fqe.py

Fitted Q Evaluation (FQE) for discrete policies (BC / CQL / BCQ) using d3rlpy.

What it does:
1) Loads your offline dataset via train.load_dataset() (ReplayBufferBase).
2) Loads trained policies (BC/CQL/BCQ) from .pt checkpoints (recommended) OR .d3 if your env supports it.
3) Trains DiscreteFQE for each target policy on the same dataset.
4) Estimates V^pi by averaging Q_fqe(s0, a_pi(s0)) over episode initial states.

Outputs:
- results/tables/fqe_summary.csv
- results/logs/fqe_<policy_name>/*  (d3rlpy logs)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# d3rlpy imports (v2.x; handle minor API differences)
import d3rlpy

try:
    # some d3rlpy versions provide a Config class
    from d3rlpy.ope import DiscreteFQE, DiscreteFQEConfig  # type: ignore
except ImportError:
    # fallback: DiscreteFQE exists but Config class name differs / is absent
    from d3rlpy.ope import DiscreteFQE  # type: ignore
    DiscreteFQEConfig = None  # type: ignore

def _resolve_discrete_fqe_config(gamma: float):
    """Resolve an available FQE config class across d3rlpy versions."""
    import inspect

    # if already imported DiscreteFQEConfig successfully, use it
    if DiscreteFQEConfig is not None:
        return DiscreteFQEConfig(gamma=gamma)  # type: ignore

    # otherwise search for any class name that ends with 'FQEConfig' under d3rlpy.ope
    import d3rlpy.ope as ope

    candidates = [
        name for name in dir(ope)
        if name.lower().endswith("fqeconfig") and "discrete" in name.lower()
    ]
    # fallback: any FQEConfig if discrete-specific isn't present.
    if not candidates:
        candidates = [name for name in dir(ope) if name.lower().endswith("fqeconfig")]

    if not candidates:
        raise ImportError(
            "Could not find any *FQEConfig class in d3rlpy.ope. "
            "Please check your d3rlpy version / installation."
        )

    cfg_cls = getattr(ope, candidates[0])

    # instantiate with gamma if supported.
    sig = inspect.signature(cfg_cls)
    if "gamma" in sig.parameters:
        return cfg_cls(gamma=gamma)

    # some configs may use different names or no args.
    return cfg_cls()

# policies (configs)
from d3rlpy.algos import (
    DiscreteBC,
    DiscreteCQL,
    DiscreteBCQ,
)
from d3rlpy.algos import (
    DiscreteBCConfig,
    DiscreteCQLConfig,
    DiscreteBCQConfig,
)

# our train.py
import train


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PolicySpec:
    name: str
    algo_cls: object
    cfg_cls: object
    ckpt_path: Path  # .pt preferred (consistent with current workflow)
    # minimal config kwargs; keep aligned with what we used during training
    cfg_kwargs: Dict


def _infer_n_actions_from_dataset(dataset) -> int:
    # ReplayBufferBase has dataset_info in d3rlpy v2.x
    info = dataset.dataset_info
    if getattr(info, "action_size", None) is None:
        raise RuntimeError("Could not infer action_size from dataset.dataset_info.")
    return int(info.action_size)


def _get_initial_observations(dataset, max_episodes: int | None = None) -> np.ndarray:
    """
    Returns array of initial states s0 for each episode.
    dataset is expected to be ReplayBufferBase with .episodes.
    """
    episodes = list(dataset.episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    s0_list = []
    for ep in episodes:
        # in d3rlpy v2.x Episode has .observations
        s0 = np.asarray(ep.observations[0], dtype=np.float32)
        s0_list.append(s0)

    if len(s0_list) == 0:
        raise RuntimeError("No episodes found in dataset (dataset.episodes empty).")

    return np.stack(s0_list, axis=0).astype(np.float32)


def _build_and_load_policy(spec: PolicySpec, dataset, device: str):
    """
    Builds algo from Config().create(), initializes impl with dataset, then loads weights.
    """
    config = spec.cfg_cls(**spec.cfg_kwargs)
    algo = config.create(device=device)

    # in d3rlpy v2.x we must build before load_model
    algo.build_with_dataset(dataset)

    if not spec.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec.ckpt_path}")

    algo.load_model(str(spec.ckpt_path))
    return algo


def run_fqe_for_policy(
    policy_algo,
    dataset,
    policy_name: str,
    device: str,
    gamma: float,
    n_steps: int,
    n_steps_per_epoch: int,
    logging_steps: int,
) -> Tuple[float, float]:
    """
    Trains FQE and returns (mean_value, std_value) across episode initial states.
    """
    # construct FQE (API differs slightly across d3rlpy versions)
    fqe_cfg = _resolve_discrete_fqe_config(gamma)

    import inspect

    sig = inspect.signature(DiscreteFQE)
    kwargs = {"algo": policy_algo, "config": fqe_cfg}
    if "device" in sig.parameters:
        kwargs["device"] = device
    fqe = DiscreteFQE(**kwargs)

    exp_name = f"fqe_{policy_name}"
    logdir = LOGS_DIR / exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    # train FQE Q-function
    fqe.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=exp_name,
        with_timestamp=False,
        logging_steps=logging_steps,
        show_progress=True,
        save_interval=1,
    )

    # estimate V^pi from initial states: V(s0) = Q_fqe(s0, a_pi(s0))
    s0 = _get_initial_observations(dataset, max_episodes=None)  # use all
    a0 = policy_algo.predict(s0)  # greedy action
    a0 = np.asarray(a0).reshape(-1)

    v0 = fqe.predict_value(s0, a0)  # shape (n_episodes,)
    v0 = np.asarray(v0, dtype=np.float32)

    return float(v0.mean()), float(v0.std(ddof=1) if len(v0) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--n_steps_per_epoch", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)

    # if we want to match our “final” set exactly, point these to our official .pt files
    parser.add_argument("--bc_pt", type=str, default="results/models/bc_final_ne5_v2.pt")
    parser.add_argument("--cql_pt", type=str, default="results/models/cql_final_ne5_v2.pt")
    parser.add_argument("--bcq_pt", type=str, default="results/models/bcq_best10k_ne5_v2.pt")

    parser.add_argument("--out_csv", type=str, default="results/tables/fqe_summary.csv")
    args = parser.parse_args()

    # load dataset (ReplayBufferBase)
    dataset = train.load_dataset()
    n_actions = _infer_n_actions_from_dataset(dataset)
    print(f"[INFO] Loaded dataset. action_size={n_actions}")

    # fefine policies to evaluate
    # (keep cfg minimal; FQE uses policy only for pi(a|s) / greedy action)
    policies: List[PolicySpec] = [
        PolicySpec(
            name="bc_final_ne5_v2",
            algo_cls=DiscreteBC,
            cfg_cls=DiscreteBCConfig,
            ckpt_path=PROJECT_ROOT / args.bc_pt,
            cfg_kwargs=dict(batch_size=256, gamma=args.gamma),
        ),
        PolicySpec(
            name="cql_final_ne5_v2",
            algo_cls=DiscreteCQL,
            cfg_cls=DiscreteCQLConfig,
            ckpt_path=PROJECT_ROOT / args.cql_pt,
            cfg_kwargs=dict(batch_size=256, gamma=args.gamma, alpha=1.0, n_critics=2),
        ),
        PolicySpec(
            name="bcq_best10k_ne5_v2",
            algo_cls=DiscreteBCQ,
            cfg_cls=DiscreteBCQConfig,
            ckpt_path=PROJECT_ROOT / args.bcq_pt,
            cfg_kwargs=dict(batch_size=256, gamma=args.gamma, n_critics=2),
        ),
    ]

    rows = []
    for spec in policies:
        print(f"\n[INFO] === FQE for {spec.name} ===")
        algo = _build_and_load_policy(spec, dataset, device=args.device)

        mean_v, std_v = run_fqe_for_policy(
            policy_algo=algo,
            dataset=dataset,
            policy_name=spec.name,
            device=args.device,
            gamma=args.gamma,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            logging_steps=args.logging_steps,
        )

        print(f"[RESULT] {spec.name}: FQE mean(V)={mean_v:.6f}, std(V)={std_v:.6f}")
        rows.append(
            dict(
                policy=spec.name,
                gamma=args.gamma,
                n_steps=args.n_steps,
                n_steps_per_epoch=args.n_steps_per_epoch,
                fqe_mean_value=mean_v,
                fqe_std_value=std_v,
                ckpt=str(spec.ckpt_path),
            )
        )

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()