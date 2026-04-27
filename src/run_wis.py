# src/run_wis.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import train  # uses existing load_dataset + STATE_COLS etc.

from d3rlpy.algos import DiscreteBC, DiscreteCQL, DiscreteBCQ
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig, DiscreteBCQConfig


# model checkpoints (.pt exported by train.py)
BC_PT  = Path("results/models/bc_final_ne5_v2.pt")
CQL_PT = Path("results/models/cql_final_ne5_v2.pt")
BCQ_PT = Path("results/models/bcq_best10k_ne5_v2.pt")  # best@10k (rerun)

OUT_CSV = Path("results/tables/wis_pd_summary.csv")


def safe_predict_proba(algo, obs: np.ndarray, n_actions: int) -> np.ndarray:
    """
    Try to get action probabilities from d3rlpy.
    Fallback: epsilon-smoothed deterministic (based on greedy predict).
    """
    # preferred: stochastic probs if available
    if hasattr(algo, "predict_proba"):
        p = algo.predict_proba(obs)
        p = np.asarray(p)
        # ensure shape (N, A)
        if p.ndim == 1:
            p = p.reshape(-1, n_actions)
        return p

    # fallback: epsilon-smoothed greedy
    greedy = np.asarray(algo.predict(obs)).reshape(-1)
    eps = 1e-3
    p = np.full((len(obs), n_actions), eps / (n_actions - 1), dtype=np.float64)
    p[np.arange(len(obs)), greedy] = 1.0 - eps
    return p


def extract_episodes(dataset):
    """
    d3rlpy MDPDataset should expose episodes.
    """
    if hasattr(dataset, "episodes"):
        return dataset.episodes
    raise RuntimeError("Dataset has no .episodes attribute. Your load_dataset() must return an MDPDataset.")


def pd_wis(episodes, pi_e, pi_b, n_actions: int, gamma: float = 0.99,
           max_log_w: float = 20.0, max_episodes=None):
    """
    Per-decision weighted importance sampling (PD-WIS).
    - pi_b is behavior estimator (BC)
    - pi_e is evaluation policy (CQL or BCQ)
    """
    numer = 0.0
    denom = 0.0

    # track effective sample size proxy (not perfect but useful)
    # ESS per-decision: (sum w)^2 / sum(w^2)
    ess_num = 0.0
    ess_den = 0.0

    used = 0

    for ep in episodes:
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.int64).reshape(-1)
        rew = np.asarray(ep.rewards, dtype=np.float32).reshape(-1)

        T = len(act)
        if T == 0:
            continue

        pe = safe_predict_proba(pi_e, obs, n_actions)
        pb = safe_predict_proba(pi_b, obs, n_actions)

        # clip to avoid div-by-zero
        pb_a = np.clip(pb[np.arange(T), act], 1e-8, 1.0)
        pe_a = np.clip(pe[np.arange(T), act], 1e-8, 1.0)

        log_ratio = np.log(pe_a) - np.log(pb_a)

        # accumulate per-decision weights in log-space
        log_w = 0.0
        for t in range(T):
            log_w = log_w + float(log_ratio[t])
            log_w_clipped = float(np.clip(log_w, -max_log_w, max_log_w))
            w_t = float(np.exp(log_w_clipped))

            g_t = (gamma ** t) * float(rew[t])

            numer += w_t * g_t
            denom += w_t

            ess_num += w_t
            ess_den += w_t * w_t

        used += 1
        if max_episodes is not None and used >= max_episodes:
            break

    wis = numer / denom if denom > 0 else np.nan
    ess = (ess_num * ess_num / ess_den) if ess_den > 0 else np.nan
    return wis, denom, ess, used


def build_and_load(algo_cls, config_cls, pt_path: Path, dataset, device="cpu"):
    """Build algo with dataset, then load a .pt checkpoint.

    In your environment, d3rlpy .d3 files are not torch-loadable. Your train.py exports
    .pt checkpoints. Some algorithms save nested dicts (e.g., q_funcs/targ_q_funcs), so
    we try algo.load_model first, then fall back to manual loading.
    """
    config = config_cls()  # don't pass kwargs; signatures differ across configs
    algo = algo_cls(config, device=device, enable_ddp=False)
    algo.build_with_dataset(dataset)

    # first try official loader
    try:
        algo.load_model(str(pt_path))
        return algo
    except Exception as e:
        print(f"[WARN] algo.load_model failed for {pt_path}: {e}")

    # fallback: manual load for nested checkpoints
    import torch

    chkpt = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(chkpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(chkpt)}")

    # case A: checkpoint is directly the impl state_dict
    try:
        algo._impl.load_state_dict(chkpt, strict=False)
        print("[INFO] Loaded checkpoint into algo._impl with strict=False")
        return algo
    except Exception:
        pass

    # case B: checkpoint has nested keys (commonly q_funcs / targ_q_funcs / policy)
    # load what we can
    if hasattr(algo, "_impl") and algo._impl is not None:
        impl = algo._impl

        # q-functions
        if "q_funcs" in chkpt and hasattr(impl, "q_funcs"):
            try:
                impl.q_funcs.load_state_dict(chkpt["q_funcs"], strict=False)
                print("[INFO] Loaded chkpt['q_funcs'] into impl.q_funcs")
            except Exception as e:
                print(f"[WARN] Failed loading q_funcs: {e}")

        if "targ_q_funcs" in chkpt and hasattr(impl, "targ_q_funcs"):
            try:
                impl.targ_q_funcs.load_state_dict(chkpt["targ_q_funcs"], strict=False)
                print("[INFO] Loaded chkpt['targ_q_funcs'] into impl.targ_q_funcs")
            except Exception as e:
                print(f"[WARN] Failed loading targ_q_funcs: {e}")

        # behavior cloning policy (if present)
        if "policy" in chkpt and hasattr(impl, "policy"):
            try:
                impl.policy.load_state_dict(chkpt["policy"], strict=False)
                print("[INFO] Loaded chkpt['policy'] into impl.policy")
            except Exception as e:
                print(f"[WARN] Failed loading policy: {e}")

    return algo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max_episodes", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # load dataset (our train.py handles BigQuery + split filtering internally)
    dataset = train.load_dataset()
    episodes = extract_episodes(dataset)

    # infer action size
    # already print n_actions in train; but we infer again for safety
    all_actions = []
    for ep in episodes[: min(50, len(episodes))]:
        all_actions.append(np.asarray(ep.actions).reshape(-1))
    all_actions = np.concatenate(all_actions) if len(all_actions) else np.array([], dtype=np.int64)
    n_actions = int(all_actions.max() + 1) if all_actions.size else 5

    print(f"[INFO] Episodes available: {len(episodes)}")
    print(f"[INFO] Using n_actions={n_actions}, gamma={args.gamma}, max_episodes={args.max_episodes}")
    print(f"[INFO] Behavior estimator (pi_b): BC @ {BC_PT}")
    print(f"[INFO] Eval policies: CQL @ {CQL_PT}, BCQ(best@10k) @ {BCQ_PT}")

    # load algos
    bc  = build_and_load(DiscreteBC,  DiscreteBCConfig,  BC_PT,  dataset, args.device)
    cql = build_and_load(DiscreteCQL, DiscreteCQLConfig, CQL_PT, dataset, args.device)
    bcq = build_and_load(DiscreteBCQ, DiscreteBCQConfig, BCQ_PT, dataset, args.device)

    rows = []

    for name, pi_e in [
        ("CQL", cql),
        ("BCQ_best10k", bcq),
        # can compute WIS for BC vs BC (will equal behavior return estimate),
        # but not informative; leaving out by default
    ]:
        wis, denom, ess, used = pd_wis(
            episodes=episodes,
            pi_e=pi_e,
            pi_b=bc,              # behavior estimator
            n_actions=n_actions,
            gamma=args.gamma,
            max_log_w=20.0,
            max_episodes=args.max_episodes,
        )
        print(f"[RESULT] {name}: PD-WIS={wis:.6f}  (used_episodes={used}, weight_sum={denom:.3e}, ESS~{ess:.1f})")
        rows.append({
            "policy": name,
            "gamma": args.gamma,
            "episodes_used": used,
            "pd_wis": wis,
            "weight_sum": denom,
            "ess_proxy": ess,
            "behavior_estimator": "BC",
            "bc_model": str(BC_PT),
            "policy_model": str(CQL_PT if name == "CQL" else BCQ_PT),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
