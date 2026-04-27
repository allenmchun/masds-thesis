from pathlib import Path
from d3rlpy import load_learnable
import numpy as np
import pandas as pd

import sys
sys.path.append("src")
import train

from d3rlpy.algos import (
    DiscreteBC, DiscreteBCConfig,
    DiscreteCQL, DiscreteCQLConfig,
    DiscreteBCQ, DiscreteBCQConfig,
)

OUT_PATH = Path("results/tables/policy_summary.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ACTIONS = np.arange(5)


# helpers

def entropy_from_probs(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(p):
    return float(entropy_from_probs(p) / np.log(len(p)))


def summarize_actions(policy, actions):
    actions = actions.astype(int)
    probs = np.array([(actions == a).mean() for a in ACTIONS])

    return {
        "policy": policy,
        "mean_action": float(actions.mean()),
        "entropy_nats": entropy_from_probs(probs),
        "normalized_entropy": normalized_entropy(probs),
        "pct_zero_action": float(probs[0]),
        "pct_extreme_action_3_4": float(probs[3] + probs[4]),
    }


def predict_actions(algo, dataset):
    preds = []
    for ep in dataset.episodes:
        obs = ep.observations
        a = algo.predict(obs)
        preds.append(np.asarray(a).reshape(-1))
    return np.concatenate(preds)


# main

def main():
    dataset = train.load_dataset()

    # clinician
    clinician_actions = []
    for ep in dataset.episodes:
        clinician_actions.append(np.asarray(ep.actions).reshape(-1))
    clinician_actions = np.concatenate(clinician_actions)

    rows = []
    rows.append(summarize_actions("clinician", clinician_actions))

    # BC
    bc_cfg = DiscreteBCConfig()
    bc = DiscreteBC(config=bc_cfg, device="cpu", enable_ddp=False)
    bc.build_with_dataset(dataset)
    bc.load_model("results/models/bc_final_ne5_v2.pt")

    bc_actions = predict_actions(bc, dataset)
    rows.append(summarize_actions("bc", bc_actions))

    # CQL
    cql_cfg = DiscreteCQLConfig(
        batch_size=256,
        gamma=0.99,
        alpha=1.0,
        n_critics=2,
    )
    cql = DiscreteCQL(config=cql_cfg, device="cpu", enable_ddp=False)
    cql.build_with_dataset(dataset)
    cql.load_model("results/models/cql_final_ne5_v2.pt")

    cql_actions = predict_actions(cql, dataset)
    rows.append(summarize_actions("cql", cql_actions))

    # BCQ
    BCQ_D3_PATH = "d3rlpy_logs/bcq_final_ne5_v2/model_10000.d3"
    bcq = load_learnable(BCQ_D3_PATH)

    bcq_actions = predict_actions(bcq, dataset)  # works because load_learnable returns a learnable with .predict
    rows.append(summarize_actions("bcq_best10k", bcq_actions))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"[OK] Wrote {OUT_PATH}")
    print(df)


if __name__ == "__main__":
    main()
