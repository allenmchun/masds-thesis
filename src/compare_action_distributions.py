import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# allow importing src/train.py
sys.path.append("src")
import train

from d3rlpy.algos import DiscreteBC, DiscreteBCConfig

BC_EXPERIMENT_NAME = "bc_final_ne5_v2"
CQL_EXPERIMENT_NAME = "cql_final_ne5_v2"

BC_MODEL_PT = Path("results/models") / f"{BC_EXPERIMENT_NAME}.pt"
CQL_MODEL_PT = Path("results/models") / f"{CQL_EXPERIMENT_NAME}.pt"

OUT_CSV = Path("results/tables/action_distribution_comparison_bc_vs_cql_vs_clinician.csv")
OUT_PNG = Path("results/figures/action_distribution_comparison_bc_vs_cql_vs_clinician.png")


# helpers
class D3RLPyEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class D3RLPyDiscreteQFunc(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self._encoder = D3RLPyEncoder(obs_dim, hidden_dim)
        self._fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return self._fc(h)


def infer_dims_from_q_state_dict(q_sd: dict):
    # "0._encoder._layers.0.weight" has shape (hidden_dim, obs_dim)
    w0 = q_sd["0._encoder._layers.0.weight"]
    hidden_dim, obs_dim = w0.shape
    # "0._fc.weight" has shape (n_actions, hidden_dim)
    wfc = q_sd["0._fc.weight"]
    n_actions = wfc.shape[0]
    return int(obs_dim), int(hidden_dim), int(n_actions)


def extract_obs_and_clin_actions(dataset):
    episodes = dataset.episodes
    obs = np.concatenate([ep.observations for ep in episodes], axis=0).astype(np.float32)
    clin = np.concatenate([ep.actions for ep in episodes], axis=0).astype(int).reshape(-1)
    return obs, clin


def proportions(actions: np.ndarray, n_actions: int) -> np.ndarray:
    bins = np.arange(n_actions + 1) - 0.5
    counts, _ = np.histogram(actions, bins=bins)
    return counts / counts.sum()


def load_bc_and_predict(dataset, obs: np.ndarray) -> np.ndarray:
    if not BC_MODEL_PT.exists():
        raise FileNotFoundError(f"Missing BC checkpoint: {BC_MODEL_PT.resolve()}")

    # build + load BC
    cfg = DiscreteBCConfig()  # default encoder sizes consistent with training unless customized
    bc = DiscreteBC(config=cfg, device="cpu", enable_ddp=False)
    bc.build_with_dataset(dataset)
    bc.load_model(str(BC_MODEL_PT))

    pred = bc.predict(obs)
    return np.asarray(pred, dtype=int).reshape(-1)


def load_cql_and_predict(obs: np.ndarray) -> Tuple[np.ndarray, int]:
    if not CQL_MODEL_PT.exists():
        raise FileNotFoundError(f"Missing CQL checkpoint: {CQL_MODEL_PT.resolve()}")

    chk = torch.load(CQL_MODEL_PT, map_location="cpu")
    q_sd = chk["q_funcs"]

    obs_dim, hidden_dim, n_actions = infer_dims_from_q_state_dict(q_sd)

    q0 = D3RLPyDiscreteQFunc(obs_dim, hidden_dim, n_actions)
    q1 = D3RLPyDiscreteQFunc(obs_dim, hidden_dim, n_actions)

    q0_sd = {k.replace("0.", "", 1): v for k, v in q_sd.items() if k.startswith("0.")}
    q1_sd = {k.replace("1.", "", 1): v for k, v in q_sd.items() if k.startswith("1.")}

    q0.load_state_dict(q0_sd, strict=True)
    q1.load_state_dict(q1_sd, strict=True)
    q0.eval(); q1.eval()

    with torch.no_grad():
        x = torch.from_numpy(obs)
        Q0 = q0(x).cpu().numpy()
        Q1 = q1(x).cpu().numpy()

    Q = 0.5 * (Q0 + Q1)
    pred = np.argmax(Q, axis=1).astype(int)
    return pred, n_actions


def main():
    dataset = train.load_dataset()
    obs, clin = extract_obs_and_clin_actions(dataset)

    # determine action size
    if hasattr(dataset, "dataset_info") and hasattr(dataset.dataset_info, "action_size"):
        n_actions = int(dataset.dataset_info.action_size)
    else:
        n_actions = int(clin.max() + 1)

    # predictions
    bc_pred = load_bc_and_predict(dataset, obs)
    cql_pred, n_actions_cql = load_cql_and_predict(obs)
    n_actions = max(n_actions, n_actions_cql)

    # proportions
    clin_pct = proportions(clin, n_actions)
    bc_pct = proportions(bc_pred, n_actions)
    cql_pct = proportions(cql_pred, n_actions)

    # table
    df = pd.DataFrame({
        "action": np.arange(n_actions),
        "clinician": clin_pct,
        "bc": bc_pct,
        "cql_greedy": cql_pct,
    })
    df["bc_minus_clinician"] = df["bc"] - df["clinician"]
    df["cql_minus_clinician"] = df["cql_greedy"] - df["clinician"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n[INFO] Action distribution comparison (proportions)")
    print(df.to_string(index=False))
    print(f"\n[OK] Wrote {OUT_CSV}")

    # plot
    x = np.arange(n_actions)
    width = 0.28

    plt.figure()
    plt.bar(x - width, clin_pct, width, label="Clinician")
    plt.bar(x,         bc_pct,   width, label="BC")
    plt.bar(x + width, cql_pct,  width, label="CQL (greedy argmax Q)")
    plt.xticks(x, [str(i) for i in x])
    plt.xlabel("action_bin")
    plt.ylabel("proportion")
    plt.title("Action distribution: Clinician vs BC vs CQL")
    plt.legend()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
