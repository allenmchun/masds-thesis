import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from google.cloud import bigquery

from d3rlpy.algos import DiscreteBC, DiscreteCQL, DiscreteBCQ
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig, DiscreteBCQConfig
from d3rlpy.dataset import MDPDataset

import inspect

def safe_call(method, **kwargs):
    sig = inspect.signature(method)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return method(**filtered)

# bigquery configuration
PROJECT_ID = "masds-thesis"
TABLE_FQDN = "masds-thesis.mass_thesis.rl_dataset_1h_scaled"

DATASET_LIMIT = 50000

STATE_COLS = [
    "age_years", "gender",
    "map_z", "hr_z", "spo2_z", "urine_ml_z",
    "lactate_z", "creatinine_z", "ph_z", "d4h_map_z",
    "ne_rate_z", "epi_rate_z", "phe_rate_z", "vasopressin_rate_z",
    "miss_map", "miss_hr", "miss_spo2",
    "miss_lactate", "miss_creatinine", "miss_ph",
]

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from typing import Optional

def fetch_icu_dataset_from_bigquery(
    project_id: str,
    table_fqdn: str,
    limit: Optional[int] = None,
):

    """
    Fetch offline ICU dataset from BigQuery (rl_dataset_1h_scaled)
    and return as an MDPDataset for d3rlpy.

    Uses:
      - State features: age_years, gender,
        *_z vitals/labs, pressor_z, and missingness flags
      - Action:  action_bin
      - Reward:  reward
      - Done:    done
      - Split:   filters to split = 'train'
    """

    client = bigquery.Client(project=project_id)

    # note: we stratify sampling by action_bin to ensure rare bins (3/4) appear in dev runs.
    # this avoids d3rlpy inferring a smaller action_size when using small samples.
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
        action_bin,
        reward,
        done
      FROM `{table_fqdn}`
      WHERE split = 'train'
    ),
    sampled AS (
      SELECT *
      FROM base
      QUALIFY
        ROW_NUMBER() OVER (PARTITION BY action_bin ORDER BY RAND()) <=
          CASE action_bin
            WHEN 0 THEN 30000
            WHEN 1 THEN  8000
            WHEN 2 THEN  8000
            WHEN 3 THEN  2000
            WHEN 4 THEN  2000
            ELSE 0
          END
    )
    SELECT * FROM sampled
    """


    print(f"[INFO] Running BigQuery:\n{query}")
    df = client.query(query).to_dataframe()

    if df.empty:
        raise RuntimeError("BigQuery returned an empty dataframe. "
                           "Check TABLE_FQDN, split values, and permissions.")

    # build state vector (observations)
    state_cols = STATE_COLS

    import pandas as pd
    import numpy as np

# sanitize state features to prevent NaNs in network
    state_df = df[state_cols].apply(pd.to_numeric, errors="coerce")

# replace inf/-inf with NaN
    state_df = state_df.replace([np.inf, -np.inf], np.nan)

# count bad rows
    n_bad = state_df.isna().any(axis=1).sum()
    if n_bad > 0:
        print(f"[WARN] Found {n_bad} rows with NaN/Inf in state features. Filling with 0.0.")

# fill NaNs with 0.0 (safe because miss_* flags exist)
    state_df = state_df.fillna(0.0)

# final conversion
    obs = state_df.to_numpy(dtype="float32")

    act = df["action_bin"].to_numpy(dtype="int64")
    print("[INFO] action_bin unique:", np.unique(act), "n_actions:", len(np.unique(act)))
    rew = df["reward"].to_numpy(dtype="float32")

    done_col = df["done"]
    if done_col.dtype == bool:
        done = done_col.to_numpy()
    else:
        done = done_col.to_numpy().astype(int).astype(bool)

    print(f"[INFO] Loaded dataset from BigQuery:")
    print(f"       N transitions:      {len(df)}")
    print(f"       obs shape:          {obs.shape}")
    print(f"       actions shape:      {act.shape}")
    print(f"       rewards shape:      {rew.shape}")
    print(f"       terminals shape:    {done.shape}")

    return MDPDataset(
        observations=obs,
        actions=act,
        rewards=rew,
        terminals=done,
    )


def load_dataset() -> MDPDataset:
    """
    Wrapper to load ICU offline dataset from BigQuery.
    Uses module-level PROJECT_ID, TABLE_FQDN, and DATASET_LIMIT.
    """
    if PROJECT_ID == "YOUR_GCP_PROJECT_ID":
        raise ValueError(
            "Please set PROJECT_ID and TABLE_FQDN at the top of train.py "
            "to your actual GCP project and BigQuery table."
        )

    return fetch_icu_dataset_from_bigquery(
        project_id=PROJECT_ID,
        table_fqdn=TABLE_FQDN,
        limit=DATASET_LIMIT,
    )


def build_algo(cfg):
    # allow either:
    #   algorithm: "bc"
    # or
    #   algorithm: { name: "bc", ... }
    algo_section = cfg.get("algorithm", "bc")
    if isinstance(algo_section, dict):
        algo_name = str(algo_section.get("name", "bc")).lower()
        algo_cfg = algo_section
    else:
        algo_name = str(algo_section).lower()
        algo_cfg = {}

    device_str = str(cfg.get("device", "cpu")).lower()
    device = "cuda:0" if device_str in ("cuda", "gpu") else "cpu"
    enable_ddp = False

    if algo_name == "bc":
        config = DiscreteBCConfig()
        return DiscreteBC(config, device, enable_ddp)

    if algo_name == "cql":
        import inspect

        training_cfg = cfg.get("training", {})

        # candidate hyperparams to control via YAML
        candidate = dict(
        batch_size=int(training_cfg.get("batch_size", 256)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        alpha=float(algo_cfg.get("alpha", 1.0)),
        tau=float(algo_cfg.get("tau", 0.005)),          # will be ignored if unsupported
        n_critics=int(algo_cfg.get("n_critics", 2)),    # will be ignored if unsupported
        )

        # keep only args that DiscreteCQLConfig actually supports in our d3rlpy
        sig = inspect.signature(DiscreteCQLConfig)
        filtered = {k: v for k, v in candidate.items() if k in sig.parameters}

        print(f"[INFO] DiscreteCQLConfig args used: {filtered}")

        config = DiscreteCQLConfig(**filtered)
        return DiscreteCQL(config, device, enable_ddp)

    elif algo_name == "bcq":
        a = algo_cfg
        config = DiscreteBCQConfig(
            batch_size=int(a.get("batch_size", 256)),
            gamma=float(a.get("gamma", 0.99)),
            n_critics=int(a.get("n_critics", 2)),
        )
        print(f"[INFO] DiscreteBCQConfig args used: "
              f"{{'batch_size': {config.batch_size}, 'gamma': {config.gamma}, 'n_critics': {config.n_critics}}}")

        # device handling – keep consistent with other branches
        device = cfg.get("device", "cpu")
        return DiscreteBCQ(config=config, device=device, enable_ddp=False)


    raise ValueError(f"Unknown algorithm in config: {algo_name}")



def main(config_path: str) -> None:
    # load YAML config
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"[INFO] Using seed: {seed}")

    # load dataset
    dataset = load_dataset()

    # build algorithm
    algo = build_algo(cfg)

    # prepare logging dirs
    project_root = Path(__file__).resolve().parents[1]
    experiment_name = cfg.get("experiment_name", "experiment")

    log_root = project_root / "results" / "logs"
    log_dir = log_root / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")

    # train with d3rlpy
    training_cfg = cfg["training"]

    # can keep n_epochs in YAML as convenient knob
    n_epochs = int(training_cfg.get("n_epochs", 50))

    # d3rlpy fit() uses steps, not epochs
    n_steps_per_epoch = int(training_cfg.get("n_steps_per_epoch", 10000))
    n_steps = n_epochs * n_steps_per_epoch

    print(f"[INFO] Starting training:")
    print(f"       n_epochs          = {n_epochs}")
    print(f"       n_steps_per_epoch = {n_steps_per_epoch}")
    print(f"       n_steps           = {n_steps}")

    algo.fit(
        dataset=dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=experiment_name,
        with_timestamp=False,
        show_progress=True,
        save_interval=1,
    )

    # also save torch-loadable checkpoint for downstream plots
    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    torch_ckpt_path = models_dir / f"{experiment_name}.pt"
    algo.save_model(str(torch_ckpt_path))
    print(f"[INFO] Torch checkpoint saved to: {torch_ckpt_path}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g., config/bc_v1.yaml)")
    args = parser.parse_args()

    main(args.config)
