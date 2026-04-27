import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from d3rlpy.ope import DiscreteFQE
# FQE config class name/location differs across d3rlpy versions.
# we try a few known locations and fall back to None (then we will filter params and pass minimal defaults).
try:
    from d3rlpy.ope import FQEConfig as _FQEConfig  # type: ignore
except Exception:  # pragma: no cover
    try:
        from d3rlpy.ope import DiscreteFQEConfig as _FQEConfig  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from d3rlpy.ope.fqe import FQEConfig as _FQEConfig  # type: ignore
        except Exception:  # pragma: no cover
            _FQEConfig = None  # type: ignore

from d3rlpy.dataset import ReplayBufferBase
from d3rlpy.algos import DiscreteBC, DiscreteCQL, DiscreteBCQ
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig, DiscreteBCQConfig

from train import load_dataset

import json
import inspect


def get_initial_states_from_replay_buffer(buffer: ReplayBufferBase) -> np.ndarray:
    """
    One initial observation per episode.
    """
    episodes = buffer.episodes
    if episodes is None or len(episodes) == 0:
        raise RuntimeError("No episodes found in ReplayBuffer. Cannot bootstrap over episodes.")

    init_obs = np.stack([ep.observations[0] for ep in episodes], axis=0).astype(np.float32)
    return init_obs


def pick_latest_model(d: Path) -> Path:
    """
    Pick the highest-step model_XXXXX.d3 in a directory.
    """
    models = list(d.glob("model_*.d3"))
    if not models:
        raise FileNotFoundError(f"No model_*.d3 found in {d}")

    def step(p: Path) -> int:
        # model_50000.d3 -> 50000
        s = p.stem.split("_")[-1]
        return int(s)

    return max(models, key=step)


# new helper: build_algo_from_run_dir
def build_algo_from_run_dir(
    algo_cls,
    config_cls,
    run_dir: Path,
    dataset: ReplayBufferBase,
    device: str,
    policy_name: str,
):
    """Rebuild an algo with the same config used during training (from params.json), then load latest model_*.d3."""
    params_path = run_dir / "params.json"
    cfg_params: dict = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            root = json.load(f)
        cfg_params = (root.get("config", {}) or {}).get("params", {}) or {}

    # filter params to those accepted by the Config class in this installed d3rlpy
    sig = inspect.signature(config_cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in cfg_params.items() if k in allowed}

    # IMPORTANT: sanitize serialized objects from params.json
    # params.json stores factories/scalers as dicts (e.g., {"type": "vector", ...})
    # the d3rlpy Config expects actual objects (EncoderFactory / OptimizerFactory / *Scaler),
    # so passing dicts will crash later (e.g., "'dict' object has no attribute 'create'").
    # for bootstrap evaluation we only need *compatible* configs; defaults are fine.
    drop_if_dict = {
        # factories
        "actor_encoder_factory",
        "critic_encoder_factory",
        "value_encoder_factory",
        "encoder_factory",
        "q_func_factory",
        "policy_encoder_factory",
        "imitator_encoder_factory",
        "actor_optim_factory",
        "critic_optim_factory",
        "optim_factory",
        # scalers
        "observation_scaler",
        "action_scaler",
        "reward_scaler",
    }

    for k in list(filtered.keys()):
        v = filtered[k]
        if k in drop_if_dict and isinstance(v, dict):
            filtered.pop(k, None)

    # d3rlpy asserts that discrete-action algorithms must NOT use action_scaler
    # even if it sneaks in as a non-dict, force-disable it
    filtered["action_scaler"] = None

    config = config_cls(**filtered)
    print(f"[INFO] {run_dir.name}: config params used (sanitized): {filtered}")
    algo = algo_cls(config=config, device=device, enable_ddp=False)
    algo.build_with_dataset(dataset)

    # prefer loading the d3rlpy .d3 checkpoint from the run dir
    # some older runs (or cross-version runs) can yield an "invalid magic number" when torch tries to load
    model_path = pick_latest_model(run_dir)
    try:
        algo.load_model(str(model_path))
    except Exception as e:
        msg = str(e).lower()
        if "invalid magic number" in msg or "corrupt" in msg:
            print(
                f"[WARN] Failed to load {model_path} due to invalid magic number. "
                f"Falling back to a torch .pt checkpoint for {policy_name}."
            )
            # for BC we usually have weights under 'imitator'; for CQL/BCQ we have q_funcs/targ_q_funcs
            desired_keys = ["imitator", "q_funcs", "targ_q_funcs", "policy"]
            pt_path = _find_compatible_pt(policy_name, desired_keys)
            print(f"[INFO] Using torch checkpoint: {pt_path}")
            _load_torch_checkpoint_into_algo(algo, pt_path)
        else:
            raise

    return algo


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    """
    Bootstrap mean of per-episode values (sample with replacement).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = values[idx].mean()

    return {
        "mean": float(boots.mean()),
        "std": float(boots.std(ddof=1)),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5)),
    }


def _load_torch_checkpoint_into_algo(algo, pt_path: Path) -> None:
    """
    Fallback loader for torch checkpoints saved by our scripts (results/models/*.pt).
    These are standard torch.save(dict) files.

    We load known submodules with strict=False to tolerate minor key mismatches across d3rlpy versions.

    # - DiscreteBC often has: {"imitator": state_dict, "optim": ...}
    # - CQL/BCQ often have: {"q_funcs": OrderedDict, "targ_q_funcs": OrderedDict, "policy": ..., "optim": ...}
    """
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing torch checkpoint: {pt_path}")

    chkpt = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(chkpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(chkpt)}")

    impl = getattr(algo, "_impl", None)
    if impl is None:
        raise RuntimeError("Algorithm impl is not initialized; call build_with_dataset first.")

    # common keys seen in d3rlpy checkpoints
    # - BC often has: {"policy": state_dict, "optim": ...}
    # - CQL/BCQ often have: {"q_funcs": OrderedDict, "targ_q_funcs": OrderedDict, "policy": ..., "optim": ...}
    loaded_any = False

    # helper: load state dict into a module if present
    def try_load(module, state, name: str) -> None:
        nonlocal loaded_any
        if module is None or state is None:
            return
        if hasattr(module, "load_state_dict"):
            missing, unexpected = module.load_state_dict(state, strict=False)
            if unexpected:
                print(f"[WARN] {name}: unexpected keys (ignored): {list(unexpected)[:10]}")
            if missing:
                print(f"[WARN] {name}: missing keys: {list(missing)[:10]}")
            loaded_any = True

    def try_load_any_module_with_state(state, label: str) -> None:
        """Best-effort loader: try to load `state` into some torch.nn.Module inside impl.

        Some d3rlpy impls expose `.policy` as a property that raises NotImplementedError.
        We must never touch it during attribute scanning.
        """
        nonlocal loaded_any
        if state is None:
            return

        def safe_getattr(obj, attr: str):
            try:
                return getattr(obj, attr)
            except NotImplementedError:
                return None
            except Exception:
                return None

        # if impl itself is a torch module, try loading directly
        try:
            import torch.nn as nn  # local import
            if isinstance(impl, nn.Module):
                try:
                    missing, unexpected = impl.load_state_dict(state, strict=False)
                    if unexpected:
                        print(f"[WARN] {label}->impl: unexpected keys (ignored): {list(unexpected)[:10]}")
                    if missing:
                        print(f"[WARN] {label}->impl: missing keys: {list(missing)[:10]}")
                    loaded_any = True
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # prefer common internal attribute names first. critically, never include 'policy'.
        preferred_names = [
            "_imitator",
            "imitator",
            "_policy",
            "_actor",
            "actor",
            "_model",
            "model",
            "_net",
            "net",
            "_q_funcs",
            "_targ_q_funcs",
            "q_funcs",
            "targ_q_funcs",
        ]

        for name in preferred_names:
            if name == "policy":
                continue
            mod = safe_getattr(impl, name)
            if mod is not None and hasattr(mod, "load_state_dict"):
                try:
                    missing, unexpected = mod.load_state_dict(state, strict=False)
                    if unexpected:
                        print(f"[WARN] {label}->{name}: unexpected keys (ignored): {list(unexpected)[:10]}")
                    if missing:
                        print(f"[WARN] {label}->{name}: missing keys: {list(missing)[:10]}")
                    loaded_any = True
                    return
                except Exception:
                    pass

        # otherwise brute-force: try every torch.nn.Module-like attribute on impl,
        # while aggressively avoiding policy-like properties.
        skip_names = {
            "policy",
            "device",
        }

        for name in dir(impl):
            if name.startswith("__"):
                continue
            if name in skip_names:
                continue
            # avoid touching any policy-like attribute names
            if "policy" in name.lower():
                continue

            mod = safe_getattr(impl, name)
            if mod is None or not hasattr(mod, "load_state_dict"):
                continue

            try:
                missing, unexpected = mod.load_state_dict(state, strict=False)
                if unexpected:
                    print(f"[WARN] {label}->{name}: unexpected keys (ignored): {list(unexpected)[:10]}")
                if missing:
                    print(f"[WARN] {label}->{name}: missing keys: {list(missing)[:10]}")
                loaded_any = True
                return
            except Exception:
                continue

    # 1) policy network (if any)
    policy_sd = chkpt.get("policy", None)
    try_load(getattr(impl, "_policy", None), policy_sd, "policy")

    # 2) q-functions for q-learning algos
    q_sd = chkpt.get("q_funcs", None)
    targ_q_sd = chkpt.get("targ_q_funcs", None)

    # in some versions, these are ModuleList; in others, a single module
    q_mod = getattr(impl, "_q_funcs", None)
    targ_q_mod = getattr(impl, "_targ_q_funcs", None)
    try_load(q_mod, q_sd, "q_funcs")
    try_load(targ_q_mod, targ_q_sd, "targ_q_funcs")

    # 3) Imitator network (BC / BCQ) stored under key "imitator" in our exported .pt files
    imitator_sd = chkpt.get("imitator", None)
    if imitator_sd is not None:
        # first try the most likely internal names
        try_load(getattr(impl, "_imitator", None), imitator_sd, "imitator")
        try_load(getattr(impl, "_policy", None), imitator_sd, "policy(from_imitator)")
        # if still nothing loaded, try any module on impl
        if not loaded_any:
            try_load_any_module_with_state(imitator_sd, "imitator")

    # if nothing was loaded, give clearer diagnostic
    if not loaded_any:
        raise RuntimeError(
            f"Could not load any known keys from {pt_path}. Keys={list(chkpt.keys())}. "
            "For BC checkpoints, weights are typically stored under key 'imitator'. "
            "If this persists, the safest fix is to re-run BC in the current environment so the .d3 loads, "
            "or re-export a BC .pt that matches this d3rlpy version's internal module names."
        )


def _find_compatible_pt(policy_name: str, desired_keys: list[str]) -> Path:
    """Search results/models for a torch checkpoint that contains at least one desired key."""
    candidates = sorted(Path("results/models").glob(f"{policy_name}*.pt"))
    # also allow generic matches (in case naming drifted)
    if not candidates:
        candidates = sorted(Path("results/models").glob("*.pt"))

    for p in candidates:
        try:
            chkpt = torch.load(str(p), map_location="cpu")
        except Exception:
            continue
        if isinstance(chkpt, dict) and any(k in chkpt for k in desired_keys):
            return p

    raise FileNotFoundError(
        f"Could not find a compatible .pt for {policy_name} containing any of keys={desired_keys}. "
        f"Searched: {Path('results/models').resolve()}"
    )


def build_eval_algo(policy_name: str, dataset: ReplayBufferBase, device: str):
    """
    Rebuild and load the evaluation policy algorithm (BC / CQL / BCQ) so that
    DiscreteFQE can compute V(s) under that policy.
    """
    # use run directories so we can reconstruct the exact training config via params.json
    policy_run_dirs = {
        "bc_final_ne5_v2": Path("d3rlpy_logs/bc_final_ne5_v2_py310"),
        "cql_final_ne5_v2": Path("d3rlpy_logs/cql_final_ne5_v2"),
        "bcq_best10k_ne5_v2": Path("d3rlpy_logs/bcq_best10k_ne5_v2"),
    }

    run_dir = policy_run_dirs[policy_name]
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing policy run directory: {run_dir}")

    if policy_name.startswith("bc_"):
        return build_algo_from_run_dir(DiscreteBC, DiscreteBCConfig, run_dir, dataset, device, policy_name)
    elif policy_name.startswith("cql_"):
        return build_algo_from_run_dir(DiscreteCQL, DiscreteCQLConfig, run_dir, dataset, device, policy_name)
    elif policy_name.startswith("bcq_"):
        return build_algo_from_run_dir(DiscreteBCQ, DiscreteBCQConfig, run_dir, dataset, device, policy_name)

    raise ValueError(f"Unknown policy_name: {policy_name}")


def build_fqe_from_params(fqe_dir: Path, algo, device: str) -> DiscreteFQE:
    """
    Create DiscreteFQE(algo=..., config=...) in a version-compatible way and
    load the trained FQE weights from fqe_dir/model_*.d3 later.

    In older d3rlpy, DiscreteFQE requires positional args: (algo, config, ...).
    We reconstruct config from fqe_dir/params.json when possible.
    """
    params_path = fqe_dir / "params.json"

    cfg_params: dict = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            root = json.load(f)
        cfg_params = (root.get("config", {}) or {}).get("params", {}) or {}

    # build config object if available; otherwise keep it as None and rely on defaults
    config_obj = None
    if _FQEConfig is not None:
        sig = inspect.signature(_FQEConfig.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in cfg_params.items() if k in allowed}
        config_obj = _FQEConfig(**filtered)

    # instantiate DiscreteFQE with only supported kwargs
    sig = inspect.signature(DiscreteFQE.__init__)
    allowed = list(sig.parameters.keys())
    # Newer: DiscreteFQE(config=..., device=...), Older: DiscreteFQE(algo, config, device=...)
    kwargs = {}
    if "device" in allowed:
        kwargs["device"] = device

    if "algo" in allowed:
        kwargs["algo"] = algo
    if "config" in allowed:
        kwargs["config"] = config_obj if config_obj is not None else (_FQEConfig() if _FQEConfig is not None else None)

    # if signature is positional (common in older versions), call positionally
    if allowed[:2] == ["self", "algo"] or (len(allowed) >= 3 and allowed[1] == "algo" and allowed[2] == "config"):
        cfg_for_pos = config_obj if config_obj is not None else (_FQEConfig() if _FQEConfig is not None else None)
        try:
            return DiscreteFQE(algo, cfg_for_pos, **kwargs)
        except TypeError:
            # Some versions don't accept device kwarg.
            kwargs.pop("device", None)
            return DiscreteFQE(algo, cfg_for_pos, **kwargs)

    # keyword-only style
    # if config is None and required, try to create empty config if possible
    if "config" in allowed and kwargs.get("config", None) is None and _FQEConfig is not None:
        kwargs["config"] = _FQEConfig()

    return DiscreteFQE(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # same dataset object you used elsewhere (ReplayBuffer)
    dataset = load_dataset()

    # episode-level bootstrap = use one init state per stay/episode
    init_obs = get_initial_states_from_replay_buffer(dataset)
    n_eps = init_obs.shape[0]
    print(f"[INFO] Episodes available (bootstrap units): {n_eps}")

    # trained FQE model directories (these should contain model_*.d3 and params.json)
    fqe_runs = {
        "bc_final_ne5_v2": Path("d3rlpy_logs/fqe_bc_final_ne5_v2"),
        "cql_final_ne5_v2": Path("d3rlpy_logs/fqe_cql_final_ne5_v2"),
        "bcq_best10k_ne5_v2": Path("d3rlpy_logs/fqe_bcq_best10k_ne5_v2"),
    }

    rows = []

    for policy_name, fqe_dir in fqe_runs.items():
        print(f"\n[INFO] Bootstrapping FQE CI for: {policy_name}")
        print(f"[INFO] FQE dir: {fqe_dir}")

        model_path = pick_latest_model(fqe_dir)
        print(f"[INFO] Using FQE checkpoint: {model_path.name}")

        #build evaluation policy algo (needed for DiscreteFQE to compute V under pi)
        eval_algo = build_eval_algo(policy_name, dataset, args.device)

        # build FQE object compatibly, then load trained FQE weights
        fqe = build_fqe_from_params(fqe_dir, eval_algo, args.device)
        fqe.build_with_dataset(dataset)
        fqe.load_model(str(model_path))

        # per-episode initial-state values
        v0 = fqe.predict_value(init_obs).reshape(-1).astype(np.float64)

        stats = bootstrap_mean(v0, n_bootstrap=args.n_bootstrap, seed=args.seed)
        rows.append({"policy": policy_name, "n_episodes": n_eps, **stats})

        print(
            f"[RESULT] {policy_name}: mean={stats['mean']:.3f} "
            f"95% CI=({stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}) "
            f"boot_std={stats['std']:.3f}"
        )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fqe_bootstrap_ci.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
