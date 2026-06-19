from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["c1 -> t", "c2 -> t", "t -> c1", "t -> c2", "c1 -> c2", "c2 -> c1"]
FAMILY_MAP = {
    "t -> c": ["t -> c1", "t -> c2"],
    "c -> t": ["c1 -> t", "c2 -> t"],
    "c -> c": ["c1 -> c2", "c2 -> c1"],
}
FAMILY_COLS = ["sum_t_to_c", "sum_c_to_t", "sum_c_to_c"]
FAMILY_LABELS = {"sum_t_to_c": "t -> c", "sum_c_to_t": "c -> t", "sum_c_to_c": "c -> c"}
SETS = range(1, 7)

DEFAULT_SELECTED_PARAMS = {
    "eval1": {
        "criterion": "gini",
        "max_depth": 3,
        "min_samples_split": 15,
        "min_samples_leaf": 19,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "ccp_alpha": 0.0,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "eval2": {
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "ccp_alpha": 0.005,
        "class_weight": "balanced",
        "random_state": 42,
    },
}

# Same search space as journal/eval1_full_feature_importance_sensitivity.py.
# OUTPUT_GRID_DIR=split5_leaf5_30: 2 * 5 * 8 * 26 * 4 * 2 * 3 = 49,920 candidates per eval.
SEARCH_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_split": list(range(5, 41, 5)),
    "min_samples_leaf": list(range(5, 31)),
    "max_leaf_nodes": [None, 4, 6, 8],
    "min_impurity_decrease": [0.0, 0.001],
    "ccp_alpha": [0.0, 0.001, 0.005],
}
GRID_ID = "split5_leaf5_30_grid_49920"
PARAM_COLUMNS = [
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "ccp_alpha",
    "class_weight",
    "random_state",
]
SELECTED_CORE_COLUMNS = ["criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes"]

_WORKER_EVAL_NAME: str | None = None
_WORKER_FRAMES: dict[int, pd.DataFrame] | None = None
_WORKER_SEARCH_EVALUATION: str = "loocv"
_WORKER_N_SPLITS: int = 5
_WORKER_SELECTED_PARAMS: dict[str, dict[str, Any]] | None = None


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "demo.ipynb").exists() and (path / "datasets" / "decisiontree" / "normalized").exists():
            return path
    raise FileNotFoundError("Could not locate repo root containing demo.ipynb and datasets/decisiontree/normalized")


def output_dir(root: Path) -> Path:
    return root / "decision_tree_hparam_search_out"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def eval_data_dirs(root: Path) -> dict[str, Path]:
    return {
        "eval1": root / "datasets" / "decisiontree" / "normalized",
        "eval2": root / "prediction" / "datasets" / "decisiontree" / "normalized",
    }


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    out.setdefault("criterion", "gini")
    out.setdefault("max_leaf_nodes", None)
    out.setdefault("min_impurity_decrease", 0.0)
    out.setdefault("ccp_alpha", 0.0)
    out.setdefault("class_weight", "balanced")
    out.setdefault("random_state", 42)
    out.setdefault("splitter", "best")
    if pd.isna(out.get("max_leaf_nodes")):
        out["max_leaf_nodes"] = None
    for key in ("max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes", "random_state"):
        if out.get(key) is not None and not isinstance(out.get(key), str):
            out[key] = int(out[key])
    for key in ("min_impurity_decrease", "ccp_alpha"):
        if out.get(key) is not None:
            out[key] = float(out[key])
    return out


def param_key(params: dict[str, Any]) -> str:
    params = normalize_params(params)
    return json.dumps({k: params.get(k) for k in PARAM_COLUMNS}, sort_keys=True)


def resolve_selected_params(selected_params: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    params = {key: dict(value) for key, value in DEFAULT_SELECTED_PARAMS.items()}
    if selected_params:
        for eval_name, eval_params in selected_params.items():
            params[eval_name] = {**params.get(eval_name, {}), **eval_params}
    return params


def selected_core_mask(
    df: pd.DataFrame,
    eval_name: str,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> pd.Series:
    params = normalize_params(resolve_selected_params(selected_params)[eval_name])
    mask = pd.Series(True, index=df.index)
    for key in SELECTED_CORE_COLUMNS:
        if params.get(key) is None:
            mask &= df[key].isna()
        else:
            mask &= df[key] == params[key]
    return mask


def build_candidates(grid: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    grid = SEARCH_GRID if grid is None else grid
    keys = list(grid)
    candidates = []
    seen = set()
    for values in product(*(grid[k] for k in keys)):
        params = normalize_params(dict(zip(keys, values)))
        key = param_key(params)
        if key not in seen:
            candidates.append(params)
            seen.add(key)
    return candidates


def expected_candidate_count(grid: dict[str, list[Any]] | None = None) -> int:
    grid = SEARCH_GRID if grid is None else grid
    count = 1
    for values in grid.values():
        count *= len(values)
    return count


def load_eval_frames(eval_name: str, root: Path) -> dict[int, pd.DataFrame]:
    base = eval_data_dirs(root)[eval_name]
    frames = {}
    for set_id in SETS:
        path = base / f"permute_{set_id - 1}.csv"
        df = pd.read_csv(path)
        missing = [col for col in ["label", *FEATURES] if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        df = df.copy()
        df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frames[set_id] = df
    return frames


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "rev_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rev_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rev_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "nonrev_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def stratified_oof_metrics(
    frame: pd.DataFrame,
    params: dict[str, Any],
    seed: int = 42,
    labels: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict[str, float]:
    x = frame[FEATURES].to_numpy()
    y = frame["label"].astype(int).to_numpy() if labels is None else np.asarray(labels, dtype=int)
    pred = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y):
        model = DecisionTreeClassifier(**normalize_params(params))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return metric_row(y, pred)


def loocv_metrics(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
    x = frame[FEATURES].to_numpy()
    y = frame["label"].astype(int).to_numpy()
    pred = np.zeros(len(y), dtype=int)
    for train, test in LeaveOneOut().split(x):
        model = DecisionTreeClassifier(**normalize_params(params))
        model.fit(x[train], y[train])
        pred[test[0]] = model.predict(x[test])[0]
    return metric_row(y, pred)


def candidate_metrics(
    frame: pd.DataFrame,
    params: dict[str, Any],
    mode: str = "loocv",
    seed: int = 42,
    n_splits: int = 5,
) -> dict[str, float]:
    if mode == "loocv":
        return loocv_metrics(frame, params)
    if mode == "stratified":
        return stratified_oof_metrics(frame, params, seed=seed, n_splits=n_splits)
    raise ValueError(f"Unknown search evaluation mode: {mode}")


def fit_full_tree(frame: pd.DataFrame, params: dict[str, Any]):
    x = frame[FEATURES]
    y = frame["label"].astype(int)
    model = DecisionTreeClassifier(**normalize_params(params)).fit(x, y)
    return model, x, y


def family_importance_from_model(model: DecisionTreeClassifier) -> dict[str, float]:
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    return {family: float(importances[cols].sum()) for family, cols in FAMILY_MAP.items()}


def evaluate_candidate(
    eval_name: str,
    frames: dict[int, pd.DataFrame],
    params: dict[str, Any],
    search_evaluation: str = "loocv",
    n_splits: int = 5,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = normalize_params(params)
    per_set = {}
    for set_id, frame in frames.items():
        per_set[set_id] = candidate_metrics(
            frame,
            params,
            mode=search_evaluation,
            seed=42 + set_id,
            n_splits=n_splits,
        )

    best_set = max(SETS, key=lambda s: (per_set[s]["accuracy"], -s))
    model, x, y = fit_full_tree(frames[best_set], params)
    train_metrics = metric_row(y, model.predict(x))
    family = family_importance_from_model(model)

    row = {
        "eval_name": eval_name,
        "grid_id": GRID_ID,
        **{key: params.get(key) for key in PARAM_COLUMNS},
        "best_set": best_set,
        "best_accuracy": per_set[best_set]["accuracy"],
        "best_macro_f1": per_set[best_set]["macro_f1"],
        "best_rev_f1": per_set[best_set]["rev_f1"],
        "accuracy_mean": float(np.mean([per_set[s]["accuracy"] for s in SETS])),
        "accuracy_worst": float(np.min([per_set[s]["accuracy"] for s in SETS])),
        "rev_f1_mean": float(np.mean([per_set[s]["rev_f1"] for s in SETS])),
        "rev_f1_worst": float(np.min([per_set[s]["rev_f1"] for s in SETS])),
        "train_accuracy_best_set": train_metrics["accuracy"],
        "train_loocv_accuracy_gap_best_set": train_metrics["accuracy"] - per_set[best_set]["accuracy"],
        "tree_depth_best_set": model.get_depth(),
        "n_leaves_best_set": model.get_n_leaves(),
        "top_feature": FEATURES[int(np.argmax(model.feature_importances_))],
        "sum_t_to_c": family["t -> c"],
        "sum_c_to_t": family["c -> t"],
        "sum_c_to_c": family["c -> c"],
    }
    for set_id in SETS:
        for metric, value in per_set[set_id].items():
            row[f"{metric}_set{set_id}"] = value
    for feature, value in zip(FEATURES, model.feature_importances_):
        row[f"fi_{feature.replace(' -> ', '_to_')}"] = value
    row["is_selected_exact_params"] = param_key(params) == param_key(resolve_selected_params(selected_params)[eval_name])
    return row


def _init_search_worker(
    eval_name: str,
    frames: dict[int, pd.DataFrame],
    search_evaluation: str,
    n_splits: int,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> None:
    global _WORKER_EVAL_NAME, _WORKER_FRAMES, _WORKER_SEARCH_EVALUATION, _WORKER_N_SPLITS, _WORKER_SELECTED_PARAMS
    _WORKER_EVAL_NAME = eval_name
    _WORKER_FRAMES = frames
    _WORKER_SEARCH_EVALUATION = search_evaluation
    _WORKER_N_SPLITS = n_splits
    _WORKER_SELECTED_PARAMS = selected_params


def _evaluate_candidate_worker(params: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVAL_NAME is None or _WORKER_FRAMES is None:
        raise RuntimeError("Search worker was not initialized")
    return evaluate_candidate(
        _WORKER_EVAL_NAME,
        _WORKER_FRAMES,
        params,
        search_evaluation=_WORKER_SEARCH_EVALUATION,
        n_splits=_WORKER_N_SPLITS,
        selected_params=_WORKER_SELECTED_PARAMS,
    )


def load_compatible_search_cache(path: Path, expected_count: int) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    if "grid_id" not in cached.columns or set(cached["grid_id"].dropna().unique()) != {GRID_ID}:
        return None
    if len(cached) != expected_count:
        return None
    return cached


def run_search(
    eval_name: str,
    root: Path,
    out_dir: Path,
    force: bool = False,
    progress_every: int = 250,
    search_evaluation: str = "loocv",
    n_splits: int = 5,
    workers: int = 1,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    candidates = build_candidates()
    expected = expected_candidate_count()
    if len(candidates) != expected:
        raise AssertionError((len(candidates), expected))

    path = out_dir / f"{eval_name}_hyperparameter_search.csv"
    if not force:
        cached = load_compatible_search_cache(path, expected)
        if cached is not None:
            print(f"{eval_name}: using compatible cache with {len(cached)} candidates: {path.relative_to(root)}")
            return cached

    frames = load_eval_frames(eval_name, root)
    print(f"{eval_name}: evaluating {len(candidates)} hyperparameter candidates")
    rows = []
    partial_path = path.with_suffix(".partial.csv")
    if workers <= 1:
        for index, params in enumerate(candidates, start=1):
            rows.append(
                evaluate_candidate(
                    eval_name,
                    frames,
                    params,
                    search_evaluation=search_evaluation,
                    n_splits=n_splits,
                    selected_params=selected_params,
                )
            )
            if index % progress_every == 0 or index == len(candidates):
                write_csv(pd.DataFrame(rows), partial_path)
                print(f"  {index}/{len(candidates)}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_search_worker,
            initargs=(eval_name, frames, search_evaluation, n_splits, selected_params),
        ) as executor:
            futures = [executor.submit(_evaluate_candidate_worker, params) for params in candidates]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % progress_every == 0 or index == len(candidates):
                    write_csv(pd.DataFrame(rows), partial_path)
                    print(f"  {index}/{len(candidates)}", flush=True)

    result = pd.DataFrame(rows)
    result["is_selected_core_params"] = selected_core_mask(result, eval_name, selected_params)
    write_csv(result, path)
    partial_path.unlink(missing_ok=True)
    print(f"saved: {path.relative_to(root)}")
    return result


def selected_family_position(
    search_df: pd.DataFrame,
    eval_name: str,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    df = search_df.copy()
    df["is_selected_core_params"] = selected_core_mask(df, eval_name, selected_params)
    selected = df[df["is_selected_core_params"]]
    if selected.empty:
        raise ValueError(f"Selected core params were not included in search for {eval_name}")

    dist = df[FAMILY_COLS].agg(["mean", "std", "median", "min", "max"]).T
    dist["selected"] = selected[FAMILY_COLS].mean()
    dist["selected_minus_mean"] = dist["selected"] - dist["mean"]
    dist["selected_z"] = dist["selected_minus_mean"] / dist["std"].replace(0, np.nan)
    dist = dist.rename(index=FAMILY_LABELS)
    dist.insert(0, "eval_name", eval_name)
    return dist.reset_index(names="family")


def selected_family_by_set(
    eval_name: str,
    root: Path,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    frames = load_eval_frames(eval_name, root)
    params = resolve_selected_params(selected_params)[eval_name]
    rows = []
    for set_id, frame in frames.items():
        model, _, _ = fit_full_tree(frame, params)
        rows.append({"eval_name": eval_name, "set": set_id, **family_importance_from_model(model)})
    return pd.DataFrame(rows)


def save_family_outputs(
    eval_name: str,
    root: Path,
    out_dir: Path,
    search_df: pd.DataFrame,
    selected_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    position = selected_family_position(search_df, eval_name, selected_params)
    by_set = selected_family_by_set(eval_name, root, selected_params)
    write_csv(position, out_dir / f"{eval_name}_feature_family_distribution.csv")
    write_csv(by_set, out_dir / f"{eval_name}_selected_feature_family_by_set.csv")
    return {"feature_family_distribution": position, "selected_feature_family_by_set": by_set}


def local_sensitivity_candidates(params: dict[str, Any]) -> list[dict[str, Any]]:
    params = normalize_params(params)
    rows = []
    seen = set()

    def add(candidate: dict[str, Any]) -> None:
        candidate = normalize_params(candidate)
        key = param_key(candidate)
        if key not in seen:
            rows.append(candidate)
            seen.add(key)

    add(params)
    for depth in sorted({max(1, params["max_depth"] - 1), params["max_depth"], params["max_depth"] + 1}):
        add(dict(params, max_depth=depth))
    for split in sorted(
        {
            max(2, params["min_samples_split"] - 10),
            max(2, params["min_samples_split"] - 5),
            params["min_samples_split"],
            params["min_samples_split"] + 5,
            params["min_samples_split"] + 10,
        }
    ):
        add(dict(params, min_samples_split=split))
    for leaf in sorted(
        {
            max(1, params["min_samples_leaf"] - 10),
            max(1, params["min_samples_leaf"] - 5),
            params["min_samples_leaf"],
            params["min_samples_leaf"] + 5,
            params["min_samples_leaf"] + 10,
        }
    ):
        add(dict(params, min_samples_leaf=leaf))
    for nodes in [None, 4, 6, 8]:
        add(dict(params, max_leaf_nodes=nodes))
    return rows


def run_selected_robustness(
    eval_name: str,
    root: Path,
    out_dir: Path,
    selected_params: dict[str, dict[str, Any]] | None = None,
    n_repeats: int = 20,
    n_permutations: int = 100,
    n_splits: int = 5,
) -> dict[str, pd.DataFrame]:
    frames = load_eval_frames(eval_name, root)
    params = resolve_selected_params(selected_params)[eval_name]

    per_set_rows = []
    for set_id, frame in frames.items():
        model, x, y = fit_full_tree(frame, params)
        train = metric_row(y, model.predict(x))
        loo = loocv_metrics(frame, params)
        leaves = pd.Series(model.apply(x)).value_counts()
        per_set_rows.append(
            {
                "eval_name": eval_name,
                "set": set_id,
                "tree_depth": model.get_depth(),
                "n_leaves": model.get_n_leaves(),
                "min_leaf_size": int(leaves.min()),
                "train_accuracy": train["accuracy"],
                "loocv_accuracy": loo["accuracy"],
                "train_loocv_accuracy_gap": train["accuracy"] - loo["accuracy"],
                "loocv_macro_f1": loo["macro_f1"],
                "loocv_rev_f1": loo["rev_f1"],
            }
        )
    per_set = pd.DataFrame(per_set_rows)

    cv_rows = []
    for repeat in range(n_repeats):
        per_repeat = pd.DataFrame(
            [
                stratified_oof_metrics(frame, params, seed=10_000 + repeat + set_id, n_splits=n_splits)
                for set_id, frame in frames.items()
            ]
        )
        cv_rows.append(
            {
                "eval_name": eval_name,
                "repeat": repeat,
                "accuracy_mean": per_repeat["accuracy"].mean(),
                "accuracy_worst": per_repeat["accuracy"].min(),
                "macro_f1_mean": per_repeat["macro_f1"].mean(),
                "rev_f1_mean": per_repeat["rev_f1"].mean(),
                "rev_f1_worst": per_repeat["rev_f1"].min(),
            }
        )
    cv_dist = pd.DataFrame(cv_rows)

    rng = np.random.RandomState(2026)
    observed = pd.DataFrame([stratified_oof_metrics(frame, params, seed=0) for frame in frames.values()]).mean(
        numeric_only=True
    )
    null_rows = []
    for permutation in range(n_permutations):
        per_perm = []
        for frame in frames.values():
            shuffled = rng.permutation(frame["label"].astype(int).to_numpy())
            per_perm.append(
                stratified_oof_metrics(
                    frame,
                    params,
                    seed=20_000 + permutation,
                    labels=shuffled,
                    n_splits=n_splits,
                )
            )
        row = pd.DataFrame(per_perm).mean(numeric_only=True).to_dict()
        null_rows.append({"eval_name": eval_name, "permutation": permutation, **row})
    null_dist = pd.DataFrame(null_rows)

    local_rows = []
    for candidate in local_sensitivity_candidates(params):
        row = evaluate_candidate(
            eval_name,
            frames,
            candidate,
            search_evaluation="loocv",
            n_splits=n_splits,
            selected_params=selected_params,
        )
        row["is_selected_exact_params"] = param_key(candidate) == param_key(params)
        local_rows.append(row)
    local = pd.DataFrame(local_rows)
    local["is_selected_core_params"] = selected_core_mask(local, eval_name, selected_params)

    p_values = {
        f"permutation_p_{metric}": (1 + (null_dist[metric] >= observed[metric]).sum()) / (n_permutations + 1)
        for metric in ["accuracy", "macro_f1", "rev_f1"]
    }
    selected_local = local[local["is_selected_exact_params"]].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "eval_name": eval_name,
                "selected_params": json.dumps(normalize_params(params), sort_keys=True),
                "loocv_accuracy_mean": per_set["loocv_accuracy"].mean(),
                "loocv_accuracy_worst": per_set["loocv_accuracy"].min(),
                "loocv_rev_f1_mean": per_set["loocv_rev_f1"].mean(),
                "loocv_rev_f1_worst": per_set["loocv_rev_f1"].min(),
                "train_loocv_accuracy_gap_mean": per_set["train_loocv_accuracy_gap"].mean(),
                "train_loocv_accuracy_gap_max": per_set["train_loocv_accuracy_gap"].max(),
                "repeated_cv_accuracy_mean": cv_dist["accuracy_mean"].mean(),
                "repeated_cv_accuracy_std": cv_dist["accuracy_mean"].std(ddof=1),
                "repeated_cv_rev_f1_mean": cv_dist["rev_f1_mean"].mean(),
                "repeated_cv_rev_f1_std": cv_dist["rev_f1_mean"].std(ddof=1),
                "null_accuracy_mean": null_dist["accuracy"].mean(),
                "null_rev_f1_mean": null_dist["rev_f1"].mean(),
                "local_accuracy_mean_min": local["accuracy_mean"].min(),
                "local_accuracy_mean_max": local["accuracy_mean"].max(),
                "selected_local_accuracy_mean": selected_local["accuracy_mean"],
                "selected_local_rev_f1_mean": selected_local["rev_f1_mean"],
                **p_values,
            }
        ]
    )

    outputs = {
        "per_set": per_set,
        "cv_dist": cv_dist,
        "null_dist": null_dist,
        "local_sensitivity": local,
        "summary": summary,
    }
    for name, frame in outputs.items():
        write_csv(frame, out_dir / f"{eval_name}_robustness_{name}.csv")
    return outputs


def run_eval(
    eval_name: str,
    root: Path | None = None,
    out_dir: Path | None = None,
    selected_params: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    skip_robustness: bool = False,
    progress_every: int = 250,
    search_evaluation: str = "loocv",
    n_splits: int = 5,
    workers: int = 1,
    robustness_repeats: int = 20,
    robustness_permutations: int = 100,
) -> dict[str, Any]:
    root = find_repo_root() if root is None else Path(root)
    out_dir = output_dir(root) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search = run_search(
        eval_name,
        root=root,
        out_dir=out_dir,
        force=force,
        progress_every=progress_every,
        search_evaluation=search_evaluation,
        n_splits=n_splits,
        workers=workers,
        selected_params=selected_params,
    )
    family = save_family_outputs(eval_name, root, out_dir, search, selected_params)
    robustness = None
    if not skip_robustness:
        robustness = run_selected_robustness(
            eval_name,
            root=root,
            out_dir=out_dir,
            selected_params=selected_params,
            n_repeats=robustness_repeats,
            n_permutations=robustness_permutations,
            n_splits=n_splits,
        )
    return {"search": search, "family": family, "robustness": robustness}


def run_all(
    root: Path | None = None,
    out_dir: Path | None = None,
    selected_params: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    skip_robustness: bool = False,
    progress_every: int = 250,
    search_evaluation: str = "loocv",
    n_splits: int = 5,
    workers: int = 1,
    robustness_repeats: int = 20,
    robustness_permutations: int = 100,
) -> dict[str, dict[str, Any]]:
    root = find_repo_root() if root is None else Path(root)
    out_dir = output_dir(root) if out_dir is None else Path(out_dir)
    results = {}
    for eval_name in ("eval1", "eval2"):
        results[eval_name] = run_eval(
            eval_name,
            root=root,
            out_dir=out_dir,
            selected_params=selected_params,
            force=force,
            skip_robustness=skip_robustness,
            progress_every=progress_every,
            search_evaluation=search_evaluation,
            n_splits=n_splits,
            workers=workers,
            robustness_repeats=robustness_repeats,
            robustness_permutations=robustness_permutations,
        )

    family_all = pd.concat(
        [results[name]["family"]["feature_family_distribution"] for name in ("eval1", "eval2")],
        ignore_index=True,
    )
    selected_by_set_all = pd.concat(
        [results[name]["family"]["selected_feature_family_by_set"] for name in ("eval1", "eval2")],
        ignore_index=True,
    )
    write_csv(family_all, out_dir / "eval1_eval2_feature_family_distribution.csv")
    write_csv(selected_by_set_all, out_dir / "eval1_eval2_selected_feature_family_by_set.csv")
    if not skip_robustness:
        robustness_all = pd.concat(
            [results[name]["robustness"]["summary"] for name in ("eval1", "eval2")],
            ignore_index=True,
        )
        write_csv(robustness_all, out_dir / "eval1_eval2_robustness_summary.csv")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision-tree hyperparameter search for MIAO eval1/eval2 datasets.")
    parser.add_argument("--eval", choices=["eval1", "eval2", "all"], default="all")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore compatible cache and recompute search CSVs.")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--search-evaluation", choices=["loocv", "stratified"], default="loocv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--robustness-repeats", type=int, default=20)
    parser.add_argument("--robustness-permutations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repo_root(args.root)
    out_dir = output_dir(root) if args.out_dir is None else args.out_dir
    print("candidate count:", expected_candidate_count())
    if args.eval == "all":
        run_all(
            root=root,
            out_dir=out_dir,
            force=args.force,
            skip_robustness=args.skip_robustness,
            progress_every=args.progress_every,
            search_evaluation=args.search_evaluation,
            n_splits=args.n_splits,
            workers=args.workers,
            robustness_repeats=args.robustness_repeats,
            robustness_permutations=args.robustness_permutations,
        )
    else:
        run_eval(
            args.eval,
            root=root,
            out_dir=out_dir,
            force=args.force,
            skip_robustness=args.skip_robustness,
            progress_every=args.progress_every,
            search_evaluation=args.search_evaluation,
            n_splits=args.n_splits,
            workers=args.workers,
            robustness_repeats=args.robustness_repeats,
            robustness_permutations=args.robustness_permutations,
        )
    print("output:", out_dir)


if __name__ == "__main__":
    main()
