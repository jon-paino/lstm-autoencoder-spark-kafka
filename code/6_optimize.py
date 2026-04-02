"""Step 6: Optimization - Hyperparameter grid search and data split optimization.

Two modes:
    python code/6_optimize.py --mode split
    python code/6_optimize.py --mode hyperparams
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.scorer import AnomalyScorer, ScorerConfig
from src.preprocess import PreprocessorConfig, preprocess_pipeline, get_test_week_info

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "nyc_taxi.csv")


# ---------------------------------------------------------------------------
# Configs & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Mirror of code/3_train_model.py TrainingConfig."""
    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 10
    min_delta: float = 1e-6
    grad_clip: float = 1.0


@dataclass
class SplitConfig:
    """Configuration for a single split experiment."""
    train_weeks: int
    val_weeks: int
    threshold_weeks: int

    @property
    def total_dev_weeks(self) -> int:
        return self.train_weeks + self.val_weeks + self.threshold_weeks


@dataclass
class HyperparamConfig:
    """Configuration for a hyperparameter experiment."""
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    threshold_percentile: float
    batch_size: int = 4

    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "threshold_percentile": self.threshold_percentile,
            "batch_size": self.batch_size,
        }


@dataclass
class ExperimentResult:
    """Results from a single experiment (works for both modes)."""
    config_dict: Dict
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f_beta_score: float
    train_loss: float
    val_loss: float
    threshold: float
    best_epoch: int = 0
    total_params: int = 0

    def to_dict(self) -> Dict:
        return {
            **self.config_dict,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "f_beta_score": self.f_beta_score,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "threshold": self.threshold,
            "best_epoch": self.best_epoch,
            "total_params": self.total_params,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_f_beta(precision: float, recall: float, beta: float = 0.1) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray, beta: float = 0.1) -> Dict[str, float]:
    tp = np.sum(predictions & actuals)
    fp = np.sum(predictions & ~actuals)
    fn = np.sum(~predictions & actuals)
    tn = np.sum(~predictions & ~actuals)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f_beta = compute_f_beta(precision, recall, beta)
    accuracy = (tp + tn) / len(predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f_beta_score": f_beta,
    }


# ---------------------------------------------------------------------------
# Inline training loop (self-contained, avoids numbered-file imports)
# ---------------------------------------------------------------------------

class _EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _train_model(model, train_loader, val_loader, device, config: TrainingConfig):
    """Minimal training loop. Returns (model, history)."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    es = _EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_state = None
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(config.epochs):
        model.train()
        t_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            t_loss += loss.item()
            n += 1
        t_loss /= n

        model.eval()
        v_loss, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                v_loss += criterion(model(x), x).item()
                nv += 1
        v_loss /= nv

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1

        if es.step(v_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, history


# ---------------------------------------------------------------------------
# Core experiment runner (shared by both modes)
# ---------------------------------------------------------------------------

def train_and_evaluate(
    data_path: str,
    device: torch.device,
    train_weeks: int,
    val_weeks: int,
    threshold_weeks: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    batch_size: int = 4,
    epochs: int = 100,
    patience: int = 10,
    beta: float = 0.1,
) -> Optional[ExperimentResult]:
    """Preprocess, train, score, evaluate. Returns ExperimentResult or None."""
    try:
        preprocess_config = PreprocessorConfig(
            train_weeks=train_weeks,
            val_weeks=val_weeks,
            threshold_weeks=threshold_weeks,
        )
        dataloaders, _, scaler, week_info, split_indices = preprocess_pipeline(
            data_path, config=preprocess_config, batch_size=batch_size,
        )

        for split in ("train", "val", "threshold_val", "test"):
            if dataloaders[split] is None or len(dataloaders[split].dataset) == 0:
                logger.warning("Empty %s set, skipping", split)
                return None

        model = create_model(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())

        tc = TrainingConfig(epochs=epochs, learning_rate=learning_rate, patience=patience)
        model, history = _train_model(model, dataloaders["train"], dataloaders["val"], device, tc)

        scorer = AnomalyScorer(config=ScorerConfig(threshold_percentile=threshold_percentile))
        scorer.fit(model, dataloaders["val"], device)
        val_scores, _ = scorer.compute_scores(model, dataloaders["threshold_val"], device)
        scorer.set_threshold(val_scores)

        test_scores, _ = scorer.compute_scores(model, dataloaders["test"], device)
        predictions = scorer.predict(test_scores)

        test_info = get_test_week_info(week_info, split_indices)
        actuals = np.array([w["is_anomaly"] for w in test_info])
        metrics = compute_metrics(predictions, actuals, beta)

        config_dict = {
            "train_weeks": train_weeks,
            "val_weeks": val_weeks,
            "threshold_weeks": threshold_weeks,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "threshold_percentile": threshold_percentile,
        }

        return ExperimentResult(
            config_dict=config_dict,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            f_beta_score=metrics["f_beta_score"],
            train_loss=history["train_loss"][-1],
            val_loss=history["val_loss"][-1] if history["val_loss"] else float("inf"),
            threshold=scorer.threshold,
            best_epoch=history.get("best_epoch", 0),
            total_params=total_params,
        )

    except Exception as e:
        logger.error("Experiment failed: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Split optimization
# ---------------------------------------------------------------------------

def generate_split_configs(
    total_dev_weeks: int, min_train: int = 4, min_val: int = 2, min_threshold: int = 1,
) -> List[SplitConfig]:
    configs = []
    for tw in range(min_train, total_dev_weeks - min_val - min_threshold + 1):
        remaining = total_dev_weeks - tw
        for vw in range(min_val, remaining - min_threshold + 1):
            thw = remaining - vw
            if thw >= min_threshold:
                configs.append(SplitConfig(train_weeks=tw, val_weeks=vw, threshold_weeks=thw))
    return configs


def run_split_optimization(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = generate_split_configs(
        args.total_dev_weeks, args.min_train, args.min_val, args.min_threshold,
    )
    print(f"\nDevice: {device}")
    print(f"Testing {len(configs)} split configurations "
          f"(dev_weeks={args.total_dev_weeks}, min_train={args.min_train}, "
          f"min_val={args.min_val}, min_thresh={args.min_threshold})")

    results: List[ExperimentResult] = []
    for i, cfg in enumerate(configs):
        set_seed(42)
        print(f"\n[{i+1}/{len(configs)}] train={cfg.train_weeks}, "
              f"val={cfg.val_weeks}, threshold={cfg.threshold_weeks}...", end="", flush=True)
        r = train_and_evaluate(
            data_path=args.data_path, device=device,
            train_weeks=cfg.train_weeks, val_weeks=cfg.val_weeks,
            threshold_weeks=cfg.threshold_weeks,
            hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
            beta=args.beta,
        )
        if r is not None:
            results.append(r)
            print(f"  F1={r.f1_score:.2%}  Prec={r.precision:.2%}  Rec={r.recall:.2%}")

    _print_results(results, sort_by="f1_score", title="SPLIT OPTIMIZATION RESULTS")
    if args.output:
        _save_results(results, args.output, len(configs), "f1_score", args.beta)


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------

def _focused_configs() -> List[HyperparamConfig]:
    return [
        HyperparamConfig(40, 1, 0.0, 1e-3, 95.0),
        HyperparamConfig(40, 1, 0.1, 1e-3, 95.0),
        HyperparamConfig(40, 1, 0.2, 1e-3, 95.0),
        HyperparamConfig(40, 2, 0.1, 1e-3, 95.0),
        HyperparamConfig(64, 1, 0.0, 1e-3, 95.0),
        HyperparamConfig(64, 1, 0.1, 1e-3, 95.0),
        HyperparamConfig(64, 1, 0.2, 1e-3, 95.0),
        HyperparamConfig(64, 2, 0.1, 1e-3, 95.0),
        HyperparamConfig(64, 2, 0.2, 1e-3, 95.0),
        HyperparamConfig(32, 1, 0.1, 1e-3, 95.0),
        HyperparamConfig(32, 2, 0.1, 1e-3, 95.0),
        HyperparamConfig(128, 1, 0.2, 1e-3, 95.0),
        HyperparamConfig(128, 1, 0.3, 1e-3, 95.0),
        HyperparamConfig(64, 1, 0.2, 5e-4, 95.0),
        HyperparamConfig(64, 1, 0.2, 2e-3, 95.0),
        HyperparamConfig(64, 1, 0.2, 1e-3, 90.0),
        HyperparamConfig(64, 1, 0.2, 1e-3, 99.0),
        HyperparamConfig(40, 1, 0.1, 5e-4, 90.0),
        HyperparamConfig(40, 1, 0.2, 1e-3, 90.0),
        HyperparamConfig(64, 1, 0.1, 5e-4, 90.0),
    ]


def _generate_grid_configs(
    hidden_dims=(32, 40, 64, 128), num_layers_list=(1, 2),
    dropouts=(0.0, 0.1, 0.2, 0.3), learning_rates=(1e-4, 5e-4, 1e-3),
    threshold_percentiles=(90.0, 95.0, 99.0),
) -> List[HyperparamConfig]:
    return [
        HyperparamConfig(h, l, d, lr, tp)
        for h, l, d, lr, tp in product(
            hidden_dims, num_layers_list, dropouts, learning_rates, threshold_percentiles
        )
    ]


def _generate_random_configs(n: int = 50) -> List[HyperparamConfig]:
    cfgs = []
    for _ in range(n):
        cfgs.append(HyperparamConfig(
            hidden_dim=random.randint(16, 128),
            num_layers=random.randint(1, 3),
            dropout=round(random.uniform(0.0, 0.4), 2),
            learning_rate=10 ** random.uniform(-4, -2),
            threshold_percentile=round(random.uniform(85.0, 99.0), 1),
        ))
    return cfgs


def run_hyperparams_optimization(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.search == "grid":
        configs = _generate_grid_configs()
    elif args.search == "random":
        configs = _generate_random_configs(args.n_configs)
    else:
        configs = _focused_configs()

    print(f"\nDevice: {device}")
    print(f"Split: train={args.train_weeks}, val={args.val_weeks}, threshold={args.threshold_weeks}")
    print(f"Search: {args.search} ({len(configs)} configurations)")

    results: List[ExperimentResult] = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        set_seed(42)
        elapsed = time.time() - t0
        print(f"\n[{i+1}/{len(configs)}] ({elapsed:.0f}s) hidden={cfg.hidden_dim} "
              f"layers={cfg.num_layers} drop={cfg.dropout} lr={cfg.learning_rate:.1e} "
              f"thresh={cfg.threshold_percentile}%...", end="", flush=True)

        r = train_and_evaluate(
            data_path=args.data_path, device=device,
            train_weeks=args.train_weeks, val_weeks=args.val_weeks,
            threshold_weeks=args.threshold_weeks,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            dropout=cfg.dropout, learning_rate=cfg.learning_rate,
            threshold_percentile=cfg.threshold_percentile,
            batch_size=cfg.batch_size, epochs=args.epochs,
            patience=args.patience, beta=args.beta,
        )
        if r is not None:
            results.append(r)
            print(f"  F1={r.f1_score:.2%}  Fb={r.f_beta_score:.2%}  "
                  f"P={r.precision:.2%}  R={r.recall:.2%}")
            if args.output:
                _save_results(results, args.output, len(configs), args.sort_by, args.beta)

    _print_results(results, sort_by=args.sort_by, title="HYPERPARAMETER OPTIMIZATION RESULTS")
    if args.output:
        _save_results(results, args.output, len(configs), args.sort_by, args.beta)
        print(f"\nResults saved to {args.output}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(results: List[ExperimentResult], sort_by: str = "f1_score",
                   title: str = "RESULTS") -> None:
    if not results:
        print("\nNo successful experiments!")
        return
    sorted_r = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")
    print(f"{'Train':>6} {'Val':>4} {'Thr':>4} {'Hidden':>7} {'Lay':>4} "
          f"{'Drop':>5} {'LR':>9} {'Thr%':>6} {'Acc':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'Fb':>7}")
    print("-" * 110)
    for r in sorted_r[:20]:
        c = r.config_dict
        print(f"{c['train_weeks']:>6} {c['val_weeks']:>4} {c['threshold_weeks']:>4} "
              f"{c['hidden_dim']:>7} {c['num_layers']:>4} {c['dropout']:>5.2f} "
              f"{c['learning_rate']:>9.1e} {c['threshold_percentile']:>6.1f} "
              f"{r.accuracy:>7.2%} {r.precision:>7.2%} {r.recall:>7.2%} "
              f"{r.f1_score:>7.2%} {r.f_beta_score:>7.2%}")
    best = sorted_r[0]
    print(f"\nBEST (by {sort_by}): {best.config_dict}")
    print(f"  F1={best.f1_score:.2%}  Fb={best.f_beta_score:.2%}  "
          f"Prec={best.precision:.2%}  Rec={best.recall:.2%}")
    print("=" * 110)


def _save_results(results: List[ExperimentResult], path: str,
                  total: int, sort_by: str, beta: float) -> None:
    sorted_r = sorted(results, key=lambda r: getattr(r, sort_by), reverse=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "configs_tested": total,
        "completed_runs": len(results),
        "progress_pct": round(100 * len(results) / total, 1) if total else 0,
        "sort_by": sort_by,
        "beta": beta,
        "results": [r.to_dict() for r in sorted_r],
        "best_config": sorted_r[0].to_dict() if sorted_r else None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimization: hyperparameter search or data-split search"
    )
    parser.add_argument("--mode", choices=["split", "hyperparams"], required=True,
                        help="Optimization mode")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta for F-beta score (paper uses 0.1)")
    parser.add_argument("--verbose", action="store_true")

    # Split-specific
    parser.add_argument("--total-dev-weeks", type=int, default=14)
    parser.add_argument("--min-train", type=int, default=4)
    parser.add_argument("--min-val", type=int, default=2)
    parser.add_argument("--min-threshold", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Fixed hidden dim (split mode)")

    # Hyperparams-specific
    parser.add_argument("--train-weeks", type=int, default=8)
    parser.add_argument("--val-weeks", type=int, default=2)
    parser.add_argument("--threshold-weeks", type=int, default=4)
    parser.add_argument("--search", choices=["grid", "random", "focused"], default="focused")
    parser.add_argument("--n-configs", type=int, default=30,
                        help="Number of random configs (random search only)")
    parser.add_argument("--sort-by", default="f1_score",
                        choices=["f1_score", "f_beta_score", "precision", "recall", "accuracy"])

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print(f"OPTIMIZATION  --  mode={args.mode}")
    print("=" * 60)

    if args.mode == "split":
        run_split_optimization(args)
    else:
        run_hyperparams_optimization(args)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
