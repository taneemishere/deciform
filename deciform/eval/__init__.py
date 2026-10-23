"""Evaluation harness: accuracy, calibration, coverage, latency, cost."""

from deciform.eval.metrics import (
    FieldRecord, accuracy, auarc, brier, coverage_curve, ece, latency_stats,
)
from deciform.eval.runner import EvalReport, run_task
from deciform.eval.tasks import EvalExample, EvalTask, TASKS

__all__ = [
    "EvalExample",
    "EvalReport",
    "EvalTask",
    "FieldRecord",
    "TASKS",
    "accuracy",
    "auarc",
    "brier",
    "coverage_curve",
    "ece",
    "latency_stats",
    "run_task",
]
