"""Stage 3 - pick the best experiment and publish its model to the Model Registry.

Scans completed training tasks in the project, ranks them by logged accuracy,
takes the best one's registered OutputModel, publishes it (making it an
immutable registry entry) and tags it ``production``/``best``.

A *published* model is what distinguishes a Registry entry from a loose artifact.

Usage:
    python src/register_best.py
"""
from __future__ import annotations

import argparse

from clearml import Model, Task

PROJECT = "sentiment-mlops"


def best_task() -> tuple[Task, float]:
    tasks = Task.get_tasks(
        project_name=PROJECT,
        task_filter={"status": ["completed"], "order_by": ["-last_update"]},
    )
    scored: list[tuple[Task, float]] = []
    for t in tasks:
        values = t.get_reported_single_values() or {}
        acc = values.get("accuracy")
        if acc is not None:
            scored.append((t, float(acc)))
    if not scored:
        raise SystemExit("No completed training tasks with an 'accuracy' value found.")
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="production")
    args = ap.parse_args()

    task, acc = best_task()
    print(f"Best task: {task.name}  (id={task.id}, accuracy={acc:.4f})")

    output_models = task.get_models().get("output", [])
    if not output_models:
        raise SystemExit("Best task has no output model registered.")

    # Re-open as a registry Model handle (supports tag editing + publish).
    model = Model(model_id=output_models[-1].id)

    # Make it a first-class registry entry: tag + publish (immutable).
    model.tags = sorted(set((model.tags or []) + ["best", args.tag]))
    if not model.published:
        model.publish()

    print("=" * 60)
    print(f"Published model to Registry: {model.name}")
    print(f"MODEL_ID = {model.id}")
    print(f"tags = {model.tags}  published = {model.published}")
    print("=" * 60)
    print("Use this MODEL_ID in Stage 4 (clearml-serving model add ...).")


if __name__ == "__main__":
    main()
