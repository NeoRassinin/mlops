"""Stage 2/3 - train a sentiment classifier, log everything to ClearML, run on an Agent.

What this script does (all visible in the ClearML UI):
  * creates a ClearML Task (auto-captures git commit if run from a git repo)
  * logs hyperparameters (connected -> editable when you clone/re-run in the UI)
  * pulls the dataset from ClearML by dataset_id (Stage 1)
  * logs accuracy + f1 (scalars and single values)
  * logs a confusion matrix as an image
  * saves the trained sklearn Pipeline as an artifact
  * registers the model in the ClearML Model Registry (OutputModel)

Remote execution:
  By default the script enqueues itself onto a queue (``--queue students``) via
  ``Task.execute_remotely`` and exits locally - the ClearML Agent then runs the
  whole script. Pass ``--local`` to run in-process (useful for a first smoke test).

Examples:
  # experiment 1
  python src/train.py --dataset-id <ID> --queue students \
      --ngram-max 1 --max-features 2000 --C 1.0
  # experiment 2 (different params)
  python src/train.py --dataset-id <ID> --queue students \
      --ngram-max 2 --max-features 8000 --C 4.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: works on the agent
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clearml import Dataset, OutputModel, Task

PROJECT = "sentiment-mlops"
DATASET_NAME = "sentiment-reviews"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default=None,
                    help="ClearML dataset id. If omitted, latest by name is used.")
    ap.add_argument("--queue", default="students", help="Agent queue to run on.")
    ap.add_argument("--local", action="store_true",
                    help="Run locally instead of enqueueing to an agent.")
    ap.add_argument("--task-name", default=None)
    # --- hyperparameters (vary these between experiments) ---
    ap.add_argument("--ngram-max", type=int, default=1,
                    help="TF-IDF ngram range is (1, ngram_max).")
    ap.add_argument("--max-features", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse LogReg regularization.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_dataset(dataset_id: str | None) -> pd.DataFrame:
    if dataset_id:
        ds = Dataset.get(dataset_id=dataset_id)
    else:
        ds = Dataset.get(dataset_project=PROJECT, dataset_name=DATASET_NAME)
    local_path = Path(ds.get_local_copy())
    csv_files = list(local_path.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV found in dataset {ds.id} at {local_path}")
    print(f"Loaded ClearML dataset {ds.id} -> {csv_files[0]}")
    return pd.read_csv(csv_files[0])


def main() -> None:
    args = parse_args()

    task_name = args.task_name or (
        f"train ngram1-{args.ngram_max} feat{args.max_features} C{args.C}")
    task = Task.init(project_name=PROJECT, task_name=task_name,
                     output_uri=True)  # output_uri=True -> artifacts/models to fileserver

    # Hyperparameters - connected so they show in the UI and are editable on clone.
    hparams = {
        "dataset_id": args.dataset_id or "",
        "ngram_max": args.ngram_max,
        "max_features": args.max_features,
        "C": args.C,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    task.connect(hparams)

    # Hand off to the agent (no-op when already running on the agent).
    if not args.local:
        print(f"Enqueueing task to '{args.queue}' for remote execution...")
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    # ---- from here on this runs on the Agent ----
    logger = task.get_logger()

    df = load_dataset(hparams["dataset_id"] or None)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=hparams["test_size"],
        random_state=hparams["seed"], stratify=df["label"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, hparams["ngram_max"]),
                                  max_features=hparams["max_features"])),
        ("clf", LogisticRegression(C=hparams["C"], max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    print(f"accuracy={acc:.4f}  f1={f1:.4f}")

    # Metrics: single values (Scalars > "single values") + a scalar series.
    logger.report_single_value("accuracy", acc)
    logger.report_single_value("f1", f1)
    logger.report_scalar("metrics", "accuracy", value=acc, iteration=0)
    logger.report_scalar("metrics", "f1", value=f1, iteration=0)

    # Confusion matrix — logged to PLOTS (interactive heatmap, renders inline in the
    # ClearML web UI, no fileserver fetch) and ALSO as an image in Debug Samples.
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # -> PLOTS
    logger.report_confusion_matrix(
        title="Confusion Matrix", series="test", matrix=cm, iteration=0,
        xaxis="Predicted", yaxis="Actual", xlabels=labels, ylabels=labels)
    # -> Debug Samples (the same matrix as a rendered image)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    logger.report_matplotlib_figure(
        title="Confusion Matrix", series="test", figure=fig,
        iteration=0, report_image=True)
    plt.close(fig)

    # Save model as an ARTIFACT.
    model_path = Path("model.pkl")
    joblib.dump(pipeline, model_path)
    task.upload_artifact("model", artifact_object=str(model_path))

    # Register the model in the Model Registry (linked to this task).
    output_model = OutputModel(
        task=task, name="sentiment-clf", framework="ScikitLearn",
        tags=["sentiment", "tfidf-logreg",
              f"acc={acc:.3f}", f"f1={f1:.3f}"])
    try:
        output_model.set_metadata("accuracy", str(acc), v_type="float")
        output_model.set_metadata("f1", str(f1), v_type="float")
    except Exception as exc:  # metadata API differs across versions; non-fatal
        print(f"set_metadata skipped: {exc}")
    output_model.update_weights(weights_filename=str(model_path), auto_delete_file=False)

    print("Done. Model registered:", output_model.id)


if __name__ == "__main__":
    main()
