"""clearml-serving custom engine for the sklearn sentiment Pipeline.

The registered model is a full sklearn Pipeline (TF-IDF + LogReg), so it accepts
raw text directly. clearml-serving downloads the model weights from the ClearML
Model Registry and calls these hooks - the model is NEVER loaded by the UI/client.

Request  body: {"text": "I love this"}        or {"text": ["a", "b"]}
Response body: {"predictions": [{"label": "positive", "score": 0.98, "scores": {...}}]}
"""
from typing import Any


class Preprocess:
    def __init__(self):
        self.model = None

    def load(self, local_file_name: str) -> Any:
        """Called once by the serving worker; loads weights pulled from the Registry."""
        import joblib
        self.model = joblib.load(local_file_name)
        return self.model

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        text = body.get("text", "")
        if isinstance(text, str):
            text = [text]
        return list(text)

    def process(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> Any:
        preds = self.model.predict(data)
        # predict_proba is available because the Pipeline ends in LogisticRegression
        proba = self.model.predict_proba(data)
        return preds, proba

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        preds, proba = data
        classes = [str(c) for c in self.model.classes_]
        out = []
        for i, label in enumerate(preds):
            row = proba[i]
            out.append({
                "label": str(label),
                "score": float(max(row)),
                "scores": {c: float(p) for c, p in zip(classes, row)},
            })
        return {"predictions": out}
