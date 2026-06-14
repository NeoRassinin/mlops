"""Stage 5 - Streamlit UI for the sentiment endpoint.

This UI talks to the clearml-serving HTTP endpoint ONLY. It never imports the
model or any ML library - all inference happens server-side over HTTP.

Run:
    streamlit run ui/app.py
Configure the endpoint via the sidebar or the SERVING_URL env var, e.g.
    SERVING_URL=http://localhost:9090/serve/sentiment streamlit run ui/app.py
"""
import os
import time

import requests
import streamlit as st

DEFAULT_URL = os.environ.get("SERVING_URL", "http://localhost:9090/serve/sentiment")

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")
st.title("💬 Sentiment Classifier")
st.caption("UI talks to the clearml-serving endpoint over HTTP - no local model.")

endpoint = st.sidebar.text_input("Serving endpoint URL", value=DEFAULT_URL)
st.sidebar.markdown(
    "Expected payload: `{\"text\": \"...\"}`\n\n"
    "Served by clearml-serving from the ClearML Model Registry.")

text = st.text_area("Enter text to classify", height=120,
                    placeholder="e.g. This product is absolutely amazing!")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            start = time.perf_counter()
            resp = requests.post(endpoint, json={"text": text}, timeout=10)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            data = resp.json()

            pred = data["predictions"][0]
            label = pred["label"]
            score = pred.get("score")

            color = "green" if label.lower() == "positive" else "red"
            st.markdown(f"### Prediction: :{color}[{label}]")
            cols = st.columns(2)
            if score is not None:
                cols[0].metric("Confidence", f"{score:.1%}")
            cols[1].metric("Latency", f"{latency_ms:.0f} ms")

            if pred.get("scores"):
                st.write("Class probabilities:")
                st.bar_chart(pred["scores"])

            with st.expander("Raw response"):
                st.json(data)

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not reach the endpoint at `{endpoint}`. "
                     "Is the serving service up?")
        except requests.exceptions.Timeout:
            st.error("❌ The endpoint timed out (10s).")
        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ Endpoint returned {exc.response.status_code}: "
                     f"{exc.response.text[:300]}")
        except (KeyError, ValueError) as exc:
            st.error(f"❌ Unexpected response format: {exc}")
