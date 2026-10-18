import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from expert_system import analyze_threat

st.set_page_config(page_title="Cybersecurity Threat Detection", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    with open(os.path.join(BASE_DIR, "features.json"), encoding="utf-8") as f:
        features = json.load(f)
    with open(os.path.join(BASE_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    return model, features, metrics


st.title("🛡️ AI + Expert System: Network Threat Detection")
st.caption(
    "A Random Forest model flags suspicious traffic, then expert rules explain "
    "the risk and suggest next steps. Educational demo, not a production security tool."
)

try:
    model, FEATURES, metrics = load_assets()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Check that model.joblib, features.json and metrics.json are in the repo, "
        "that model.joblib is a real file (not a Git LFS pointer), and that the "
        "scikit-learn version in requirements.txt matches the one used for training."
    )
    st.stop()

RISK_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

tab1, tab2, tab3 = st.tabs(["🔍 Single record", "📁 Batch CSV", "📊 Model performance"])

# ---------------- Single record ----------------
with tab1:
    st.subheader("Analyze one traffic record")
    defaults = {
        "Destination Port": 80,
        "Flow Duration": 1000000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Flow Bytes/s": 5000.0,
        "Flow Packets/s": 20.0,
        "SYN Flag Count": 0,
        "ACK Flag Count": 1,
    }
    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            values[feat] = st.number_input(
                feat, value=float(defaults.get(feat, 0)), format="%.4f", key=f"in_{feat}"
            )

    if st.button("Analyze", type="primary"):
        row = pd.DataFrame([values], columns=FEATURES)
        pred = int(model.predict(row)[0])
        result = analyze_threat(values, pred)

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[0][1])

        st.markdown(f"### {RISK_ICON.get(result['risk'], '')} {result['threat']}")
        c1, c2 = st.columns(2)
        c1.metric("Risk level", result["risk"])
        if proba is not None:
            c2.metric("Attack probability", f"{proba:.1%}")
        st.write(f"**Reason:** {result['reason']}")
        st.write(f"**Suggested action:** {result['action']}")

# ---------------- Batch CSV ----------------
with tab2:
    st.subheader("Analyze a CSV of traffic records")
    st.write("Required columns: " + ", ".join(f"`{f}`" for f in FEATURES))
    up = st.file_uploader("Upload CSV", type="csv")

    if up is not None:
        try:
            df = pd.read_csv(up, low_memory=False)
            df.columns = df.columns.str.strip()
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
                X = X.replace([float("inf"), float("-inf")], float("nan"))
                preds = model.predict(X)

                results = [
                    analyze_threat(X.iloc[i].to_dict(), int(preds[i]))
                    for i in range(len(X))
                ]
                out = df.copy()
                out["AI Prediction"] = ["Attack" if p == 1 else "Benign" for p in preds]
                out["Threat"] = [r["threat"] for r in results]
                out["Risk"] = [r["risk"] for r in results]
                out["Suggested Action"] = [r["action"] for r in results]

                a, b, c = st.columns(3)
                a.metric("Records", f"{len(out):,}")
                b.metric("Flagged as attack", f"{int((preds == 1).sum()):,}")
                c.metric("High risk", f"{int((out['Risk'] == 'High').sum()):,}")

                st.dataframe(out.head(1000), use_container_width=True)
                st.download_button(
                    "Download results",
                    out.to_csv(index=False).encode("utf-8"),
                    "threat_analysis_results.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ---------------- Metrics ----------------
with tab3:
    st.subheader("Test-set performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1 score", f"{metrics['f1_score']:.2%}")
    st.caption(f"Evaluated on {metrics['test_records']:,} held-out records.")

    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    st.pyplot(fig)

    st.warning(
        "Very high scores on CIC-IDS2017 are common but may not carry over to real "
        "networks. Treat the results as a demonstration."
    )
