import streamlit as st
from transformers import pipeline

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🙂", layout="centered")

# ---------------- Load Model (Cached) ---------------- #
@st.cache_resource
def load_sentiment_model():
    # Load the Hugging Face model from your repo on CPU
    return pipeline(
        "sentiment-analysis",
        model="rohit1105/sentiment-analysis-bert",
        tokenizer="rohit1105/sentiment-analysis-bert",
        device=-1  # Force CPU
    )

# ---------------- UI ---------------- #
st.title("📝 Sentiment Analysis App")
st.write(
    "Analyze the sentiment of your text using a fine-tuned **BERT model** hosted on Hugging Face."
)

# Text input
text_input = st.text_area("✏️ Enter text for sentiment analysis:", height=150)

# Analyze button
if st.button("🔍 Analyze"):
    if text_input.strip():
        with st.spinner("Analyzing sentiment... Please wait."):
            sentiment_pipeline = load_sentiment_model()
            results = sentiment_pipeline(text_input)

        if results:
            res = results[0]
            label = res["label"]
            score = res["score"]

            # Optional mapping if model uses generic labels
            label_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            human_label = label_map.get(label, label)

            col1, col2 = st.columns(2)
            col1.markdown(f"**Sentiment:** {human_label}")
            col2.markdown(f"**Confidence:** {score:.2%}")
    else:
        st.error("⚠️ Please enter some text before clicking Analyze.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers")
