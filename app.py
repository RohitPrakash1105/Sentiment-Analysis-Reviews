import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# App title
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Analyze the sentiment of any review using a fine-tuned BERT model hosted on Hugging Face.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("rohit1105/sentiment-analysis-bert")
    model = AutoModelForSequenceClassification.from_pretrained("rohit1105/sentiment-analysis-bert")
    return tokenizer, model

tokenizer, model = load_model()

# User input
text = st.text_area("✏️ Enter your review here:", height=150)

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=1).item()
    return prediction, probs[0].tolist()

# Button action
if st.button("🔍 Analyze Sentiment"):
    if text.strip():
        label_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
        pred, probs = predict_sentiment(text)

        # Color-coded sentiment display
        if pred == 0:
            st.error(f"**Prediction:** {label_map[pred]}")
        elif pred == 1:
            st.warning(f"**Prediction:** {label_map[pred]}")
        else:
            st.success(f"**Prediction:** {label_map[pred]}")

        # Show probability bars
        st.subheader("Confidence Scores")
        st.progress(int(probs[pred] * 100))
        st.write(f"**Negative:** {probs[0]:.2f}")
        st.write(f"**Neutral:** {probs[1]:.2f}")
        st.write(f"**Positive:** {probs[2]:.2f}")
    else:
        st.warning("Please enter some text before analyzing.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face Transformers](https://huggingface.co/).")
