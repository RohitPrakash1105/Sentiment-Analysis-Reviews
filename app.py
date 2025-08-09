import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("rohit1105/sentiment-analysis-bert")
model = AutoModelForSequenceClassification.from_pretrained("rohit1105/sentiment-analysis-bert")

st.title("Sentiment Analysis App")
st.write("Enter a review and see the predicted sentiment.")

text = st.text_area("Your review:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Prediction: **{labels[prediction]]}**")
