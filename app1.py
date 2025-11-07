import streamlit as st
from transformers import pipeline
import torch

# -------------------------------------------------------------
# 🔧 Ensure PyTorch is available
# -------------------------------------------------------------
if not torch.cuda.is_available():
    device = "cpu"
else:
    device = 0  # use GPU if available

# -------------------------------------------------------------
# 🚀 Load the summarization model
# -------------------------------------------------------------
@st.cache_resource  # caches the model so it loads only once
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

summarizer = load_summarizer()

# -------------------------------------------------------------
# 🎨 Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="AI Text Summarizer 🧠", layout="centered")

st.title("🧠 AI Text Summarizer")
st.write("Enter or paste your long text below and get a concise summary instantly!")

# Input area
text = st.text_area("✏️ Enter Text:", height=200, placeholder="Paste your paragraph here...")

# Summary length control
summary_length = st.slider("📏 Summary length (approx. words):", 30, 200, 80)

# Button to summarize
if st.button("✨ Summarize"):
    if text.strip():
        with st.spinner("Summarizing... Please wait ⏳"):
            try:
                # Generate summary
                summary = summarizer(
                    text,
                    max_length=summary_length,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']

                # Display result
                st.subheader("📄 Summary:")
                st.success(summary)

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
    else:
        st.warning("Please enter some text to summarize!")

# Footer
st.markdown("""
---
🧩 *Built with [Hugging Face Transformers](https://huggingface.co/) and [Streamlit](https://streamlit.io/).*
""")

