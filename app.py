# import streamlit as st
# from main import summarize_pdf

# st.title("PDF Summarizer with Clustering")
# uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# if uploaded_file:
#     with open("data/temp.pdf", "wb") as f:
#         f.write(uploaded_file.read())
#     summary = summarize_pdf("data/temp.pdf")
#     st.subheader("Summary")
#     st.text(summary)

import os
import streamlit as st
from main import summarize_pdf
from utils.pdf_utils import extract_text_from_pdf
from utils.sentiment import get_sentiment

from docx import Document 

st.title("📄 PDF/Word/Text Summarizer with Clustering & Sentiment")

# ✅ Input method selector
input_mode = st.radio("Choose input method:", ("Upload File (PDF/Word)", "Enter Text Manually"))

text = ""
file_path = "data/temp"

# ✅ Option 1: Upload PDF or Word
if input_mode == "Upload File (PDF/Word)":
    uploaded_file = st.file_uploader("Upload a file (PDF or Word)", type=["pdf", "docx"])

    if uploaded_file:
        if not os.path.exists("data"):
            os.makedirs("data")

        # Save the uploaded file
        file_extension = uploaded_file.name.split('.')[-1]
        full_file_path = f"{file_path}.{file_extension}"

        with open(full_file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Extracting text and analyzing sentiment..."):
            if file_extension == "pdf":
                text = extract_text_from_pdf(full_file_path)
            elif file_extension == "docx":
                doc = Document(full_file_path)
                text = "\n".join([para.text for para in doc.paragraphs])

            sentiment = get_sentiment(text)

# ✅ Option 2: Enter text manually
elif input_mode == "Enter Text Manually":
    text = st.text_area("Enter your text here:")
    if text:
        with st.spinner("Analyzing sentiment..."):
            sentiment = get_sentiment(text)

# ✅ Show sentiment if text is available
if text:
    st.markdown("### 🧠 Sentiment Analysis")
    st.markdown(f"**Sentiment:** {sentiment}")

    # ✅ Summarize button
    if st.button("✨ Summarize"):
        with st.spinner("Summarizing..."):
            if input_mode == "Upload File (PDF/Word)":
                summary = summarize_pdf(
                    full_file_path if file_extension == "pdf" else None,
                    custom_text=text if file_extension == "docx" else None
                )
            else:
                from utils.text_utils import clean_and_split_sentences
                from models.embedding import train_word2vec, get_embeddings
                from models.clustering import cluster_sentences
                from models.dl_scorer import build_scoring_model
                from models.dl_scorer_ga import run_ga

                sentences = clean_and_split_sentences(text)
                summary = summarize_pdf(pdf_path=None, custom_text=text)

        st.markdown("### 📝 Summary")
        summary_text = ' '.join(summary)
        st.markdown(summary_text)

