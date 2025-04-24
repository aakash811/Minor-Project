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


import streamlit as st
import os
from main import summarize_pdf

st.title("PDF Summarizer with Clustering")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Summarizing..."):
        summary = summarize_pdf("data/temp.pdf")

    st.markdown("### Summary")
    st.text(summary)
