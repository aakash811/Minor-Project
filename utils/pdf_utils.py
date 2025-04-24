import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def extract_sentences_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    # Clean and split into sentences
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10] 
