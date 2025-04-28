from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]
    return result['label'].lower()
