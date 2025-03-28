import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing
def preprocess(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Sample review to analyze
review = "The mouse is useful and sleek but it broke down in a few months so not great for me"

# Preprocess and predict
cleaned_review = preprocess(review)
vectorized = vectorizer.transform([cleaned_review])
prediction = model.predict(vectorized)[0]
probabilities = model.predict_proba(vectorized)[0]

print(f"\nReview: {review}")
print(f"Predicted Sentiment: {prediction}")
print("Confidence Scores:")
print(f"Negative: {probabilities[0]:.2%}")
print(f"Neutral: {probabilities[1]:.2%}") 
print(f"Positive: {probabilities[2]:.2%}")

# Product recommendation logic
def get_recommendation(probabilities):
    positive = probabilities[2]
    negative = probabilities[0]
    neutral = probabilities[1]
    
    if positive > 0.8 and negative < 0.1 and 'broke' not in review.lower():
        return "STRONG BUY (Highly positive reviews)"
    elif positive > 0.6 and negative < 0.2 and 'broke' not in review.lower():
        return "BUY (Mostly positive)"
    elif negative > 0.3 or 'broke' in review.lower() or 'not great' in review.lower():
        return "CAUTION (Mixed reviews with durability concerns)"
    elif positive > negative and positive > neutral:
        return "Consider (Mixed but leans positive)"
    else:
        return "Neutral (Inconclusive reviews)"

recommendation = get_recommendation(probabilities)
print(f"\nRecommendation: {recommendation}")
