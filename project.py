import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def load_and_clean_data():
    """Load and preprocess the Amazon reviews dataset"""
    try:
        df = pd.read_csv('processed_amazon_reviews.csv')
        
        # Ensure rating column is numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])  # Remove rows with invalid ratings
        
        # Clean text data
        def clean_text(text):
            text = str(text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            return text
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        df['cleaned_text'] = df['reviewText'].apply(clean_text)
        df['cleaned_text'] = df['cleaned_text'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words])
        )
        
        # Convert ratings to sentiment labels
        df['sentiment'] = df['rating'].apply(
            lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive')
        )
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_and_evaluate():
    """Train and evaluate sentiment analysis models"""
    df = load_and_clean_data()
    if df is None:
        return
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    
    best_model = None
    best_accuracy = 0
    
    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['negative','neutral','positive'],
                   yticklabels=['negative','neutral','positive'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    # Save best model and vectorizer
    if best_model:
        joblib.dump(best_model, 'sentiment_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        print(f"\nSaved best model with accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_and_evaluate()