import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_quality_classifier(dataframe):
    """
    Assumes a pandas DataFrame with two columns: 
    'text' (the extracted web page text) and 'label' (1 for high-quality, 0 for low-quality)
    """
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe['text'], 
        dataframe['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # 2. Vectorize the text (convert words to numbers)
    # We limit features to 50,000 to keep memory usage reasonable
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=50000, 
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Train the model
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_vec, y_train)
    
    # Optional: Print evaluation metrics
    predictions = classifier.predict(X_test_vec)
    print(classification_report(y_test, predictions))
    
    return classifier, vectorizer

# Example usage:
# df = pd.DataFrame({'text': ["Great wiki article...", "Buy cheap meds now..."], 'label': [1, 0]})
# model, vectorizer = train_quality_classifier(df)