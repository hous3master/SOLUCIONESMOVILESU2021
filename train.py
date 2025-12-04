import os
import polars as pl
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Configuration
INPUT_DIR = 'input'
OUTPUT_DIR = 'static'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'classifier.pkl')

def load_data():
    data_frames = []
    # Map filenames to categories
    files = {
        'bakery.csv': 'BAKERY',
        'canned_packaged.csv': 'CANNED_PACKAGES',
        'dairy.csv': 'DAIRY',
        'meat_seafood.csv': 'MEAT_SEAFOOD',
        'produce.csv': 'PRODUCE'
    }
    
    for filename, category in files.items():
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.exists(filepath):
            try:
                # Read csv using polars
                df = pl.read_csv(filepath)
                # The file view showed 'product' as the column name
                if 'product' in df.columns:
                    # Add category column
                    df = df.with_columns(pl.lit(category).alias('category'))
                    data_frames.append(df.select(['product', 'category']))
                else:
                    print(f"Warning: 'product' column not found in {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found in {INPUT_DIR}")
    
    if not data_frames:
        raise ValueError("No data loaded!")
        
    # Concatenate all dataframes
    return pl.concat(data_frames)

def train_and_save():
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} samples.")
    
    # 2. Prepare Data
    X = df['product'].to_list()
    y = df['category'].to_list()
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build Pipeline
    print("Building and training model...")
    # Using TF-IDF for text representation and Logistic Regression for classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # 5. Train
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluate
    print("\nEvaluation Results:")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 7. Save Artifacts
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\nModel saved to {OUTPUT_FILE}")
    
    # Test prediction with the saved artifact to ensure it works
    print("\nVerifying saved artifact...")
    with open(OUTPUT_FILE, 'rb') as f:
        loaded_model = pickle.load(f)
        
    test_words = ["Pan", "Leche", "Carne", "Manzana"]
    print(f"Testing predictions on: {test_words}")
    
    predictions = loaded_model.predict(test_words)
    for word, pred in zip(test_words, predictions):
        print(f"Input: '{word}' -> Predicted: {pred}")

if __name__ == "__main__":
    train_and_save()
