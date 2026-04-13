import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    return text  # DO NOT remove punctuation

def preprocess_data(df):
    df = df[['feedback_text', 'sentiment_label']].dropna()
    df['clean_text'] = df['feedback_text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/feedback.csv")
    df = preprocess_data(df)
    print(df.head())