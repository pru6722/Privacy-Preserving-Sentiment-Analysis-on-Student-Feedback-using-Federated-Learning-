import pandas as pd

# Load reviews dataset
df = pd.read_csv("../data/coursera_reviews.csv")

# Keep only required columns
df = df[['reviews', 'rating']].dropna()

# Convert rating → sentiment
def convert_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df['sentiment'] = df['rating'].apply(convert_sentiment)

# Rename column for consistency
df = df.rename(columns={'reviews': 'text'})

# Save cleaned dataset
df.to_csv("../data/cleaned_reviews.csv", index=False)

print("Dataset prepared successfully ✅")
print(df['sentiment'].value_counts())