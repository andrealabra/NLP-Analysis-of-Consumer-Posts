# Import necessary libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Local file paths
no_reddit = "/Users/andrealabraorozco/Downloads/(Clean) Diabetes Geo US No Reddit 2023 50K rows.xlsx"
redditOnly = "/Users/andrealabraorozco/Downloads/(Clean) Diabetes Reddit Data Combined.xlsx"

# Load datasets
df = pd.read_excel(no_reddit)
df2 = pd.read_excel(redditOnly)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to compute sentiment using VADER
def vader_sentiment(text):
    if pd.isnull(text):
        return "neutral"  # Default for missing text
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Normalize pre-computed sentiment labels
for dataset in [df, df2]:
    dataset['Sentiment'] = dataset['Sentiment'].str.lower().str.rstrip('s')  # Converts "Negatives" to "negative"

# Apply VADER sentiment analysis and calculate agreement
for dataset, name in [(df, "No Reddit"), (df2, "Reddit Only")]:
    # Compute VADER sentiment
    dataset['VADER_Sentiment'] = dataset['Sound Bite Text'].apply(vader_sentiment)

    # Compare pre-computed vs VADER sentiment
    dataset['Agreement'] = dataset['Sentiment'] == dataset['VADER_Sentiment']

    # Calculate agreement percentage
    agreement_percentage = dataset['Agreement'].mean() * 100
    print(f"Agreement Percentage for {name} dataset: {agreement_percentage:.2f}%")

# Visualize sentiment distribution comparison
def visualize_sentiment_distribution(dataset, name):
    precomputed_dist = dataset['Sentiment'].value_counts(normalize=True)
    vader_dist = dataset['VADER_Sentiment'].value_counts(normalize=True)

    # Create bar chart
    labels = precomputed_dist.index.union(vader_dist.index)
    precomputed_values = precomputed_dist.reindex(labels, fill_value=0)
    vader_values = vader_dist.reindex(labels, fill_value=0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, precomputed_values, alpha=0.7, label='Pre-computed', width=0.4, align='center')
    plt.bar(labels, vader_values, alpha=0.7, label='VADER', width=0.4, align='edge')
    plt.title(f"Sentiment Distribution Comparison ({name})")
    plt.xlabel("Sentiment")
    plt.ylabel("Proportion")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualize No Reddit Dataset
visualize_sentiment_distribution(df, "No Reddit")

# Visualize Reddit Only Dataset
visualize_sentiment_distribution(df2, "Reddit Only")

# Visualize agreement percentages
def visualize_agreement(dataset1, dataset2):
    agreements = {
        "No Reddit": dataset1['Agreement'].mean() * 100,
        "Reddit Only": dataset2['Agreement'].mean() * 100
    }

    plt.figure(figsize=(6, 4))
    plt.bar(agreements.keys(), agreements.values(), color=['blue', 'orange'])
    plt.title("Agreement Percentage Between Pre-computed and VADER Sentiments")
    plt.ylabel("Agreement Percentage")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Visualize agreement comparison
visualize_agreement(df, df2)


# Create a bar plot for agreement percentages
agreements = {
    'Geo Data VADER': vader_geo_agreement,
    'Geo Data CoreNLP': corenlp_geo_agreement,
    'Reddit Data VADER': vader_reddit_agreement,
    'Reddit Data CoreNLP': corenlp_reddit_agreement
}

plt.bar(agreements.keys(), agreements.values())
plt.ylabel("Agreement Percentage")
plt.title("Sentiment Agreement Comparison")
plt.xticks(rotation=45)
plt.show()
