# Text Processing

###############################################################################
"""
In this .py file, we will clean games reviews and conduct sentimental analysis.
"""
###############################################################################

# import packages
import pandas as pd
import sys
from textblob import TextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
# In the future may adopt different packages for sentimental analysis
from transformers import pipeline
import re
from wordcloud import WordCloud
import spacy
import en_core_web_sm
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

PATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works'
DATAPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\data'


# %% Read Data from DataCleaning

sys.path.append(PATH)
import DataCleaning as data
games = data.import_and_clean_games()
twitch = data.import_and_clean_twitch()


# %% Word cloud

def clean_reviews(text):
    """Substitute selected patterns."""

    def apply_re_subs(text, substitutions):
        """Replace incorrect patterns in a text with correct substitution."""
        for pattern, replacement in substitutions.items():
            text = re.sub(pattern, replacement, text)
        return text

    # create a dictionary with unwanted patterns and intended edited patterns
    subs = {
            r'\[': r'',
            r'\]': r'',
            r'\\n': '',
            r'ï¿½': '',
            r' s ': ''
            }

    text = apply_re_subs(text, subs)
    return text


def remove_non_ascii(text):
    """Clean text by removing non-ASCII characters."""
    return ''.join([char for char in text if ord(char) < 128])


def extract_keywords(text):
    """Extract adjectives and nouns from text."""
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000  # Adjust the value based on text length
    cleaned_text = remove_non_ascii(text)
    doc = nlp(cleaned_text)
    stopwords = {'n'}  # Add any additional stopwords as needed
    keywords = [token.text for token in doc if token.pos_ in ('ADJ', 'NOUN')
                and token.text.lower() not in stopwords]
    return ' '.join(keywords)


def filter_positive_reviews(dataframe, filter_col, text_col, quantile):
    """Filter reviews based on the quantiles of another column."""
    rating = dataframe[filter_col].quantile([quantile])[quantile]
    text_data = ' '.join(dataframe[dataframe[filter_col] > rating][text_col])
    return text_data


def filter_negative_reviews(dataframe, filter_col, text_col, quantile):
    """Filter reviews based on the quantiles of another column."""
    rating = dataframe[filter_col].quantile([quantile])[quantile]
    text_data = ' '.join(dataframe[dataframe[filter_col] < rating][text_col])
    return text_data


# Function execution
games['cleanReviews'] = games['Reviews'].apply(clean_reviews)
positive_text_data = filter_positive_reviews(games, 'Rating', 'cleanReviews',
                                             0.9)
negative_text_data = filter_positive_reviews(games, 'Rating', 'cleanReviews',
                                             0.1)
positive_cleaned_text_keywords = extract_keywords(positive_text_data)
negative_cleaned_text_keywords = extract_keywords(negative_text_data)
# perform wordcloud specific text cleaning, removing useless texts like "game"
# for wordcloud, but we do not remove them from reviews
for pattern in [r"game", r"time", r" s "]:
    for sub in ["", "", ""]:
        positive_cleaned_text_keywords = re.sub(pattern, sub,
                                                positive_cleaned_text_keywords)
        negative_cleaned_text_keywords = re.sub(pattern, sub,
                                                negative_cleaned_text_keywords)


# %% Sentiment analysis

# use vader_lexicon, a lexicon and rule-based sentiment analysis tool that is
# specifically attuned to sentiments expressed in social media
# nltk.download('vader_lexicon')


def analyze_sentiment(dataframe, text_column):
    """Perform sentiment analysis and add a new 'Sentiment' column."""
    sia = SentimentIntensityAnalyzer()
    dataframe['Sentiment'] = dataframe[text_column].apply(
        lambda x: sia.polarity_scores(x)['compound'])


def create_sentiment_categories(dataframe):
    """Create sentiment categories."""
    dataframe['Sentiment_Category'] = pd.cut(dataframe['Sentiment'],
                                             bins=[-1, -0.1, 0.1, 1],
                                             labels=['Negative', 'Neutral',
                                                     'Positive'])


def analyze_subjectivity(review):
    """Perform sentiment analysis and add a new 'Subjectivity' column."""
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob');
    analysis = TextBlob(review)
    return analysis.sentiment.subjectivity


# Function execution
analyze_sentiment(games, 'cleanReviews')
create_sentiment_categories(games)
# Print sentiment statistics and category counts
sentiment_stats = games['Sentiment'].describe()
sentiment_category_counts = games['Sentiment_Category'].value_counts()


# Comparison of sentiment category distribution
actblz_sentiment_counts = games[games['actblz_indicator'] == 1]\
    .groupby('Sentiment_Category').count()
actblz_sentiment_percentages = (actblz_sentiment_counts /
                                actblz_sentiment_counts.sum() * 100)[
                                    'Sentiment']
games_sentiment_counts = games[games['actblz_indicator'] == 0]\
    .groupby('Sentiment_Category').count()
games_sentiment_percentages = (games_sentiment_counts / games_sentiment_counts
                               .sum() * 100)['Sentiment']


# Merge after text processing
def merge_data_for_regression(twitch_df):
    """Perform merge on twitch df so it can combine games information."""
    # perform the merge after text processing to have sentiment columns
    twitch_merge = twitch_df.merge(games[['Title', 'Team', 'Year', 'Publisher',
                                          'N_of_players', 'actblz_indicator',
                                          'Sentiment', 'Sentiment_Category']],
                                   left_on='Game', right_on='Title',
                                   how='inner')
    twitch_merge['Game_same'] = twitch_merge['Game'].copy()
    twitch_merge['Game_same'] = twitch_merge['Game_same'].\
        apply(lambda x: 'Call of Duty' if 'Call of Duty' in x else x)
    actblz_stats = twitch_merge[twitch_merge['actblz_indicator'] == 1]
    return twitch_merge, actblz_stats
