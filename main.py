import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from langdetect import detect, detect_langs

nltk.download('punkt')
nltk.download('stopwords')

# Data Preprocessing
# Define preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Classification
def classify_sentiment(text):
    # Implementiere Sentiment-Analyse
    return sentiment_label

def classify_topic(text):
    # Implementiere Themenklassifizierung
    return topic_label

# Clustering
def cluster_texts(texts, num_clusters):
    # Implementiere Text-Clustering
    return clusters

# Result Presentation
def visualize_clusters(clusters):
    # Implementiere Visualisierung der Clustering-Ergebnisse
    plt.scatter(clusters[:, 0], clusters[:, 1], c='blue', alpha=0.5)
    plt.show()

# Function to detect language and count words
def detect_and_count_words(text):
    lang = detect_langs(text)
    tokens = word_tokenize(text)
    return lang, len(tokens)


def main():
    # Assuming your CSV file is named 'example.csv'
    csv_file_path = r'C:\Data\ai_case_study\data\posts.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path, header=None, names=["Blog_Post"])

    # Print the first few rows of the DataFrame
    print(df.head())


    # Calculate language and word count for each blog post
    language_word_count = df["Blog_Post"].head(10).apply(detect_and_count_words)

    # Separate language_word_count into English and German counts
    # Separate language_word_count into English and German counts
    english_word_count = language_word_count[
        language_word_count.apply(lambda x: any(lang.lang == 'en' for lang in x[0]))].apply(lambda x: x[1])
    german_word_count = language_word_count[
        language_word_count.apply(lambda x: any(lang.lang == 'de' for lang in x[0]))].apply(lambda x: x[1])
    total_words = sum(language_word_count.apply(lambda x: x[1]))


    # Total word count for English and German blog posts within those 10 rows
    total_english_words = english_word_count.sum()
    total_german_words = german_word_count.sum()
    #total_words = total_english_words + total_german_words
    total_other_words = total_words-total_german_words-total_english_words

    # Calculate percentages
    english_percentage = (total_english_words / total_words) * 100
    german_percentage = (total_german_words / total_words) * 100
    other_percentage = (total_other_words / total_words) * 100

    # Print word counts and percentages
    print(f"Total English words: {total_english_words}, Percentage: {english_percentage:.2f}%")
    print(f"Total German words: {total_german_words}, Percentage: {german_percentage:.2f}%")
    print(f"Total other words: {total_other_words}, Percentage: {other_percentage:.2f}%")


    # Apply preprocessing function to each blog post
    df['Preprocessed_Blog_Post'] = df['Blog_Post'].apply(preprocess_text)
    x=1


if __name__ == '__main__':
    main()
