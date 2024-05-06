import json
import os
from multiprocessing import cpu_count, Pool
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from langdetect import detect, detect_langs, DetectorFactory
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob

def download_missing_nltk_dataset():
    try:
        tokens = word_tokenize("blaa")
        print('punkt exist')
    except:
        print('punkt exists')
        nltk.download('punkt')

    try:
        stop_words = set(stopwords.words('english'))
        print('stopwords exist')
    except:
        print('stopwords exists')
        nltk.download('stopwords')

    try:
        lemmatizer = WordNetLemmatizer()
        print('wordnet exist')
    except:
        print('wordnet exists')
        nltk.download('wordnet')
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

    DetectorFactory.seed = 42
    lang = detect_langs(text)
    tokens = word_tokenize(text)
    return lang, len(tokens)


def filter_dataframe(df, column_name, search_string):
    # Split the search string into individual words if no exact phrase is provided
    search_terms = search_string.split() if ' ' in search_string else [search_string]

    # Initialize an empty list to store the rows that match the search criteria
    matching_rows = []

    # Iterate over each word in search_terms and filter rows where the term is found in the column
    for term in search_terms:
        # Create a boolean mask indicating rows where the term is found in the column
        mask = df[column_name].str.contains(term, case=False, regex=True)

        # Append the rows where the term is found to the matching_rows list
        matching_rows.extend(df[mask].index.tolist())

    # Remove duplicates from the matching_rows list
    matching_rows = list(set(matching_rows))

    # Create a new DataFrame with only the matching rows
    filtered_df = df.loc[matching_rows]

    # Sort the filtered DataFrame by index
    filtered_df.sort_index(inplace=True)

    return filtered_df

def word_count_histogram(df, column_name):
    # Count the number of words in each row of the specified column
    word_counts = df[column_name].str.split().apply(len)

    # Create a histogram using Plotly
    hist_data, bins = np.histogram(word_counts, bins='auto')

    return hist_data, bins

def word_count_violin_plot(df, column_name):
    # Count the number of words in each row of the specified column
    word_counts = df[column_name].str.split().apply(len)

    # Create a DataFrame for Plotly
    data = pd.DataFrame({'Word Count': word_counts})

    # Create a violin plot using Plotly
    fig = px.violin(data, y='Word Count', box=True, points='all')
    fig.update_layout(title='Word Count of Each Post Violin Plot',
                      yaxis_title='Number of Words')
    fig.show()


def calculate_word_count_statistics(df, column_name):
    # Count the number of words in each row of the specified column
    word_counts = df[column_name].str.split().apply(len)

    # Calculate mean, median, and standard deviation of word counts
    # Calculate mean, median, and standard deviation of word counts
    mean_value = round(word_counts.mean(), 2)
    median_value = round(word_counts.median(), 2)
    std_value = round(word_counts.std(), 2)

    return mean_value, median_value, std_value
def calculate_word_count(series):
    # Initialize a dictionary to store word counts for each language
    word_counts = {}

    # Iterate over each value in the series
    for item in series:
        # Extract language codes and word counts from the value
        languages, count = item
        # Split the languages and their probabilities
        # Iterate over each language and its corresponding word count
        word_count = int(count)
        for lang in languages:
            lang_code = lang.lang
            prob = lang.prob
            # Update the word count for the language in the dictionary
            if lang_code in word_counts:
                word_counts[lang_code] += word_count*prob
            else:
                word_counts[lang_code] = word_count*prob

    return word_counts


def parallelize_series_processing(series, func, num_processes=None):
    num_processes = num_processes or cpu_count()

    with Pool(processes=num_processes) as pool:
        results = pool.map(func, series)

    return pd.Series(results)


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq
def data_initial_statistics(df):



    # mean_val, median_val, std_val = calculate_word_count_statistics(df, 'blog_post')
    # print("Mean Word Count:", mean_val)
    # print("Median Word Count:", median_val)
    # print("Standard Deviation of Word Count:", std_val)


    ###################################
    #
    # matching_df = filter_dataframe(df, 'blog_post', "Öl")


    ###################################



    #
    # hist_data, bins = word_count_histogram(df, 'blog_post')
    # fig = go.Figure(data=[go.Bar(x=bins, y=hist_data)])
    # fig.update_layout(title='Word Count Histogram of Each Post',
    #                    xaxis_title='Number of Words',
    #                    yaxis_title='Frequency')
    # fig.show()
    #




    ###################################

    # word_count_violin_plot(df, 'blog_post')


    ###################################
    #
    # filename = "language_word_counts_rounded.json"
    # # Check if language_word_counts_rounded is already stored
    # if os.path.exists(filename):
    #     # Load the stored result
    #     with open(filename, 'r') as file:
    #         language_word_counts_rounded = json.load(file)
    # else:
    #     # Calculate language word counts for each row in the DataFrame
    #     #language_word_count = df["blog_post"].apply(detect_and_count_words)
    #
    #
    #     # Parallelize the execution of detect_and_count_words across the DataFrame
    #     language_word_count = parallelize_series_processing(df["blog_post"], detect_and_count_words)
    #
    #     # Calculate word counts for each language
    #     language_word_counts = calculate_word_count(language_word_count)
    #     # Round the values to the nearest integer
    #     language_word_counts_rounded = {lang: round(count) for lang, count in language_word_counts.items()}
    #     # Save the result
    #     with open(filename, 'w') as file:
    #         json.dump(language_word_counts_rounded, file)
    #
    # # Combine languages with a percentage below 1 into "other"
    # threshold = 0.01
    # total_word_count = sum(language_word_counts_rounded.values())
    # language_word_counts_combined = {'other': 0}
    # for lang, count in language_word_counts_rounded.items():
    #     if count / total_word_count < threshold:
    #         language_word_counts_combined['other'] += count
    #     else:
    #         language_word_counts_combined[lang] = count
    #
    # # Print total word count for each language
    # print("Total word count for each language:")
    # for lang, count in language_word_counts_combined.items():
    #     print(f"{lang}: {count}")
    #
    # # Create labels and values for the pie chart
    # labels = list(language_word_counts_combined.keys())
    # values = list(language_word_counts_combined.values())
    #
    # # Create a pie chart using Plotly
    # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    # fig.update_layout(title='Total word count for each language')
    # fig.show()

    ###################################

    # n = 3
    # filename = f"{n}_ngram_result.json"
    # if os.path.exists(filename):
    #     with open(filename, 'r') as file:
    #         top_nrams = json.load(file)
    # else:
    #     top_nrams = get_top_ngram(df["blog_post"], n=n)
    #     top_nrams = [(item[0], int(item[1])) for item in top_nrams]
    #
    #     with open(filename, 'w') as file:
    #         json.dump(top_nrams, file)
    # top_n = 30
    # top_nrams = top_nrams[:top_n]
    # x, y = map(list, zip(*top_nrams))
    # # Reverse the order of the data
    # x = x[::-1]
    # y = y[::-1]
    # # Create a bar plot using Plotly
    # fig = go.Figure(data=[go.Bar(x=y, y=x, orientation='h')])
    # fig.update_layout(title='Bar Plot', xaxis_title='Count', yaxis_title='Category')
    # fig.show()




    ###################################



def main():

    download_missing_nltk_dataset()


    # Assuming your CSV file is named 'example.csv'
    csv_file_path = r'data\posts.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path, header=None, names=["blog_post"])

    # Print the first few rows of the DataFrame
    print(df.head())
    print(len(df.index))
    df.dropna(inplace=True)
    print(len(df.index))


    data_initial_statistics(df)







    # Apply preprocessing function to each blog post
    #df['preprocessed_blog_post'] = df['blog_post'].apply(preprocess_text)
   # x=1


if __name__ == '__main__':
    main()
