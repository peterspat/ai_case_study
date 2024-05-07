import ast
import json
import os
import pickle
from collections import Counter
from copy import copy
from multiprocessing import cpu_count, Pool
from gensim import corpora, models
import matplotlib.pyplot as plt
import scienceplots
from sklearn import metrics
from sklearn.cluster import KMeans

plt.style.use('science')

from HanTa import HanoverTagger as ht
from sklearn.feature_extraction.text import CountVectorizer
from spellchecker.spellchecker import SpellChecker

import spacy

import de_core_news_md #spacy download de_core_news_md

from collections import defaultdict
from textblob_de import TextBlobDE as TextBlobDE  # 2
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
from textblob import TextBlob as tb
from tqdm import tqdm


def download_missing_nltk_dataset():

    try:
        tokens = word_tokenize("blaa")
        print('punkt exist')
    except:
        print('punkt not exists')
        nltk.download('punkt')

    try:
        stop_words = set(stopwords.words('english'))
        print('stopwords exist')
    except:
        print('stopwords not exists')
        nltk.download('stopwords')

    try:
        lemmatizer = WordNetLemmatizer()
        print('wordnet exist')
    except:
        print('wordnet not exists')
        nltk.download('wordnet')
# Data Preprocessing
# Define preprocessing function



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
    return fig


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
    num_processes = num_processes or cpu_count()-3

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


def polarity(text):
    return TextBlobDE(text).sentiment.polarity


def parallelize_dataframe(df, func):
    with Pool(cpu_count()-3) as pool:
        result_list = pool.map(func, df.iterrows())
    return pd.Series(result_list, index=df.index)

# Assuming you have already defined the polarity function
def compute_polarity(row):
    index, data = row
    return polarity(data['blog_post'])
def sentiment(x):
    if x < 0:
        return 'negative'
    elif x == 0:
        return 'neutral'
    else:
        return 'positive'


def language_word_counts_rounded_calc(df):

    filename = "language_word_counts_rounded.json"
    # Check if language_word_counts_rounded is already stored
    if os.path.exists(filename):
        # Load the stored result
        with open(filename, 'r') as file:
            language_word_counts_rounded = json.load(file)
    else:
        # Calculate language word counts for each row in the DataFrame
        #language_word_count = df["blog_post"].apply(detect_and_count_words)


        # Parallelize the execution of detect_and_count_words across the DataFrame
        language_word_count = parallelize_series_processing(df["blog_post"], detect_and_count_words)

        # Calculate word counts for each language
        language_word_counts = calculate_word_count(language_word_count)
        # Round the values to the nearest integer
        language_word_counts_rounded = {lang: round(count) for lang, count in language_word_counts.items()}
        # Save the result
        with open(filename, 'w') as file:
            json.dump(language_word_counts_rounded, file)

    # Combine languages with a percentage below 1 into "other"
    threshold = 0.01
    total_word_count = sum(language_word_counts_rounded.values())
    language_word_counts_combined = {'other': 0}
    for lang, count in language_word_counts_rounded.items():
        if count / total_word_count < threshold:
            language_word_counts_combined['other'] += count
        else:
            language_word_counts_combined[lang] = count

    # Print total word count for each language
    print("Total word count for each language:")
    for lang, count in language_word_counts_combined.items():
        print(f"{lang}: {count}")

    # Create labels and values for the pie chart
    labels = list(language_word_counts_combined.keys())
    values = list(language_word_counts_combined.values())

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title='Total word count for each language')

    # # Create pie chart using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # colors = ['#1f77b4', '#aec7e8', '#7b97c7', '#386cb0', '#ff7f0e', '#ffbb78', '#ff9896', '#d62728', '#9467bd',
    #           '#8c564b']
    #
    # plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    # plt.title('Total word count for each language')
    # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    # plt.savefig(f'total_word.png', dpi=300)
    # plt.show()
    return fig


def n_gram_calc(df, n):
    filename = f"{n}_ngram_result.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            top_nrams = json.load(file)
    else:
        top_nrams = get_top_ngram(df["blog_post"], n=n)
        top_nrams = [(item[0], int(item[1])) for item in top_nrams]

        with open(filename, 'w') as file:
            json.dump(top_nrams, file)
    top_n = 30
    top_nrams = top_nrams[:top_n]
    x, y = map(list, zip(*top_nrams))
    # Reverse the order of the data
    x = x[::-1]
    y = y[::-1]
    # Create a bar plot using Plotly
    fig = go.Figure(data=[go.Bar(x=y, y=x, orientation='h')])
    fig.update_layout(
        title='Bar Plot',
        xaxis_title='Count',
        yaxis_title='Category',
        yaxis_tickfont=dict(size=8)  # Adjust the font size as needed
    )
    #fig.show()

    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(x, y)
    # plt.title('Bar Plot')
    # plt.xlabel('Count')
    # plt.ylabel('Category')
    # plt.savefig(f'n_gram.png', dpi=300)
    # plt.show()
    return fig


def polarity_score_calc(df):

    filename = f"polarity_score.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        #df['polarity_score'] = df['blog_post'].apply(lambda x: polarity(x))

        # Applying polarity calculation row-wise in parallel
        df['polarity_score'] = parallelize_dataframe(df, compute_polarity)
        df.to_csv(filename, index=False)


    # Calculate polarity_score using polarity function

    # Define the number of bins
    num_bins = 10

    # Calculate bin size dynamically
    bin_size = 2 / num_bins

    # Create a histogram trace
    # Create a histogram trace
    trace_hist = go.Histogram(x=df['polarity_score'],
                              xbins=dict(start=-1, end=1, size=bin_size),
                              hoverinfo='y+text',  # Show count and custom text on hover
                              #hovertemplate='Bin Range: %{x:.2f} -%{x+x:.2f} <br>Count: %{y}')
                              hovertemplate='Count: %{y}')

    # Create layout for histogram
    layout_hist = go.Layout(title='Polarity Score Distribution',
                            xaxis=dict(title='Polarity Score'),
                            yaxis=dict(title='Count'))

    # Create histogram figure
    fig = go.Figure(data=[trace_hist], layout=layout_hist)

    # Show the histogram
    # fig_hist.show()
    #
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.hist(df['polarity_score'], bins=int((1 - (-1)) / bin_size), range=(-1, 1), edgecolor='black')
    # plt.title('Polarity Score Distribution')
    # plt.xlabel('Polarity Score')
    # plt.ylabel('Count')
    # plt.grid(True)
    # plt.savefig(f'polarity_score.png', dpi=300)
    # plt.show()

    # Calculate polarity using sentiment function
    df['polarity'] = df['polarity_score'].map(lambda x: sentiment(x))
    # Calculate the percentage of each polarity
    polarity_percentages = df['polarity'].value_counts()#(normalize=True) * 100

    # Create a bar plot
    trace_bar = go.Bar(x=polarity_percentages.index,
                       y=polarity_percentages.values)

    # Create layout for bar plot
    layout_bar = go.Layout(title='Polarity Distribution',
                           xaxis=dict(title='Polarity'),
                           yaxis=dict(title='Count'))

    # Create bar plot figure
    fig_simp = go.Figure(data=[trace_bar], layout=layout_bar)

    # Show the bar plot
    #fig_simp.show()

    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.bar(polarity_percentages.index, polarity_percentages.values)
    # plt.title('Polarity Distribution')
    # plt.xlabel('Polarity')
    # plt.ylabel('Count')
    #
    # plt.grid(True)
    # plt.savefig(f'polarity.png', dpi=300)
    # plt.show()
    return fig, fig_simp






def analysis_polarity_score_calc(df):
    filename = f"evaluation_polarity_score.csv"
    if os.path.exists(filename):
        df_pol = pd.read_csv(filename)
    else:
        # df['polarity_score'] = df['blog_post'].apply(lambda x: polarity(x))

        # Applying polarity calculation row-wise in parallel
        df_pol = copy(df)
        df_pol['blog_post'] = df['blog_post'].apply(lambda x: ' '.join(x))
        df_pol['polarity_score'] = parallelize_dataframe(df_pol, compute_polarity)
        df_pol.to_csv(filename, index=False)

        # Calculate polarity_score using polarity function

        # Define the number of bins
    num_bins = 10

    # Calculate bin size dynamically
    bin_size = 2 / num_bins

    # Create a histogram trace
    # Create a histogram trace
    trace_hist = go.Histogram(x=df_pol['polarity_score'],
                              xbins=dict(start=-1, end=1, size=bin_size),
                              hoverinfo='y+text',  # Show count and custom text on hover
                              # hovertemplate='Bin Range: %{x:.2f} -%{x+x:.2f} <br>Count: %{y}')
                              hovertemplate='Count: %{y}')

    # Create layout for histogram
    layout_hist = go.Layout(title='Polarity Score Distribution',
                            xaxis=dict(title='Polarity Score'),
                            yaxis=dict(title='Count'))

    # Create histogram figure
    fig = go.Figure(data=[trace_hist], layout=layout_hist)

    # Show the histogram
    # fig_hist.show()
    #
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.hist(df_pol['polarity_score'], bins=int((1 - (-1)) / bin_size), range=(-1, 1), edgecolor='black')
    # plt.title('Polarity Score Distribution')
    # plt.xlabel('Polarity Score')
    # plt.ylabel('Count')
    # plt.grid(True)
    # plt.savefig(f'evaluation_polarity_score.png', dpi=300)
    # plt.show()

    # Calculate polarity using sentiment function
    df_pol['polarity'] = df_pol['polarity_score'].map(lambda x: sentiment(x))
    # Calculate the percentage of each polarity
    polarity_percentages = df_pol['polarity'].value_counts()  # (normalize=True) * 100

    # Create a bar plot
    trace_bar = go.Bar(x=polarity_percentages.index,
                       y=polarity_percentages.values)

    # Create layout for bar plot
    layout_bar = go.Layout(title='Polarity Distribution',
                           xaxis=dict(title='Polarity'),
                           yaxis=dict(title='Count'))

    # Create bar plot figure
    fig_simp = go.Figure(data=[trace_bar], layout=layout_bar)

    # Show the bar plot
    #fig_simp.show()
    #
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.bar(polarity_percentages.index, polarity_percentages.values)
    # plt.title('Polarity Distribution')
    # plt.xlabel('Polarity')
    # plt.ylabel('Count')
    #
    # plt.grid(True)
    # plt.savefig(f'evaluation_polarity.png', dpi=300)
    # plt.show()


    return fig, fig_simp


def data_initial_statistics(df):

    mean_val, median_val, std_val = calculate_word_count_statistics(df, 'blog_post')
    print("Mean Word Count:", mean_val)
    print("Median Word Count:", median_val)
    print("Standard Deviation of Word Count:", std_val)


    ##################################

    matching_df = filter_dataframe(df, 'blog_post', "Öl")


    ##################################




    hist_data, bins = word_count_histogram(df, 'blog_post')
    fig = go.Figure(data=[go.Bar(x=bins, y=hist_data)])
    fig.update_layout(title='Word Count Histogram of Each Post',
                       xaxis_title='Number of Words',
                       yaxis_title='Frequency')
    # fig.show()
    #
    # # Assuming you have already defined bins and hist_data
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # #plt.hist(hist_data, bins=bins)
    # plt.hist(bins[:-1], bins, weights=hist_data)
    #
    # plt.title('Word Count Histogram of Each Post')
    # plt.xlabel('Number of Words')
    # plt.ylabel('Frequency')
    # # fig.subplots_adjust(left=0.15, top=0.95)
    # plt.savefig(f'word_count_histogram.png', dpi=300)
    # plt.show()





    ##################################

    fig = word_count_violin_plot(df, 'blog_post')
    #fig.show()

    # Count the number of words in each row of the specified column
    #word_counts = df["blog_post"].str.split().apply(len)

    # Create a DataFrame for Matplotlib
    #data = pd.DataFrame({'Word Count': word_counts})

    # Create violin plot using Matplotlib
    # plt.subplots(figsize=(4, 6), layout='constrained')
    # plt.violinplot(data['Word Count'], showmeans=True, showextrema=True)
    # plt.title('Word Count of Each Post Violin Plot')
    # plt.xlabel('Posts')
    # plt.ylabel('Number of Words')
    #
    # plt.savefig(f'word_count_violin_plot.png', dpi=300)
    # plt.show()


    ##################################
    fig = language_word_counts_rounded_calc(df)

    #fig.show()


    ##################################

    fig = n_gram_calc(df,1)
    fig = n_gram_calc(df,2)
    fig = n_gram_calc(df,3)
    #fig.show()


    ##################################

    fig,fig_simp = polarity_score_calc(df)
    #fig.show()
    #fig_simp()


# Define a function to lemmatize a list of strings using spaCy
def lemmatize_text(text_list):
    # Initialize an empty list to store the lemmatized tokens
    lemmatized_tokens = []

    nlp = spacy.load('de_core_news_md')
    # Iterate through each string in the list
    for text in text_list:
        # Process the text using spaCy
        doc = nlp(text)

        # Lemmatize each token in the processed text and append to the lemmatized_tokens list
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        lemmatized_tokens.append(lemmatized_text)

    return lemmatized_tokens


# Define a function to parse a string representation of a list back into a list
def parse_string_to_list(string_repr):
    return ast.literal_eval(string_repr)

    # Tried multiple parts of speech and obtained best topic results using Nouns and Adjectives!

def compute_lemmatize_text(row):
    index, data = row
    return lemmatize_text(data['blog_post'])


def remove_stopwords_german(df):
    stop_words_de = set(stopwords.words('german'))

    dict_file = "removed_german_stopwords.pkl"
    if os.path.exists(dict_file):
        # Load the dictionary from file
        with open(dict_file, "rb") as f:
            sorted_dic = pickle.load(f)
    else:
        # If the dictionary file doesn't exist, create it
        new = df['blog_post'].values.tolist()
        corpus = [word for i in new for word in i]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in stop_words_de:
                dic[word] += 1

        # Sort the dictionary by value (frequency) in descending order
        sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))

        # Save the dictionary to file
        with open(dict_file, "wb") as f:
            pickle.dump(sorted_dic, f)

    # Select the top x stopwords
    top_x = 10  # You can change this value to plot more or fewer stopwords
    top_stopwords = dict(list(sorted_dic.items())[:top_x])

    # Extract stopwords and their frequencies
    stopwords_list = list(top_stopwords.keys())[::-1]
    frequencies = list(top_stopwords.values())[::-1]

    # Create a bar plot
    fig_stopword_de = go.Figure(data=[go.Bar(x=frequencies, y=stopwords_list, orientation='h')])

    # Customize layout
    fig_stopword_de.update_layout(
        title='Top {} Stopwords'.format(top_x),
        xaxis=dict(title='Stopwords'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    #
    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(stopwords_list, frequencies)
    # plt.title('Top {} Stopwords'.format(top_x))
    # plt.xlabel('Frequency')
    # plt.ylabel('Stopwords')
    # # plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    #
    # plt.savefig(f'stopword_de.png', dpi=300)
    # plt.show()

    ######## plot remaining words:
    counter_file = "remaining_words_after_german_stopwords.pkl"
    if os.path.exists(counter_file):
        # Load the Counter from file
        with open(counter_file, "rb") as f:
            counter = pickle.load(f)
    else:
        # If the Counter file doesn't exist, create it
        new = df['blog_post'].values.tolist()

        corpus = [word for sublist in new for word in sublist if isinstance(word, str)]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in stop_words_de:
                dic[word] += 1

        counter = Counter(corpus)

        # Save the Counter to file
        with open(counter_file, "wb") as f:
            pickle.dump(counter, f)
    most = counter.most_common()

    top_n_words = 10
    x, y = [], []
    for word, count in most:
        if (word not in stop_words_de):
            x.append(word)
            y.append(count)

    # x = x[::-1]
    # y = y[::-1]

    # Create a bar plot
    fig_remaining_words_after_de = go.Figure(data=[go.Bar(x=y[:top_n_words][::-1], y=x[:top_n_words][::-1], orientation='h')])

    # Customize layout
    fig_remaining_words_after_de.update_layout(
        title='Top {} Non-Stop'.format(top_x),
        xaxis=dict(title='Non-Stopwords'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(x[:top_n_words][::-1], y[:top_n_words][::-1])
    # plt.title('Top {} Non-Stop'.format(top_n_words))
    # plt.xlabel('Non-Stopwords')
    # plt.ylabel('Frequency')
    # # plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    # plt.savefig(f'remaining_words_after_german_stopwords.png', dpi=300)
    # plt.show()

    df['blog_post'] = df['blog_post'].apply(lambda x: [word for word in x if word not in stop_words_de])

    return fig_stopword_de,fig_remaining_words_after_de,df


def remove_stopwords_english(df):

    # Remove stopwords
    stop_words_en = set(stopwords.words('english'))

    dict_file = "removed_en_stopwords.pkl"
    if os.path.exists(dict_file):
        # Load the dictionary from file
        with open(dict_file, "rb") as f:
            sorted_dic = pickle.load(f)
    else:
        # If the dictionary file doesn't exist, create it
        new = df['blog_post'].values.tolist()
        corpus = [word for i in new for word in i]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in stop_words_en:
                dic[word] += 1

        # Sort the dictionary by value (frequency) in descending order
        sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))

        # Save the dictionary to file
        with open(dict_file, "wb") as f:
            pickle.dump(sorted_dic, f)

    # Select the top x stopwords
    top_x = 10  # You can change this value to plot more or fewer stopwords
    top_stopwords = dict(list(sorted_dic.items())[:top_x])

    # Extract stopwords and their frequencies
    stopwords_list = list(top_stopwords.keys())[::-1]
    frequencies = list(top_stopwords.values())[::-1]

    # Create a bar plot
    fig_stopword_en = go.Figure(data=[go.Bar(x=frequencies, y=stopwords_list, orientation='h')])

    # Customize layout
    fig_stopword_en.update_layout(
        title='Top {} Stopwords'.format(top_x),
        xaxis=dict(title='Stopwords'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(stopwords_list, frequencies)
    # plt.title('Top {} Stopwords'.format(top_x))
    # plt.xlabel('Frequency')
    # plt.ylabel('Stopwords')
    # #plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    #
    # plt.savefig(f'stopword_en.png', dpi=300)
    # plt.show()




    ######## plot remaining words:
    counter_file = "remaining_words_after_en_stopwords.pkl"
    if os.path.exists(counter_file):
        # Load the Counter from file
        with open(counter_file, "rb") as f:
            counter = pickle.load(f)
    else:
        # If the Counter file doesn't exist, create it
        new = df['blog_post'].values.tolist()

        corpus = [word for sublist in new for word in sublist if isinstance(word, str)]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in stop_words_en:
                dic[word] += 1

        counter = Counter(corpus)

        # Save the Counter to file
        with open(counter_file, "wb") as f:
            pickle.dump(counter, f)
    most = counter.most_common()

    top_n_words = 10
    x, y = [], []
    for word, count in most:
        if (word not in stop_words_en):
            x.append(word)
            y.append(count)


    # Create a bar plot
    fig_remaining_words_after_en = go.Figure(data=[go.Bar(x=y[:top_n_words][::-1], y=x[:top_n_words][::-1], orientation='h')])

    # Customize layout
    fig_remaining_words_after_en.update_layout(
        title='Top {} Non-Stop'.format(top_x),
        xaxis=dict(title='Non-Stopwords'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(x[:top_n_words][::-1], y[:top_n_words][::-1])
    # plt.title('Top {} Non-Stop'.format(top_n_words))
    # plt.xlabel('Non-Stopwords')
    # plt.ylabel('Frequency')
    # #plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    # plt.savefig(f'remaining_words_after_en_stopwords.png', dpi=300)
    # plt.show()


    df['blog_post'] = df['blog_post'].apply(lambda x: [word for word in x if word not in stop_words_en])


    return fig_stopword_en,fig_remaining_words_after_en,df


def remove_modalpartikeln_german(df):
    # Remove modalpartikeln
    modalpartikeln = [
        "ja", "doch", "eben", "halt", "denn", "wohl", "mal", "schon", "bloß", "auch",
        "nur", "eigentlich", "etwa", "immerhin", "ebenso", "wirklich", "immer", "aber",
        "gerade", "ruhig", "jedoch", "genau", "dann", "zumindest", "wenigstens",
        "wahrscheinlich", "manchmal", "vielleicht", "schließlich", "tatsächlich",
        "ungefähr", "gar", "sowieso", "keineswegs", "möglicherweise", "hoffentlich",
        "außerdem", "irgendwie", "sogar", "fast", "lediglich", "mal", "meistens",
        "nun", "scheinbar", "bisweilen", "freilich", "selten", "durchaus", "generell",
        "letztlich", "endlich", "offenbar", "jedenfalls", "zunächst", "beinahe", "also",
        "geradezu", "allenfalls", "gewiss", "definitiv", "kaum", "bislang", "notfalls",
        "umso", "allemal", "schier", "bestenfalls", "vorerst", "einigermaßen",
        "glücklicherweise", "nachweislich", "vor allem", "inzwischen", "unter Umständen"
    ]
    modal_particle_word_de = set(modalpartikeln)

    dict_file = "modal_particle_word_de.pkl"
    if os.path.exists(dict_file):
        # Load the dictionary from file
        with open(dict_file, "rb") as f:
            sorted_dic = pickle.load(f)
    else:
        # If the dictionary file doesn't exist, create it
        new = df['blog_post'].values.tolist()
        corpus = [word for i in new for word in i]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in modal_particle_word_de:
                dic[word] += 1

        # Sort the dictionary by value (frequency) in descending order
        sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))

        # Save the dictionary to file
        with open(dict_file, "wb") as f:
            pickle.dump(sorted_dic, f)

    # Select the top x stopwords
    top_x = 10  # You can change this value to plot more or fewer stopwords
    top_stopwords = dict(list(sorted_dic.items())[:top_x])

    # Extract stopwords and their frequencies
    stopwords_list = list(top_stopwords.keys())[::-1]
    frequencies = list(top_stopwords.values())[::-1]

    # Create a bar plot
    fig_modal_de = go.Figure(data=[go.Bar(x=frequencies, y=stopwords_list, orientation='h')])

    # Customize layout
    fig_modal_de.update_layout(
        title='Top {} Modal Particle'.format(top_x),
        xaxis=dict(title='Modal Particle'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(stopwords_list, frequencies)
    # plt.title('Top {} Modal Particle'.format(top_x))
    # plt.xlabel('Frequency')
    # plt.ylabel('Modal Particle')
    # # plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    #
    # plt.savefig(f'modal_particle_word_de.png', dpi=300)
    # plt.show()

    ######## plot remaining words:
    counter_file = "remaining_words_after_modal_particle_word_de.pkl"
    if os.path.exists(counter_file):
        # Load the Counter from file
        with open(counter_file, "rb") as f:
            counter = pickle.load(f)
    else:
        # If the Counter file doesn't exist, create it
        new = df['blog_post'].values.tolist()

        corpus = [word for sublist in new for word in sublist if isinstance(word, str)]

        # Assuming stop_words_de is defined somewhere
        dic = defaultdict(int)
        for word in corpus:
            if word in modal_particle_word_de:
                dic[word] += 1

        counter = Counter(corpus)

        # Save the Counter to file
        with open(counter_file, "wb") as f:
            pickle.dump(counter, f)
    most = counter.most_common()

    top_n_words = 10
    x, y = [], []
    for word, count in most:
        if (word not in modal_particle_word_de):
            x.append(word)
            y.append(count)

    # Create a bar plot
    fig_remaining_words_after_modal_de = go.Figure(data=[go.Bar(x=y[:top_n_words][::-1], y=x[:top_n_words][::-1], orientation='h')])

    # Customize layout
    fig_remaining_words_after_modal_de.update_layout(
        title='Top {} Non-Modal-Particle'.format(top_x),
        xaxis=dict(title='Non-Modal-Particle'),
        yaxis=dict(title='Frequency')
    )

    # Show the plot
    # fig.show()
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(x[:top_n_words][::-1], y[:top_n_words][::-1])
    # plt.title('Top {} Non-Stop'.format(top_n_words))
    # plt.xlabel('Non-Modal-Particle')
    # plt.ylabel('Frequency')
    # # plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at the top
    # plt.savefig(f'remaining_words_after_modal_particle_word_de.png', dpi=300)
    # plt.show()

    df['blog_post'] = df['blog_post'].apply(lambda x: [word for word in x if word not in modal_particle_word_de])

    return fig_modal_de, fig_remaining_words_after_modal_de, df


def preprocess(df):
    # Remove URLs
    filename = 'removed_urls.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df['blog_post'] = df['blog_post'].apply(lambda x: re.sub(r'http\S+', '', x))
        df.to_csv(filename, index=False)



    # Remove non-alphanumeric characters

    filename = 'removed_non_alphanumeric.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        # pattern = r'[^a-zA-Z0-9ßäöüÄÖÜẞ\-\s]'
        pattern = r'[^a-zA-ZßäöüÄÖÜẞ\s]'
        df['blog_post'] = df['blog_post'].apply(lambda x: re.sub(pattern, ' ', x))
        df.to_csv(filename, index=False)


    # Convert text to lowercase
    filename = 'lowercase.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df['blog_post'] = df['blog_post'].apply(lambda x: x.lower())
        df.to_csv(filename, index=False)


    # Tokenize the text

    filename = 'tokenize.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
    else:
        df['blog_post'] = df['blog_post'].apply(lambda x: word_tokenize(x))
        df.to_csv(filename, index=False)


    # Remove stopwords german

    fig_stopword_de,fig_remaining_words_after_de,df =  remove_stopwords_german(df)



    ###
    fig_stopword_en, fig_remaining_words_after_en, df = remove_stopwords_english(df)



    ###

    fig_modal_de, fig_remaining_words_after_modal_de, df = remove_modalpartikeln_german(df)
  





    filename = 'after_removing_unwated_words.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
    else:
        df.to_csv(filename, index=False)



    # Lemmatize the tokens ENGLISH
    filename = 'lemmatize_english.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)


        # Apply the parsing function to each element in the DataFrame column
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
    else:
        lemmatizer = WordNetLemmatizer()
        df['blog_post'] = df['blog_post'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        df.to_csv(filename, index=False)

    # LEMMATIZE GERMAN:
    filename = 'lemmatize_german.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
    else:

        df['blog_post'] = df['blog_post'].apply(lambda x: ' '.join(x))
        # Load spaCy pipeline
        nlp = spacy.load('de_core_news_md')


        lemma_text_list = []
        for doc in tqdm(nlp.pipe(df["blog_post"]), total=len(df)):
            lemma_text_list.append(" ".join(token.lemma_ for token in doc))

        df['blog_post'] = lemma_text_list

        df['blog_post'] = df['blog_post'].apply(lambda x: x.lower())
        df['blog_post'] = df['blog_post'].apply(lambda x: word_tokenize(x))
        df.to_csv(filename, index=False)
    print('done')
    return df


def analysis_n_gram_calc(df, n):
    filename = f"evaluation_{n}_ngram_result.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            top_nrams = json.load(file)
    else:
        comb = df['blog_post'].apply(lambda x: ' '.join(x))
        top_nrams = get_top_ngram(comb, n=n)
        top_nrams = [(item[0], int(item[1])) for item in top_nrams]

        with open(filename, 'w') as file:
            json.dump(top_nrams, file)
    top_n = 30
    top_nrams = top_nrams[:top_n]
    x, y = map(list, zip(*top_nrams))
    # Reverse the order of the data
    x = x[::-1]
    y = y[::-1]
    # Create a bar plot using Plotly
    fig = go.Figure(data=[go.Bar(x=y, y=x, orientation='h')])
    fig.update_layout(
        title='Bar Plot',
        xaxis_title='Count',
        yaxis_title='Category',
        yaxis_tickfont=dict(size=8)  # Adjust the font size as needed
    )
    # fig.show()
    #
    # # Create horizontal bar plot using Matplotlib
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # plt.barh(x, y)
    # plt.title('Bar Plot')
    # plt.xlabel('Count')
    # plt.ylabel('Category')
    # plt.savefig(f'evaluation_n_gram.png', dpi=300)
    # plt.show()
    return fig


def evaluation(df):
    # word_counts = df['blog_post'].apply(len)
    # hist_data, bins = np.histogram(word_counts, bins='auto')
    # fig = go.Figure(data=[go.Bar(x=bins, y=hist_data)])
    # fig.update_layout(title='Word Count Histogram of Each Post',
    #                   xaxis_title='Number of Words',
    #                   yaxis_title='Frequency')
    # fig.show()
    #
    # # Assuming you have already defined bins and hist_data
    # plt.subplots(figsize=(6, 4), layout='constrained')
    # # plt.hist(hist_data, bins=bins)
    # plt.hist(bins[:-1], bins, weights=hist_data)
    #
    # plt.title('Word Count Histogram of Each Post')
    # plt.xlabel('Number of Words')
    # plt.ylabel('Frequency')
    # # fig.subplots_adjust(left=0.15, top=0.95)
    # plt.savefig(f'evaluation_word_count_histogram.png', dpi=300)
    # plt.show()
    #
    #
    #
    # word_counts = df['blog_post'].apply(len)
    #
    # # Create a DataFrame for Plotly
    # data = pd.DataFrame({'Word Count': word_counts})
    #
    # # Create a violin plot using Plotly
    # fig = px.violin(data, y='Word Count', box=True, points='all')
    # fig.update_layout(title='Word Count of Each Post Violin Plot',
    #                   yaxis_title='Number of Words')
    # fig.show()
    #
    # # Count the number of words in each row of the specified column
    # word_counts = df["blog_post"].apply(len)
    #
    # # Create a DataFrame for Matplotlib
    # data = pd.DataFrame({'Word Count': word_counts})
    #
    # # Create violin plot using Matplotlib
    # plt.subplots(figsize=(4, 6), layout='constrained')
    # plt.violinplot(data['Word Count'], showmeans=True, showextrema=True)
    # plt.title('Word Count of Each Post Violin Plot')
    # plt.xlabel('Posts')
    # plt.ylabel('Number of Words')
    #
    # plt.savefig(f'evaluation_word_count_violin_plot.png', dpi=300)
    # plt.show()


    fig = analysis_n_gram_calc(copy(df),1)
    fig = analysis_n_gram_calc(copy(df),2)
    fig = analysis_n_gram_calc(copy(df),3)


    ##################################

    fig = analysis_polarity_score_calc(copy(df))








    count_vect = CountVectorizer()
    comb = df['only_adj_noun_propn'].apply(lambda x: ' '.join(x))
    bow = count_vect.fit_transform(comb.values)
    terms = count_vect.get_feature_names_out()
    print(terms[1:10])
    n_cluster = 10
    model = KMeans(n_clusters=n_cluster, init='k-means++', random_state=42)
    model.fit(bow)
    labels = model.labels_
    cluster_center = model.cluster_centers_
    silhouette_score = metrics.silhouette_score(bow, labels, metric='euclidean')
    df['labels'] = model.labels_  # the last column you can see the label numebers
    df.head(2)
    print(df.groupby(['labels']).count())
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names_out()
    for i in range(n_cluster):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :n_cluster]:
            print(' %s' % terms[ind], end='')
            print()
    plt.subplots(figsize=(6, 4), layout='constrained')
    plt.bar([x for x in range(n_cluster)], df.groupby(['labels'])['only_adj_noun_propn'].count(), alpha=0.4)
    plt.title('KMeans cluster points')
    plt.xlabel("Cluster number")
    plt.ylabel("Number of points")
    plt.savefig(f'CountVectorizer_kmeans_only_adj_noun_propn.png', dpi=300)
    plt.show()


    dic = corpora.Dictionary(df['only_adj_noun_propn'])
    bow_corpus = [dic.doc2bow(doc) for doc in df['only_adj_noun_propn']]

    lda_model = models.LdaMulticore(bow_corpus,
                                    random_state=42,
                                           num_topics=10,
                                           id2word=dic,
                                            chunksize=20,
                                           passes=10,
                                    iterations=40,
                                           workers=5)


    print(lda_model.show_topics())

    #pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    x=1


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

    # LEMMATIZE GERMAN:
    #df = preprocess(df)
    filename = 'final_preprocessed.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
    else:
        df = preprocess(df)
        df.to_csv(filename, index=False)

    comb = df['blog_post'].apply(lambda x: ' '.join(x))

    #df=df[:10]

    #tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')
    #words = nltk.word_tokenize("Jahr ja wir es und mal schon")
    #print(tagger_de.tag_sent(words))
    #tokens = [word for (word, x, pos) in tagger_de.tag_sent(words, taglevel=1) if pos == 'NN']

    filename = 'tagged.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['blog_post'] = df['blog_post'].apply(parse_string_to_list)
        df['only_adj_noun_propn'] = df['only_adj_noun_propn'].apply(parse_string_to_list)
    else:
        nlp = spacy.load('de_core_news_md')

        pos_text_list = []
        for doc in tqdm(nlp.pipe(comb), total=len(comb)):
            pos_text_list.append(" ".join(token.pos_ for token in doc))

        df['tags'] = pos_text_list
        df['tags'] = df['tags'].apply(lambda x: word_tokenize(x))
        # Zip the elements of the lists together element-wise
        df['zipped_column'] = [[(blog, tag) for blog, tag in zip(blog_post, tags)]
                               for blog_post, tags in zip(df['blog_post'], df['tags'])]

        allowed_values = {'NOUN', 'ADV', 'PROPN'}
        # only_adj_noun_propn = [[(x, y) for x, y in row if y in allowed_values] for row in df['zipped_column']]
        only_adj_noun_propn = [[x for x, y in row if y in allowed_values] for row in df['zipped_column']]
        df['only_adj_noun_propn'] = only_adj_noun_propn
        # all_adj_nouns = [word for (word, tag) in df['zipped_column'] if (tag == "NOUN" or tag == "ADV" or tag == "PROPN")]
        df.to_csv(filename, index=False)


    evaluation(df)





if __name__ == '__main__':
    main()
