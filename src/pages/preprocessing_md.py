"""
Module Name: Page Markdown Template
Author: Kenneth Leung
Last Modified: 19 Mar 2023
"""

preprocessing_page = """
<|navbar|lov={[("preprocessing","Start"),("/text_streamlining", "Streamlining Text"), ("/word_filter", "Word Filtering"), ("/word_standardizing", "Word Standardizing"), ("/word_types", "Word Types")]}|>
# Preprocessing

## Streamlining Text
- **Removing unwanted characters**: This step involves eliminating any characters in the text that are not relevant for analysis or may introduce noise such as removing special characters, punctuation, and symbols.

- **Transition to lowercase**: Converting all the text to lowercase ensures consistency and reduces the complexity of text analysis for preventing prevents treating words with different cases .

- **Tokenize words**: Tokenization involves breaking down the text into individual words or tokens. 

## Word Filtering
- **Remove unwanted words (e.g. stopwords)**: Stopwords are common words that do not carry significant meaning in a given context, such as "the", "and", "is", etc. 

## Word Standardizing
- **Lemmatize to adjust it**: Lemmatization involves reducing words to their base or root form, which helps in standardizing words with similar meanings. For example, "running" and "ran" would both be lemmatized to "run".


## Word Types
- **Assign word types for each word**: This step involves tagging each word with its corresponding part of speech, such as noun, verb, adjective, etc. 
"""