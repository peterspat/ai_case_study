"""
Module Name: Page Markdown Template
Author: Kenneth Leung
Last Modified: 19 Mar 2023
"""

word_standardizing = """
<|navbar|lov={[("preprocessing","Start"),("/text_streamlining", "Streamlining Text"), ("/word_filter", "Word Filtering"), ("/word_standardizing", "Word Standardizing"), ("/word_types", "Word Types")]}|>

# Word Filtering

Before Lemmatizing

<|{before_lemmatize}|table|show_all|>

After Lemmatizing (German and English)


<|{final_lemmatize}|table|show_all|>


"""