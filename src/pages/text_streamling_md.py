"""
Module Name: Page Markdown Template
Author: Kenneth Leung
Last Modified: 19 Mar 2023
"""

text_streamlining = """
<|navbar|lov={[("preprocessing","Start"),("/text_streamlining", "Streamlining Text"), ("/word_filter", "Word Filtering"), ("/word_standardizing", "Word Standardizing"), ("/word_types", "Word Types")]}|>

# Streamlining Text

<|{table}|table|show_all|>

## Removing URLS and non-alphanumeric characters


<|{table_after_all}|table|show_all|>

"""