
analysis_n_gram_page = """
<|navbar|lov={[("analysis","Start"), ("/analysis_n_gram", "N-Gram"), ("/analysis_polarity", "Polarity")]}|>
# N-Grams of Data
Number of N-Grams:

<|{analysis_slider_ngram_value}|slider|min=1|max=3|on_change=on_slider_ngram_analysis|>


<|chart|figure={analysis_histogram_ngram}|height=70vh|>







"""