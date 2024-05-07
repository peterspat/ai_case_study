
analysis_clustering_page = """
<|navbar|lov={[("analysis","Start"), ("/analysis_n_gram", "N-Gram"), ("/analysis_polarity", "Polarity"), ("/analysis_clustering", "Clustering")]}|>

# Clustering

Apply different Vectorization Methods for k-Means Clustering:

<|{sel}|selector|lov={[("CountVectorizer", "CountVectorizer"), ("TfidfVectorizer","TfidfVectorizer")]}|on_change=update_chart|>

<|{word_image_clustering}|image|height=150vh|width=150vh|>



"""