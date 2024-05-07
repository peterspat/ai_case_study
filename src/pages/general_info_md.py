
general_info = """
<|navbar|lov={[("data-exploration","Start"),("/general_info", "General Info"), ("/n_gram", "N-Gram"), ("/polarity", "Polarity")]}|>
# General Information about Data
## Word Count of Rows in Data

<|chart|figure={word_count_histogram}|height=70vh|>

<|chart|figure={word_count_violine}|height=70vh|>

Mean: <|{mean_val}|>

Median: <|{median_val}|>

Std: <|{std_val}|>



## Language Distribution of Data
<|chart|figure={language_pie}|height=70vh|>



"""