
import pandas as pd
from taipy.gui import notify

data_page = """
<|navbar|lov={[("/general_info", "General Info"), ("/n_gram", "N-Gram"), ("/polarity", "Polarity")]}|>
# Data Viewer

Explore the data set of blog.fefe.de.

Showing the first 3 rows.

<|{table}|table|show_all|>


Searching for Text in Data:
<br/>
<|{text}|input|> <|Search|button|on_action=on_button_action|>

<|{table_search}|table|show_all|on_action=on_button_action|>

"""
