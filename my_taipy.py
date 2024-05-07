import json
import os

import pandas as pd
from taipy.gui import Gui, navigate, Icon, notify

from main import filter_dataframe, calculate_word_count_statistics, word_count_histogram, parallelize_series_processing, \
    detect_and_count_words, calculate_word_count, polarity, sentiment, get_top_ngram, language_word_counts_rounded_calc, \
    word_count_violin_plot, n_gram_calc, polarity_score_calc
from src.pages.analysis_md import analysis_page
from src.pages.data_viewer_md import data_page
from src.pages.general_info_md import general_info
from src.pages.home_md import home_page
from src.pages.info_md import info_page
from src.pages.n_gram_md import n_gram
from src.pages.page1_md import page1
from src.pages.page2_md import page2
from src.pages.polarity_md import polarity_page
from src.pages.preprocessing_md import preprocessing_page


import plotly.graph_objects as go
import plotly.express as px



def get_table():
    # Assuming your CSV file is named 'example.csv'
    csv_file_path = r'data\posts.csv'

    # Read only the first 10 columns and a specific number of rows from the CSV file into a pandas DataFrame
    num_rows=3
    df = pd.read_csv(csv_file_path, header=None, names=["blog_post"],  nrows=num_rows)
    df.dropna(inplace=True)
    return df

table = get_table()
table_full = pd.read_csv( r'data\posts.csv', header=None, names=["blog_post"])
table_full.dropna(inplace=True)



text = "Öl"

table_search = filter_dataframe(table_full, 'blog_post', "Öl")
def on_button_action(state):
    if state.text != '':
        notify(state, 'info', f'Search Text is: {state.text}')
        state.table_search = filter_dataframe(table_full, 'blog_post', state.text)

def on_change(state, var_name, var_value):
    if var_name == "text" and var_value == "Reset":
        state.text = ""
        return





mean_val, median_val, std_val = calculate_word_count_statistics(table_full, 'blog_post')

hist_data, bins = word_count_histogram(table_full, 'blog_post')
word_count_histogram = go.Figure(data=[go.Bar(x=bins, y=hist_data)])
word_count_histogram.update_layout(title='Word Count Histogram of Each Post',
                   xaxis_title='Number of Words',
                   yaxis_title='Frequency')



language_pie = language_word_counts_rounded_calc(table_full)


# Create a violin plot using Plotly
word_count_violine = word_count_violin_plot(table_full, 'blog_post')


polarity_histogram,polarity_histogram_simplified = polarity_score_calc(table_full)



slider_ngram_value = 1

histogram_ngram =  n_gram_calc(table_full, slider_ngram_value)





def on_slider_ngram(state):
    state.histogram_ngram =  n_gram_calc(table_full,state.slider_ngram_value)


# =======================
#       Setup menu
# =======================
menu = [("home", Icon('src/assets/home.png', 'Home')),
        ('data-exploration', Icon('src/assets/database.png', 'Data Exploration')),
        ("preprocessing", Icon('src/assets/work-process.png', 'Preprocessing')),
        ("analysis", Icon('src/assets/data-analysis.png', 'Analysis')),
        ('info', Icon('src/assets/info.png', 'Info')),

        ]

login_open = False
password = ''
page_markdown = """
<|toggle|theme|>
<|menu|label=Menu|lov={menu}|on_action=on_menu|>

<|{login_open}|dialog|title=Login|width=30%|


**Password**
<|{password}|input|password|label=Password|class_name=fullwidth|>


<br/>
<|Sign in|button|class_name=fullwidth plain|on_action=login|>
|>
"""
# https://stackoverflow.com/questions/77745948/is-there-any-routes-mechanism-or-anything-for-replacement-in-taipy

pages = {"/": page_markdown,
         "home": home_page,
         "data-exploration": data_page,
         "preprocessing": preprocessing_page,
         "analysis": analysis_page,
         "info": info_page,
         "page1": page1,
         "page2": page2,
         "general_info": general_info,
         "n_gram": n_gram,
         "polarity": polarity_page,
         }


def login(state):
    # Put your own authentication system here
    if state.password == "vector123":
        state.login_open = False
        notify(state, "success", "Logged in!")
    else:
        notify(state, "error", "Wrong password!")


def on_menu(state, action, info):
    page = info["args"][0]
    navigate(state, to=page)


if __name__ == "__main__":
    # tp.Core().run()

    Gui(pages=pages).run(title="AI CASE STUDY",
                         host="0.0.0.0",
                         port=5000,
                         dark_mode=True,
                         # debug=True,
                         # use_reloader=True
                         )
