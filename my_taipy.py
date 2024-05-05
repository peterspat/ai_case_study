from taipy.gui import Gui, navigate, Icon, notify

from src.pages.analysis_md import analysis_page
from src.pages.data_viewer_md import data_page
from src.pages.home_md import home_page
from src.pages.info_md import info_page
from src.pages.page1_md import page1
from src.pages.page2_md import page2
from src.pages.preprocessing_md import preprocessing_page

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
         "page2": page2
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
