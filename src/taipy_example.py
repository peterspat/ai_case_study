### Carry over variables from previous sections
import pandas as pd
import taipy as tp
from taipy.gui import Gui

penguin_file_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguin_df = pd.read_csv(penguin_file_url)
###

target_names = list(penguin_df.species.unique())  # ["Adelie", "Gentoo", "Chinstrap"]
species = target_names[0]  # "Adelie"

penguin_species_selector = """
### Penguin species: 
<|{species}|selector|lov={target_names}|dropdown=True|width=100%|>
"""

if __name__ == "__main__":
    #tp.Core().run()
    print('yay')
    Gui(page=penguin_species_selector).run(title="AI CASE STUDY",
                                           host="0.0.0.0",
                                           port=5000,
                                           dark_mode=True,
                                           #debug=True,
                                           #use_reloader=True
                                           )


    print('yay2')
