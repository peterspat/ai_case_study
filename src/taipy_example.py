### Carry over variables from previous sections
from taipy.gui import Gui
import pandas as pd


penguin_file_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguin_df = pd.read_csv(penguin_file_url)
###

target_names = list(penguin_df.species.unique()) # ["Adelie", "Gentoo", "Chinstrap"]
species = target_names[0] # "Adelie"

penguin_species_selector = """
### Penguin species: 
<|{species}|selector|lov={target_names}|dropdown=True|width=100%|>
"""

Gui(page=penguin_species_selector).run(dark_mode=False)