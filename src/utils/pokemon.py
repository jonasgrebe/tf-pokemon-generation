import os
import pandas as pd
import numpy as np


pokedex = pd.read_csv('pokemon.txt', na_filter=False)

# drop all pokemon after generation 5
idxs = pokedex[ pokedex['generation_id'] > 5 ].index
pokedex.drop(idxs, inplace=True)


def extract_pokedex_id(img_file_path):
    return int(os.path.basename(img_file_path).split('_')[0].split('-')[0])


def get_pokedex_entry(dex_id, columns):
    return pokedex.loc[dex_id-1, columns]


def get_all_pokedex_categories(column, sort=False):
    categories = pokedex.loc[:,column].unique()
    return list(sorted(categories)) if sort else list(categories)


def get_all_pokedex_ids_by_entry_values(columns, values):
    query = ' & '.join([c + '==' + f'"{v}"' for (c, v) in zip(columns, values)])
    entries = pokedex.query(query)
    return entries['id'].values


