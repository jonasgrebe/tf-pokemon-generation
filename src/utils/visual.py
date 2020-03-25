import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array

import numpy as np
import os

from utils.images import load_single_image, save_single_image
from utils.pokemon import get_pokedex_entry

import cv2


def get_pokemon_type_bar(primary, secondary, reverse=False):
   
    primary_icon = load_single_image(f'utils/type_icons/{primary}.png')

    if secondary == 'NA' or secondary is None:
        secondary_icon = np.zeros_like(primary_icon)
    else:
        secondary_icon = load_single_image(f'utils/type_icons/{secondary}.png')

    return np.concatenate([primary_icon, secondary_icon] if not reverse else [secondary_icon, primary_icon], axis=1)


def get_pokemon_shape_icon(shape):
    shape_icon = load_single_image(f'utils/shape_icons/{shape}.png')    
    return shape_icon


def get_pokemon_sprite_by_id(dex_id, data_dir, shiny=False):
    dex_id_files = [file for file in os.listdir(data_dir) if file.startswith(str(dex_id)+'_')]

    if shiny:
        dex_id_files = [file for file in dex_id_files if '_s' in file]
    else:
        dex_id_files = [file for file in dex_id_files if '_s' not in file]

    if len(dex_id_files) < 1:
        raise ValueError('no sprites found')

    img_file = np.random.choice(dex_id_files)
    load_path = os.path.join(data_dir, img_file)
    
    return load_single_image(load_path)


def get_pokemon_block_by_id(dex_id, data_dir, type2=True, shiny=False):
    sprite = get_pokemon_sprite_by_id(dex_id, data_dir, shiny=shiny)
    type_1, type_2 = get_pokedex_entry(dex_id, ['type_1', 'type_2'])
    shape = get_pokedex_entry(dex_id, ['shape'])[0]
    
    return assemble_pokemon_block(sprite, type_1, type_2 if type2 else 'NA', shape)


def assemble_pokemon_block(sprite, type_1, type_2):
    type_bar = get_pokemon_type_bar(type_1, type_2)
    return np.concatenate([type_bar, sprite], axis=0)


def assemble_pokemon_type_predictions(types, scores, k=3):
    idxs = np.argsort(scores)[::-1]
    top_types = [types[idx] for idx in idxs[:k]]
    top_scores = [scores[idx] for idx in idxs[:k]]
    
    type_bars = [get_pokemon_type_bar(type1, 'NA') for type1 in top_types]
    
    for i in range(k):
        score = top_scores[i]
        cv2.putText(type_bars[i], '{:1.3f}'.format(top_scores[i]), (57, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0, 255))
        
    block = np.concatenate(type_bars, axis=0)
    
    return block


def get_pokemon_vis_by_sprite(sprite, type_1, types, type_scores):
    block = assemble_pokemon_block(sprite, type_1, 'NA')
    type_block = assemble_pokemon_type_predictions(types, type_scores, k=4)
    vis = np.concatenate([block, type_block], axis=0)
    return vis


def plot_history(history, y_label, labels, save_path):
    history = np.swapaxes(history, 0, 1)

    plt.clf()
    for h, label in zip(history, labels):
        plt.plot(h, label=label)
    plt.legend()
    plt.xlabel('batches')
    plt.ylabel(y_label)
    plt.savefig(save_path, dpi=256)
    
    