import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array

import numpy as np
import os

from images import load_single_image, save_single_image
from pokemon import get_pokedex_entry

import cv2


def get_pokemon_type_bar(primary, secondary, reverse=False):
   
    primary_icon = load_single_image(f'type_icons/{primary}.png')

    if secondary == 'NA' or secondary is None:
        secondary_icon = np.zeros_like(primary_icon)
    else:
        secondary_icon = load_single_image(f'type_icons/{secondary}.png')

    return np.concatenate([primary_icon, secondary_icon] if not reverse else [secondary_icon, primary_icon], axis=1)


def get_pokemon_shape_icon(shape):
    shape_icon = load_single_image(f'shape_icons/{shape}.png')    
    return shape_icon


def convert_hex_to_rgba(list_of_hex):
    for i, code in enumerate(list_of_hex):
        if code == 'NA':
            list_of_hex[i] = '#FFFFFF'
    return to_rgba_array(list_of_hex)
    

def get_pokemon_color_bar(color_1, color_2='NA', color_f='NA', shape=(16, 96, 4)):
    c1, c2, cf = convert_hex_to_rgba([color_1, color_2, color_f]) * 255.0
    
    if color_f != 'NA':
        c1 = cf

    icons = np.zeros(shape)
    icons[:,:] = c1
    
    return icons


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


def assemble_pokemon_block(sprite, type_1, type_2, shape):
    type_bar = get_pokemon_type_bar(type_1, type_2)
    shape_icon = cv2.resize(get_pokemon_shape_icon(shape), (16, 16), cv2.INTER_NEAREST)
    
    spacing = 2
    bar = np.concatenate([shape_icon, np.zeros((16, spacing, 4)), type_bar], axis=1)
    bar = cv2.resize(bar, (96, 13), cv2.INTER_NEAREST)

    return np.concatenate([bar, sprite], axis=0)


def get_pokemon_block_by_id(dex_id, data_dir, type2=True, shiny=False):
    sprite = get_pokemon_sprite_by_id(dex_id, data_dir, shiny=shiny)
    type_1, type_2 = get_pokedex_entry(dex_id, ['type_1', 'type_2'])
    shape = get_pokedex_entry(dex_id, ['shape'])[0]
    
    return assemble_pokemon_block(sprite, type_1, type_2 if type2 else 'NA', shape)
    

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


def assemble_pokemon_shape_predictions(shapes, scores, k=3):
    
    idxs = np.argsort(scores)[::-1]
    top_shapes = [shapes[idx] for idx in idxs[:k]]
    top_scores = [scores[idx] for idx in idxs[:k]]
    
    shape_icons = [get_pokemon_shape_icon(shape) for shape in top_shapes]
    
    bar = np.concatenate(shape_icons, axis=1)
    plt.imshow(bar)
    plt.show()
    
    return cv2.resize(bar, (96, 96 // k), cv2.INTER_NEAREST)


def get_pokemon_vis_by_sprite(sprite, type_1, type_2, shape, pred_types, pred_shapes, probs):
    block = assemble_pokemon_block(sprite, type_1, type_2, shape)
    type_block = assemble_pokemon_type_predictions(pred_types, probs, k=4)
    shape_bar = assemble_pokemon_shape_predictions(pred_shapes, probs, k=4)
     
    return np.concatenate([block, type_block, shape_bar], axis=0)


DATA_DIR = 'C:/Users/Jonas/Documents/GitHub/pokemon-generation/data/sprites'
types = ['fire', 'NA', 'grass', 'flying']
shapes = ['wings', 'upright', 'quadruped', 'heads']
scores = [0.3131, 0.05123, 0.2123222, 0.0]

sprite = get_pokemon_sprite_by_id(120, DATA_DIR)
vis = get_pokemon_vis_by_sprite(sprite, 'NA', 'NA', 'wings', types, shapes, scores)

plt.imshow(vis / 255)
plt.axis('off')
plt.show()
