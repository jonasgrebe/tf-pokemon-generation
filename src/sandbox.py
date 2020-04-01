import numpy as np
import os

from utils.images import save_single_image, load_single_image
from dcgan import DCGAN

models = os.listdir('../results')
models = [m for m in models if '_old' not in m]

sample_interval = 5

medium_dir = '../medium'
if not os.path.isdir(medium_dir):
    os.mkdir(medium_dir)

seeds = None
for model in models:
    gan = DCGAN(name=model, reload=True)
    
    gan.load_weights(250)
    gan.plot_history(sample_interval=sample_interval)
    
    if seeds is None:
        seeds = gan.generate_random_noise(8)
    
    save_dir = os.path.join(medium_dir, model)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    images = []
    for e in range(0, 251, sample_interval):
        gan.load_weights(e)
        
        samples = gan.sample(seeds)
        
        block = np.concatenate(samples, axis=1)
        save_single_image(os.path.join(save_dir, f'{e}.png'), block)
        
       # block[np.where(block[:,:,-1] < 10)] = 255
        
        images.append(block)
        
    import imageio
    imageio.mimsave(os.path.join(save_dir, 'anim.gif'), images)
    
