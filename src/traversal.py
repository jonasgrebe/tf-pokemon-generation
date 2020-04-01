import numpy as np
import os
import imageio
    
from utils.images import save_single_image, load_single_image
from dcgan import DCGAN

models = os.listdir('../results')
models = [m for m in models if '_old' not in m]

sample_interval = 5

medium_dir = '../medium'
if not os.path.isdir(medium_dir):
    os.mkdir(medium_dir)


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high
 
start_seed = None
target_seed = None

steps = 199

linear_step_frames = {}
spherical_step_frames = {}

for n, step_frames in zip(['linear', 'spherical'], [linear_step_frames, spherical_step_frames]):
    
    for model in models:
        gan = DCGAN(name=model, reload=True)
        gan.load_weights(250)
        
        if start_seed is None:
            start_seed, target_seed = gan.generate_random_noise(2)
    
        
        for s, val in enumerate([s/steps for s in range(steps+1)]):
            
            if n == 'spherical':
                seed = np.expand_dims(slerp(val, start_seed, target_seed), axis=0)
            elif n == 'linear':
                seed = np.expand_dims((val * target_seed), axis=0)
            
            if s not in step_frames:
                step_frames[s] = []
            
            step_frames[s].append(gan.sample(seed)[0])
      
    save_dir = os.path.join(medium_dir, 'traversal', n)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
            
    for s in range(steps+1):
        save_single_image(os.path.join(save_dir, f'{s}.png'), np.concatenate(step_frames[s], axis=1))
