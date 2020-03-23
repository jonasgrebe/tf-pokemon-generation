import tensorflow as tf
from imageio import imread, imwrite

def load_single_image(img_file_path):
    return imread(img_file_path, pilmode='RGBA')

def save_single_image(img_file_path, img):
    return imwrite(img_file_path, img)

def transform_input(x):
    return (x - 127.5) / 127.5

def transform_output(x):
    return (x * 127.5) + 127.5

# data augmentation

def randomly_flip_horizontal(x):
    return tf.image.random_flip_left_right(x)

def randomly_jitter(x):
    x = tf.image.resize(x, [100, 100], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.random_crop(x, size=(96, 96, 4))
    return x

def randomly_adjust_hue(x, max_delta=0.01):
    color = x[:,:,:3]
    color = tf.image.random_hue(color, max_delta=max_delta)
    return tf.concat([color, x[:,:,-1:]], axis=-1) 

def randomly_rotate(x, max_angle=10):
    x = tf.keras.preprocessing.image.randomly_rotate(x, 0, max_angle)

def add_instance_noise(x, stddev=0.1):
    return x + tf.random.normal(shape=x.shape, mean=0, stddev=stddev, dtype=x.dtype)