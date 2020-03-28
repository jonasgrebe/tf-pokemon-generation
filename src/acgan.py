import tensorflow as tf
import numpy as np
import os

from modules import upsampling_module, downsampling_module
from utils.images import load_single_image, save_single_image
from utils.images import transform_input, transform_output
from utils.images import randomly_flip_horizontal, add_instance_noise

from utils.pokemon import get_pokedex_entry, get_all_pokedex_categories, extract_pokedex_id

from dcgan import DCGAN

class ACGAN(DCGAN):

    def __init__(self, name='acgan', label_column='type_1', config={}):
        classes = get_all_pokedex_categories(label_column) # pokemon-specic

        c = {
            'label_column': label_column,
            'classes': classes,
            'n_classes': len(classes),
            'latent_size': 32,
            'g_aux_loss_weight': 1.0,
            'd_aux_loss_weight': 1.0,
            'g_initial_label_tensor_channels': 1
        }
        c.update(config)
        super(ACGAN, self).__init__(name, c)

        self.class_label = {cls: label for label, cls in enumerate(self.config['classes'])}
        print(self.class_label)

    def build_generator(self):
        filter_sizes = self.config['g_filter_sizes']
        kernel_sizes = self.config['g_kernel_sizes']
        dropout_rates = self.config['g_dropout_rates']

        N = 2 ** (1 + len(filter_sizes))

        height, width, _ = self.config['target_shape']
        initial_noise_tensor_shape = (height // N, width // N, self.config['g_initial_tensor_channels'])
        initial_label_tensor_shape = (height // N, width // N, self.config['g_initial_label_tensor_channels'])

        input_noise = tf.keras.layers.Input(self.config['noise_size'], name='input-noise')
        input_label = tf.keras.layers.Input(1, name='input-label')

        x = tf.keras.layers.Dense(np.prod(initial_noise_tensor_shape), kernel_initializer=self.config['initializer'])(input_noise)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(initial_noise_tensor_shape)(x)

        y = tf.keras.layers.Embedding(self.config['n_classes'], self.config['latent_size'])(input_label)
        y = tf.keras.layers.Dense(np.prod(initial_label_tensor_shape), kernel_initializer=self.config['initializer'])(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Reshape(initial_label_tensor_shape)(y)

        x = tf.keras.layers.Concatenate()([x, y])

        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = upsampling_module(x, filters, kernels, strides=2, dropout=dropout, spectral_norm=False, initializer=self.config['initializer'])

        x = tf.keras.layers.Conv2DTranspose(filters=self.config['target_shape'][-1], kernel_size=5, strides=2, padding='same', kernel_initializer=self.config['initializer'])(x)
        x = tf.keras.layers.Activation('tanh')(x)

        return tf.keras.Model(inputs=[input_noise, input_label], outputs=[x], name='generator')


    def build_discriminator(self):
        filter_sizes = self.config['d_filter_sizes']
        kernel_sizes = self.config['d_kernel_sizes']
        dropout_rates = self.config['d_dropout_rates']

        input_img = tf.keras.layers.Input(self.config['target_shape'], name='input-img')

        x = input_img
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = downsampling_module(x, filters, kernels, strides=2, dropout=dropout, spectral_norm=self.config['spectral_norm'], initializer=self.config['initializer'])

        x = tf.keras.layers.Flatten()(x)

        src = tf.keras.layers.Dense(1, kernel_initializer=self.config['initializer'])(x)
        label = tf.keras.layers.Dense(self.config['n_classes'], kernel_initializer=self.config['initializer'])(x)

        return tf.keras.Model(inputs=[input_img], outputs=[src, label], name='discriminator')


    def step_preparation(self, s, img_file_paths, batch_size):

        def get_label_for_img(load_path):
            dex_id = extract_pokedex_id(load_path) # pokemon-specic
            cls = get_pokedex_entry(dex_id, [self.config['label_column']])[0] # pokemon-specic
            return self.class_label[cls]

        batch_file_paths = img_file_paths[s*batch_size:(s+1)*batch_size]

        img_batch = np.zeros((batch_size,) + self.config['target_shape'])
        label_batch = np.zeros((batch_size,1))
        for i in range(batch_size):
            img_batch[i] = self.load_image(batch_file_paths[i])
            label_batch[i] = get_label_for_img(batch_file_paths[i])

        noise_batch = self.generate_random_noise(batch_size)

        return [noise_batch, img_batch, label_batch]


    def training_step(self, s, input_batches, batch_size):

        aux_loss_fct = lambda true, pred: tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(true, pred, from_logits=True))

        def compute_losses(real_src_batch, fake_src_batch, real_aux_batch, fake_aux_batch, label_batch):
            g_real_src_loss = self.loss_fct(self.valid, fake_src_batch)
            d_real_src_loss = self.loss_fct(self.valid, real_src_batch)
            d_fake_src_loss = self.loss_fct(self.fake, fake_src_batch)

            d_real_aux_loss = aux_loss_fct(label_batch, real_aux_batch)
            d_fake_aux_loss = aux_loss_fct(label_batch, fake_aux_batch)
            aux_loss = 0.5 * (d_real_aux_loss + d_fake_aux_loss)

            g_loss = g_real_src_loss + self.config['g_aux_loss_weight'] * aux_loss
            d_loss = 0.5 * (d_real_src_loss + d_fake_src_loss) + self.config['d_aux_loss_weight'] * aux_loss

            return g_loss, d_loss, aux_loss

        noise_batch, img_batch, label_batch = input_batches
        fake_batch = self.generator([noise_batch, label_batch])

        if self.config['instance_noise_stddev'] not in [None, 0.0]:
            img_batch = add_instance_noise(img_batch, stddev=self.config['instance_noise_stddev'])
            fake_batch = add_instance_noise(fake_batch, stddev=self.config['instance_noise_stddev'])

        real_src_batch, real_aux_batch = self.discriminator(img_batch)
        fake_src_batch, fake_aux_batch = self.discriminator(fake_batch)

        g_loss, d_loss, aux_loss = compute_losses(real_src_batch, fake_src_batch, real_aux_batch, fake_aux_batch, label_batch)

        return g_loss, d_loss, [g_loss, d_loss, aux_loss]


    def sample(self, seeds=None, labels=None, n_seeds=8):
        if seeds is None:
            seeds = self.generate_random_noise(n_seeds)
        if labels is None:
            labels = np.random.choice(np.arange(self.config['n_classes']), n_seeds)

        fakes = self.generator([seeds, labels])
        fakes = transform_output(fakes)
        return fakes


    def show_performance(self, seeds, height, width):

        for cls in self.config['classes']:
            label = self.class_label[cls]
            label_batch = np.full((len(seeds), 1), label)

            fakes = self.sample(seeds, label_batch)
            fakes = np.reshape(fakes, (height, width,) + self.config['target_shape'])
            fakes = np.squeeze(np.concatenate(np.split(fakes, height, axis=0), axis=2))
            fakes = np.concatenate(fakes, axis=1)

            save_path = os.path.join(self.result_dir, 'samples', cls)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_single_image(os.path.join(save_path, f'{self.epoch}.png'), fakes)


if __name__ == '__main__':
    DATA_DIR = 'C:/Users/Jonas/Documents/GitHub/pokemon-generation/data/sprites'

    config = {}
    acgan = ACGAN(name='acgan_shape', label_column='shape', config=config)

    acgan.fit(DATA_DIR, 500, 32, 1)
