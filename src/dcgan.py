import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import json

from modules import upsampling_module, downsampling_module
from utils.images import load_single_image, save_single_image
from utils.images import transform_input, transform_output
from utils.images import randomly_flip_horizontal, add_instance_noise, randomly_jitter

from spectral_norm import SpectralNorm

class DCGAN:

    def __init__(self, name='dcgan', config={}, reload=False):

        self.name = name
        self.result_dir = os.path.join('../results', self.name)

        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        elif not reload:
            print(f'Directory for model {self.name} already exists: Choose another name or delete the old one manually.')
            exit()

        self.epoch = 0

        self.config = {
            'target_shape': (96, 96, 4),
            'noise_size': 128,
            'noise_type': 'normal',

            'g_lr': 1e-4,
            'd_lr': 1e-4,

            'spectral_norm': False,
            'initializer': 'glorot_normal',
            'one_sided_label_smoothing': 0.1,

            'discard_shiny': False,
            'flip_labels': False,
            'randomly_flip_labels': None,

            'instance_noise_stddev': 0.2,
            'instance_noise_decay': 0.01,
            'randomly_flip_horizontal': True,
            'randomly_jitter': False,

            'g_initial_tensor_channels': 256,
            'g_filter_sizes': [256, 128, 64],
            'g_kernel_sizes': [3, 5, 5],
            'g_dropout_rates': [None, 0.3, 0.3],

            'd_filter_sizes': [64, 128, 256],
            'd_kernel_sizes': [5, 5, 3],
            'd_dropout_rates': [None, 0.3, 0.3]
            }

        self.config.update(config)
        with open(os.path.join(self.result_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator.summary()
        self.discriminator.summary()

        self.initializer = 'glorot_normal'
        self.noise_type = 'normal'

        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(self.result_dir, 'generator.png'), show_shapes=True, dpi=128)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(self.result_dir, 'discriminator.png'), show_shapes=True, dpi=128)

        self.g_optimizer = tf.keras.optimizers.Adam(lr=self.config['g_lr'], beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(lr=self.config['d_lr'], beta_1=0.5)


    def fit(self, data_dir, epochs, batch_size, sample_interval=1):
        sample_height = 8
        sample_width = 8

        seeds = self.generate_random_noise(sample_height*sample_width)

        if self.epoch == 0:
            self.save_weights()
            self.show_performance(seeds, sample_height, sample_width)
            np.save(os.path.join(self.result_dir, 'seeds.npy'), seeds)
        else:
            seeds = np.load(os.path.join(self.result_dir, 'seeds.npy'))

        img_file_paths = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir)]
        steps = len(img_file_paths) // batch_size

        step_losses = None

        for self.epoch in range(self.epoch+1, self.epoch+epochs+1):
            img_file_paths = np.random.permutation(img_file_paths)

            for s in range(steps):
                 input_batches = self.step_preparation(s, img_file_paths, batch_size)

                 with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

                    g_loss, d_loss, single_losses = self.training_step(s, input_batches, batch_size)

                    g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
                    d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

                    self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
                    self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

                 status = f"[{self.epoch}|{epochs} : {s}|{steps}] g_loss: {g_loss} | d_loss: {d_loss}"
                 print(status)

                 if s == 0:
                    step_losses = np.zeros(shape=(steps, len(single_losses)))
                 step_losses[s] = single_losses

            if self.config['instance_noise_decay'] is not None and self.config['instance_noise_stddev'] is not None:
                 self.config['instance_noise_stddev'] *= (1.0 - self.config['instance_noise_decay'])

            save_path = os.path.join(self.result_dir, 'losses')
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            np.save(os.path.join(save_path, f'{self.epoch}.npy'), step_losses)

            if self.epoch % sample_interval == 0:
                 self.save_weights()
                 self.show_performance(seeds, sample_height, sample_width)


    def generate_random_noise(self, batch_size):
        if self.config['noise_type'] == 'normal':
            return np.random.normal(size=(batch_size, self.config['noise_size']))
        elif  self.config['noise_type'] == 'uniform':
            return np.random.uniform(-1, 1, size=(batch_size, self.config['noise_size']))


    def load_image(self, load_path):
        x = load_single_image(load_path)
        x = transform_input(x)
        if self.config['randomly_flip_horizontal']:
            x = randomly_flip_horizontal(x)
        if self.config['randomly_jitter']:
            x = randomly_jitter(x)
        return x


    def save_weights(self):
        weight_save_path = os.path.join(self.result_dir, 'weights')
        g_weight_save_path = os.path.join(weight_save_path, 'generator')
        d_weight_save_path = os.path.join(weight_save_path, 'discriminator')

        if not os.path.isdir(g_weight_save_path):
            os.makedirs(g_weight_save_path)
        if not os.path.isdir(d_weight_save_path):
            os.makedirs(d_weight_save_path)

        # save weights to file
        self.generator.save_weights(os.path.join(g_weight_save_path, f'{self.epoch}.h5'))
        self.discriminator.save_weights(os.path.join(d_weight_save_path, f'{self.epoch}.h5'))

        # save entire model to file as well (for convenience regarding future structural changes)
        self.generator.save(os.path.join(weight_save_path, f'g_model_state.h5'))
        self.discriminator.save(os.path.join(weight_save_path, f'd_model_state.h5'))


    def load_weights(self, epoch):
        weight_save_path = os.path.join(self.result_dir, 'weights')
        g_weight_save_path = os.path.join(weight_save_path, 'generator')
        d_weight_save_path = os.path.join(weight_save_path, 'discriminator')

        # load model structure since architecture might have changed
        self.generator = tf.keras.models.load_model(os.path.join(weight_save_path, f'g_model_state.h5'))
        self.discriminator= tf.keras.models.load_model(os.path.join(weight_save_path, f'd_model_state.h5'), custom_objects={'SpectralNorm': SpectralNorm})

        # restore weights
        self.generator.load_weights(os.path.join(g_weight_save_path, f'{epoch}.h5'))
        self.discriminator.load_weights(os.path.join(d_weight_save_path, f'{epoch}.h5'))

        with open(os.path.join(self.result_dir, 'config.json'), 'r') as f:
            self.config.update(json.load(f))
            self.config['instance_noise_stddev'] *= self.config['instance_noise_decay'] ** epoch

        self.epoch = epoch

    # ===================== SPECIFIC METHODS ===================================

    def build_generator(self):
        filter_sizes = self.config['g_filter_sizes']
        kernel_sizes = self.config['g_kernel_sizes']
        dropout_rates = self.config['g_dropout_rates']

        N = 2 ** (1 + len(filter_sizes))

        height, width, _ = self.config['target_shape']
        initial_tensor_shape = (height // N, width // N, self.config['g_initial_tensor_channels'])

        input_noise = tf.keras.layers.Input(self.config['noise_size'], name='input-noise')

        x = tf.keras.layers.Dense(np.prod(initial_tensor_shape), kernel_initializer=self.config['initializer'])(input_noise)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape(initial_tensor_shape)(x)

        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = upsampling_module(x, filters, kernels, strides=2, dropout=dropout, initializer=self.config['initializer'])

        x = tf.keras.layers.Conv2DTranspose(filters=self.config['target_shape'][-1], kernel_size=5, strides=2, padding='same', kernel_initializer=self.config['initializer'])(x)

        x = tf.keras.layers.Activation('tanh')(x)

        return tf.keras.Model(inputs=[input_noise], outputs=[x], name='generator')


    def build_discriminator(self):
        filter_sizes = self.config['d_filter_sizes']
        kernel_sizes = self.config['d_kernel_sizes']
        dropout_rates = self.config['d_dropout_rates']

        input_img = tf.keras.layers.Input(self.config['target_shape'], name='input-img')

        x = input_img
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = downsampling_module(x, filters, kernels, strides=2, dropout=dropout, spectral_norm=self.config['spectral_norm'], initializer=self.config['initializer'])

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, kernel_initializer=self.config['initializer'], kernel_constraint=SpectralNorm() if self.config['spectral_norm'] else None)(x)

        return tf.keras.Model(inputs=[input_img], outputs=[x], name='discriminator')


    def step_preparation(self, s, img_file_paths, batch_size):
        batch_file_paths = img_file_paths[s*batch_size:(s+1)*batch_size]

        img_batch = np.zeros((batch_size,) + self.config['target_shape'])
        for i in range(batch_size):
            img_batch[i] = self.load_image(batch_file_paths[i])

        noise_batch = self.generate_random_noise(batch_size)

        return [noise_batch, img_batch]


    def training_step(self, s, input_batches, batch_size):
        valid = tf.ones(shape=(batch_size, 1))
        fake = tf.zeros(shape=(batch_size, 1))

        if self.config['flip_labels']:
            valid, fake = fake, valid
            valid_smooth = valid * self.config['one_sided_label_smoothing']
        else:
            valid_smooth = valid * (1.0 - self.config['one_sided_label_smoothing'])

        if self.config['randomly_flip_labels'] is not None:
            if tf.random.uniform((), 0, 1) < self.config['randomly_flip_labels']:
                valid, fake, = fake, valid

        loss_fct = lambda true, pred: tf.reduce_mean(tf.keras.losses.binary_crossentropy(true, pred, from_logits=True))

        def compute_losses(real_src_batch, fake_src_batch):
            g_real_src_loss = loss_fct(valid, fake_src_batch)
            d_real_src_loss = loss_fct(valid, real_src_batch)
            d_fake_src_loss = loss_fct(fake, fake_src_batch)

            g_loss = g_real_src_loss
            d_loss = 0.5 * (d_real_src_loss + d_fake_src_loss)

            return g_loss, d_loss, [g_real_src_loss, d_real_src_loss // 2, d_fake_src_loss // 2]

        noise_batch, img_batch = input_batches
        fake_batch = self.generator(noise_batch)

        if self.config['instance_noise_stddev'] not in [None, 0.0]:
            img_batch = add_instance_noise(img_batch, stddev=self.config['instance_noise_stddev'])
            fake_batch = add_instance_noise(fake_batch, stddev=self.config['instance_noise_stddev'])

        real_src_batch = self.discriminator(img_batch)
        fake_src_batch = self.discriminator(fake_batch)

        g_loss, d_loss, single_losses = compute_losses(real_src_batch, fake_src_batch)

        return g_loss, d_loss, single_losses


    def sample(self, seeds=None, n_seeds=8):
        if seeds is None:
            seeds = self.generate_random_noise(n_seeds)

        fakes = self.generator(seeds)
        fakes = transform_output(fakes)
        return fakes


    def show_performance(self, seeds, height, width):
        fakes = self.sample(seeds)
        fakes = np.reshape(fakes, (height, width,) + self.config['target_shape'])
        fakes = np.squeeze(np.concatenate(np.split(fakes, height, axis=0), axis=2))
        fakes = np.concatenate(fakes, axis=1)

        save_path = os.path.join(self.result_dir, 'samples')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        save_single_image(os.path.join(save_path, f'{self.epoch}.png'), fakes)


    def plot_history(self, sample_interval=16):
        def plot(history, x_label, y_label, labels, save_path):
            history = np.swapaxes(history, 0, 1)
            plt.clf()
            for h, label in zip(history, labels):
                plt.plot(h, label=label)
            plt.legend()
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(save_path, dpi=256)

        load_path = os.path.join(self.result_dir, 'losses')

        step_losses = []
        epoch_losses = []

        loss_files = os.listdir(load_path)

        for e in range(0, self.epoch+1, sample_interval):
            if e == 0:
                continue

            step_loss = np.load(os.path.join(load_path, f'{e}.npy'))
            step_losses.append(step_loss)
            epoch_losses.append(np.mean(step_loss, axis=0))

        step_losses = np.concatenate(step_losses, axis=0)
        print(step_losses.shape)

        plot(step_losses, 'batches', 'loss', ['g_real_src_loss', 'd_real_src_loss', 'd_fake_src_loss'], os.path.join(self.result_dir, 'step_history.png'))
        plot(epoch_losses, 'epochs', 'loss', ['g_real_src_loss', 'd_real_src_loss', 'd_fake_src_loss'], os.path.join(self.result_dir, 'epoch_history.png'))

if __name__ == '__main__':
    DATA_DIR = 'C:/Users/Jonas/Documents/GitHub/pokemon-generation/data'

    config = {}
    dcgan = DCGAN(name='dcgan', config=config)

    dcgan.fit(DATA_DIR, 500, 32, 1)
