import tensorflow as tf
import numpy as np
import os

from modules import upsampling_module, downsampling_module
from utils.images import load_single_image, save_single_image
from utils.images import transform_input, transform_output
from utils.images import randomly_flip_horizontal, add_instance_noise

class DCGAN:

    def __init__(self, name='dcgan', config={}):

        self.name = name
        self.result_dir = os.path.join('../results', self.name)

        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        self.epoch = 0

        self.config = {
            'target_shape': (96, 96, 4),
            'noise_size': 128,
            'g_lr': 1e-4,
            'd_lr': 1e-4,
            'discard_shiny': False,
            'one_sided_label_smoothing': 0.1,
            'instance_noise': True
            }
        self.config.update(config)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator.summary()
        self.discriminator.summary()

        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(self.result_dir, 'generator.png'), show_shapes=True, dpi=128)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(self.result_dir, 'discriminator.png'), show_shapes=True, dpi=128)

        self.g_optimizer = tf.keras.optimizers.Adam(lr=self.config['g_lr'], beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(lr=self.config['d_lr'], beta_1=0.5)


    def build_generator(self):
        filter_sizes = [256, 256, 128, 128]
        kernel_sizes = [3, 5, 5, 5]
        dropout_rates = [None, None, None, None]

        N = 2 ** len(filter_sizes)

        height, width, _ = self.config['target_shape']
        initial_tensor_shape = (height // N, width // N, 1)

        input_noise = tf.keras.layers.Input(self.config['noise_size'], name='input-noise')

        x = tf.keras.layers.Dense(np.prod(initial_tensor_shape))(input_noise)
        x = tf.keras.layers.Reshape(initial_tensor_shape)(x)

        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = upsampling_module(x, filters, kernels, strides=2, dropout=dropout)

        x = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=1, padding='same', activation='tanh')(x)

        return tf.keras.Model(inputs=[input_noise], outputs=[x], name='generator')


    def build_discriminator(self):
        filter_sizes = [128, 256, 512]
        kernel_sizes = [5, 5, 3]
        dropout_rates = [0.1, 0.2, 0.4]

        input_img = tf.keras.layers.Input(self.config['target_shape'], name='input-img')

        x = input_img
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = downsampling_module(x, filters, kernels, strides=2, dropout=dropout)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=[input_img], outputs=[x], name='discriminator')


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

            self.save_weights()
            if self.epoch % sample_interval == 0:
                self.show_performance(seeds, sample_height, sample_width)


    def step_preparation(self, s, img_file_paths, batch_size):
        batch_file_paths = img_file_paths[s*batch_size:(s+1)*batch_size]

        img_batch = np.zeros((batch_size,) + self.config['target_shape'])
        for i in range(batch_size):
            img_batch[i] = self.load_image(batch_file_paths[i])

        noise_batch = self.generate_random_noise(batch_size)

        return [noise_batch, img_batch]


    def training_step(self, s, input_batches, batch_size):

        valid = tf.ones(shape=(batch_size, 1))
        valid_smooth = valid * (1.0 - self.config['one_sided_label_smoothing'])
        fake = tf.zeros(shape=(batch_size, 1))

        loss_fct = lambda true, pred: tf.reduce_mean(tf.keras.losses.binary_crossentropy(true, pred, from_logits=True))

        def compute_losses(real_src_batch, fake_src_batch):
            g_real_src_loss = loss_fct(valid, fake_src_batch)
            d_real_src_loss = loss_fct(valid, real_src_batch)
            d_fake_src_loss = loss_fct(fake, fake_src_batch)

            g_loss = g_real_src_loss
            d_loss = 0.5 * (d_real_src_loss + d_fake_src_loss)

            return g_loss, d_loss

        noise_batch, img_batch = input_batches
        fake_batch = self.generator(noise_batch)

        if self.config['instance_noise']:
            img_batch = add_instance_noise(img_batch)
            fake_batch = add_instance_noise(fake_batch)

        real_src_batch = self.discriminator(img_batch)
        fake_src_batch = self.discriminator(fake_batch)

        g_loss, d_loss = compute_losses(real_src_batch, fake_src_batch)

        return g_loss, d_loss, [g_loss, d_loss]


    def sample(self, seeds=None, n_seeds=8):
        if seeds is None:
            seeds = self.generate_random_noise(n_seeds)

        fakes = self.generator(seeds)
        fakes = transform_output(fakes)
        return fakes


    def generate_random_noise(self, batch_size):
        return np.random.uniform(low=-1, high=1, size=(batch_size, self.config['noise_size']))


    def load_image(self, load_path):
        x = load_single_image(load_path)
        x = transform_input(x)
        return x


    def show_performance(self, seeds, height, width):
        fakes = self.sample(seeds)
        fakes = np.reshape(fakes, (height, width,) + self.config['target_shape'])
        fakes = np.squeeze(np.concatenate(np.split(fakes, height, axis=0), axis=2))
        fakes = np.concatenate(fakes, axis=1)

        save_path = os.path.join(self.result_dir, 'samples')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        save_single_image(os.path.join(save_path, f'{self.epoch}.png'), fakes)


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
        self.discriminator= tf.keras.models.load_model(os.path.join(weight_save_path, f'd_model_state.h5'))

        # restore weights
        self.generator.load_weights(os.path.join(g_weight_save_path, f'{epoch}.h5'))
        self.discriminator.load_weights(os.path.join(d_weight_save_path, f'{epoch}.h5'))

        self.epoch = epoch

if __name__ == '__main__':
    DATA_DIR = 'C:/Users/Jonas/Documents/GitHub/pokemon-generation/data/sprites'
    dcgan = DCGAN()

    dcgan.fit(DATA_DIR, 100, 16, 1)
