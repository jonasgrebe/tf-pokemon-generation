import tensorflow as tf
import numpy as np
import os

from utils.pokemon import get_all_pokedex_categories, get_pokedex_entry, extract_pokedex_id
from utils.images import load_single_image, save_single_image, transform_input, transform_output
from utils.images import randomly_flip_horizontal, randomly_jitter, add_instance_noise

from utils.visual import get_pokemon_vis_by_sprite

class PokeGAN:
    
    def __init__(self, name='pokegan'):
        
        self.name = name
        self.result_dir = os.path.join('../results', self.name)
        
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        
        self.target_shape = (96, 96, 4)
        self.noise_size = 128
        
        self.types = get_all_pokedex_categories('type_1')
        self.shapes = get_all_pokedex_categories('shape')
        
        self.n_types = len(self.types)
        self.n_shapes = len(self.shapes)
        
        self.type_labels = {}
        for label, t in enumerate(self.types):
            self.type_labels[t] = label
        
        self.shape_labels = {}
        for label, t in enumerate(self.shapes):
            self.shape_labels[t] = label
        
        self.get_type_1_label = lambda dex_id: self.type_labels[get_pokedex_entry(dex_id, ['type_1'])[0]]
        self.get_shape_label = lambda dex_id: self.shape_labels[get_pokedex_entry(dex_id, ['shape'])[0]]
        
        self.initializer = 'glorot_uniform'
        self.epoch = 0
        
        self.generator = self.build_generator()
        self.generator.summary()
        
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        
        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(self.result_dir, 'generator.png'), show_shapes=True, dpi=128)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(self.result_dir, 'discriminator.png'), show_shapes=True, dpi=128)
        
        self.src_loss_fct = lambda true, pred: tf.reduce_mean(tf.keras.losses.binary_crossentropy(true, pred, from_logits=True))
        self.aux_loss_fct = lambda true, pred: tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(true, pred, from_logits=True))

        self.g_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
    
    
    def embed_input_as_tensor(self, x, latent_size, shape):
        if x.shape[1] == 1:
            x = tf.keras.layers.Embedding(len(self.types), latent_size)(x)
        else:
            x = tf.keras.layers.Dense(units=latent_size, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.ReLU()(x)          
        
        x = tf.keras.layers.Dense(units=np.prod(shape), kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape(target_shape=shape)(x)
        return x
    
    
    def upsampling_module(self, x, filters, kernels, steps=1, dropout=None):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=2, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.ReLU()(x)
        
        if dropout:
            x = tf.keras.layers.Dropout(rate=dropout)(x)
            
        for _ in range(steps-1):
            
            x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            x = tf.keras.layers.ReLU()(x)
            
            if dropout:
                x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        return x
    

    def downsampling_module(self, x, filters, kernels, steps=1, dropout=None):               
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=2, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        if dropout:
            x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        for _ in range(steps-1):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            
            if dropout:
                x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        return x
    
    
    def classifier(self, x, output_units, dropout, name):
        x = tf.keras.layers.Dense(output_units, name=name)(x)
        return x
    

    def build_generator(self):
        filter_sizes = (512, 256, 128, 64)
        kernel_sizes = (3, 5, 5, 5)
        dropout_rates = (0.1, 0.2, 0.2, 0.2)
        
        input_noise = tf.keras.layers.Input(self.noise_size, name='input-noise')
        input_type_1 = tf.keras.layers.Input(1, name='input-type_1-onehot')
        input_shape = tf.keras.layers.Input(1, name='input-shape-onehot')
        
        initial_tensor_shape = (6, 6, 3)
        
        noise_tensor = self.embed_input_as_tensor(input_noise, latent_size=128, shape=initial_tensor_shape)
        type_1_tensor = self.embed_input_as_tensor(input_type_1, latent_size=64, shape=initial_tensor_shape)
        shape_tensor = self.embed_input_as_tensor(input_shape, latent_size=64, shape=initial_tensor_shape)
        
        aux_tensor = tf.keras.layers.Concatenate()([type_1_tensor, shape_tensor])
        
        m = noise_tensor
        a = aux_tensor
        x = tf.keras.layers.Concatenate()([m, a])
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = self.upsampling_module(x, filters, kernels, 1, dropout)
            #a = self.upsampling_module(a, filters, kernels, 1, dropout)
        
        output = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer,
                                                 activation='tanh')(x)
                
        assert output.shape[1:] == self.target_shape
        
        return tf.keras.Model(inputs=[input_noise, input_type_1, input_shape], outputs=[output])
        
    def build_discriminator(self):
        
        filter_sizes = (256, 128, 64)
        kernel_sizes = (3, 5, 5)
        dropout_rates = (0.25, 0.25, 0.25)
        clf_dropout = 0.4
        
        input_img = tf.keras.layers.Input(self.target_shape, name='input-image')

        x = input_img
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = self.downsampling_module(x, filters, kernels, 1, dropout)
        
        x = tf.keras.layers.Flatten()(x)
        
        d_src = self.classifier(x, 1, dropout=clf_dropout, name='src-output')
        d_aux_type_1 = self.classifier(x, self.n_types, dropout=clf_dropout, name='aux-type_1-output')
        d_aux_shape = self.classifier(x, self.n_shapes, dropout=clf_dropout, name='aux-shape-output')
        
        return tf.keras.Model(inputs=[input_img], outputs=[d_src, d_aux_type_1, d_aux_shape])
        
    
    def compute_losses(self, real_src_batch, fake_src_batch, real_aux_type_1_batch, real_aux_shape_batch, fake_aux_type_1_batch, fake_aux_shape_batch, type_1_batch, shape_batch):

        # compute partial source losses
        d_real_src_loss = self.src_loss_fct(tf.ones_like(real_src_batch)*0.9, real_src_batch)
        d_fake_src_loss = self.src_loss_fct(tf.zeros_like(fake_src_batch), fake_src_batch)
        g_src_loss = self.src_loss_fct(tf.ones_like(fake_src_batch), fake_src_batch)

        # compute partial auxiliary losses
        real_aux_type_1_loss = self.aux_loss_fct(type_1_batch, real_aux_type_1_batch)
        fake_aux_type_1_loss = self.aux_loss_fct(type_1_batch, fake_aux_type_1_batch)
        real_aux_shape_loss = self.aux_loss_fct(shape_batch, real_aux_shape_batch)
        fake_aux_shape_loss = self.aux_loss_fct(shape_batch, fake_aux_shape_batch)

        aux_type_1_loss = real_aux_type_1_loss + fake_aux_type_1_loss
        aux_shape_loss = real_aux_shape_loss + fake_aux_shape_loss

        aux_loss = 0.5 * (aux_type_1_loss + aux_shape_loss)
        aux_loss *= 1.0

        g_loss = g_src_loss + aux_loss
        d_loss = 0.5 * (d_real_src_loss + d_fake_src_loss) + aux_loss

        return g_loss, d_loss, aux_loss, (g_src_loss, d_real_src_loss, d_fake_src_loss, aux_type_1_loss, aux_shape_loss)
    

    
    def generate_random_noise(self, batch_size):
        return tf.random.normal((batch_size, self.noise_size))
    
    
    def load_image(self, load_path):
        x = load_single_image(load_path)
        x = transform_input(x)
        x = randomly_flip_horizontal(x)
        x = randomly_jitter(x)
        return x
    
    
    def fit(self, data_dir, epochs, batch_size, sample_interval, n_seeds=8):
        
        seeds = self.generate_random_noise(n_seeds)

        if self.epoch == 0:
            self.save_weights()
            self.check_performance(seeds=seeds)
            np.save(os.path.join(self.result_dir, 'seeds.npy'), seeds)
        else:
            seeds = np.load(os.path.join(self.result_dir, 'seeds.npy'))

        img_file_paths = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir)]
        steps = len(img_file_paths) // batch_size
        
        for self.epoch in range(self.epoch+1, self.epoch+epochs+1):
            img_file_paths = np.random.permutation(img_file_paths)
            
            loss_per_steps = np.zeros(shape=(steps, 5))
            for s in range(steps):
                batch_file_paths = img_file_paths[s*batch_size:(s+1)*batch_size]
                
                img_batch = np.zeros(shape=(batch_size,) + self.target_shape)  
                type_1_batch = np.zeros(shape=(batch_size, 1))  
                shape_batch = np.zeros(shape=(batch_size, 1))
                
                for i, img_file_path in enumerate(batch_file_paths):
                    img_batch[i] = self.load_image(img_file_path)
                    
                    dex_id = extract_pokedex_id(img_file_path)
                    type_1 = self.get_type_1_label(dex_id)
                    shape = self.get_shape_label(dex_id)
                    
                    type_1_batch[i] = type_1
                    shape_batch[i] = shape
                    
                with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                    noise_batch = self.generate_random_noise(batch_size)
                    fake_batch = self.generator([noise_batch, type_1_batch, shape_batch])
                    
                    img_batch = add_instance_noise(img_batch)
                    fake_batch = add_instance_noise(fake_batch)

                    real_src_batch, real_aux_type_1_batch, real_aux_shape_batch = self.discriminator(img_batch)
                    fake_src_batch, fake_aux_type_1_batch, fake_aux_shape_batch = self.discriminator(fake_batch)

                    g_loss, d_loss, aux_loss, losses = self.compute_losses(real_src_batch, fake_src_batch, real_aux_type_1_batch, real_aux_shape_batch, fake_aux_type_1_batch, fake_aux_shape_batch, type_1_batch, shape_batch)
 
                    g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
                    d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

                    self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
                    self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
    
                loss_per_steps[s] = losses
                
                status = f"[{self.epoch}|{epochs} : {s}|{steps}] g_loss: {g_loss-aux_loss} | d_loss: {d_loss-aux_loss} | aux_loss: {aux_loss}"
                print(status)
                
            if not os.path.isdir(os.path.join(self.result_dir, 'history', 'loss')):
                os.makedirs(os.path.join(self.result_dir, 'history', 'loss'))

            np.save(os.path.join(self.result_dir, 'history', 'loss', f'{self.epoch}.npy'), loss_per_steps)
            
            if self.epoch % sample_interval == 0:
                self.save_weights()
                self.check_performance(seeds)


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


    def check_performance(self, seeds):
        batch_size = len(seeds)    
    
        save_path = os.path.join(self.result_dir, 'samples')
        
        combinations = [
            ('grass', 'ball'),
            ('grass', 'quadruped'),
            ('bug', 'armor'),
            ('bug', 'bug-wings'),
            ('fire', 'humanoid'),
            ('fire', 'legs'),
            ('water', 'fish'),
            ('water', 'quadruped'),
            ('ice', 'heads'),
            ('steel', 'heads'),
            ('dragon', 'wings'),
            ('dark', 'wings'),
            ('electric', 'squiggle'),
            ('fairy', 'quadruped'),
            ('fairy', 'humanoid')
            ]
    
        for type_1, shape in combinations:
    
            label_type_1 = self.types.index(type_1)
            label_shape = self.shapes.index(shape)

            label_type_1_batch = np.full((batch_size, 1), label_type_1)
            label_shape_batch =  np.full((batch_size, 1), label_shape)

            fake_batch = self.generator([seeds, label_type_1_batch, label_shape_batch])
            
            _, pred_type_1_batch, pred_shape_batch = self.discriminator(fake_batch)
            
            pred_type_1_batch = tf.nn.softmax(pred_type_1_batch)
            pred_shape_batch = tf.nn.softmax(pred_shape_batch)

            fake_batch = transform_output(fake_batch)
            
            vis_list = [get_pokemon_vis_by_sprite(sprite, type_1, self.types, type_1_scores) for sprite, type_1_scores, shape_scores in zip(fake_batch, pred_type_1_batch, pred_shape_batch)]
            spacing = np.zeros(shape=(vis_list[0].shape[0], 4, 4))
            for i in range(len(vis_list)-1):
                vis_list[i] = np.concatenate([vis_list[i], spacing], axis=1)
            vis = np.concatenate(vis_list, axis=1)

            combi_save_path = os.path.join(save_path, f'{type_1}-{shape}')
            if not os.path.isdir(combi_save_path):
                os.makedirs(combi_save_path)

            save_single_image(os.path.join(combi_save_path, f'{self.epoch}.png'), vis)


    def plot_entire_history(self):
        loss_histories = []
        for e in range(1, self.epoch+1):
            loss_histories.append(np.load(os.path.join(self.result_dir, 'history', 'loss', f'{e}.npy')))
        loss_histories = np.concatenate(loss_histories, axis=0)

        plot_history(loss_histories, 'loss', ['g_src_loss', 'd_real_src_loss', 'd_fake_src_loss', 'aux_type_1_loss', 'aux_shape_loss'], os.path.join(self.result_dir, 'history', 'entire_loss_history.jpg'))

if __name__ == '__main__':
    DATA_DIR = 'C:/Users/Jonas/Documents/GitHub/pokemon-generation/data/sprites'
    gan = PokeGAN(name='pokegan_3')
    
    gan.fit(DATA_DIR, 100, 16, 1)