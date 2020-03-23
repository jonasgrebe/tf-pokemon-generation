import tensorflow as tf
import numpy as np
import os


class PokeGAN:
    
    def __init__(self, name='pokegan'):
        
        self.name = name
        self.result_dir = os.path.join('../../results', self.name)
        
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        
        self.target_shape = (96, 96, 4)
        self.noise_size = 128
        self.n_types = 18
        self.n_shapes = 14
        
        self.initializer = 'glorot_uniform'
        
        self.generator = self.build_generator()
        self.generator.summary()
        
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        
        tf.keras.utils.plot_model(self.generator, to_file=os.path.join(self.result_dir, 'generator.png'), show_shapes=True, dpi=128)
        tf.keras.utils.plot_model(self.discriminator, to_file=os.path.join(self.result_dir, 'discriminator.png'), show_shapes=True, dpi=128)
     
    
    def embed_input_as_tensor(self, x, latent_size, shape):
        x = tf.keras.layers.Dense(units=latent_size)(x)
        x = tf.keras.layers.Dense(units=np.prod(shape))(x)
        x = tf.keras.layers.Reshape(target_shape=shape)(x)
        return x
        
    
    def upsampling_module(self, x, filters, kernels, steps=1, dropout=None):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=2, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        if dropout:
            x = tf.keras.layers.Dropout(rate=dropout)(x)
            
        for _ in range(steps-1):
            x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernels, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            
            if dropout:
                x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        return x
    

    def downsampling_module(self, x, filters, kernels, steps=1, dropout=None):               
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=2, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        if dropout:
            x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        for _ in range(steps-1):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            
            if dropout:
                x = tf.keras.layers.Dropout(rate=dropout)(x)
                
        return x
    
    
    def classifier(self, x, output_units, dropout, name):
        x = self.downsampling_module(x, filters=x.shape[-1] // 2, kernels=3, steps=1, dropout=dropout)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(output_units, name=name)(x)
        return x
    

    def build_generator(self):
        
        filter_sizes = (256, 128, 64)
        kernel_sizes = (3, 5, 5)
        dropout_rates = (0.2, 0.2, 0.2)
        
        input_noise = tf.keras.layers.Input(self.noise_size, name='input-noise')
        input_types = tf.keras.layers.Input(self.n_types, name='input-types-onehot')
        input_shape = tf.keras.layers.Input(self.n_shapes, name='input-shape-onehot')
        
        noise_tensor = self.embed_input_as_tensor(input_noise, latent_size=128, shape=(12, 12, 1))
        types_tensor = self.embed_input_as_tensor(input_types, latent_size=128, shape=(12, 12, 2))
        shape_tensor = self.embed_input_as_tensor(input_shape, latent_size=128, shape=(12, 12, 1))
        
        aux_tensor = tf.keras.layers.Concatenate()([types_tensor, shape_tensor])
        
        m = noise_tensor
        a = aux_tensor
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = tf.keras.layers.Concatenate()([m, a])
            m = self.upsampling_module(x, filters, kernels, 2)
            a = self.upsampling_module(a, filters, kernels, 1, dropout)
        
        output = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=1, padding='same', use_bias=False, kernel_initializer=self.initializer,
                                                 activation='tanh')(m)
        
        assert output.shape[1:] == self.target_shape
        
        return tf.keras.Model(inputs=[input_noise, input_types, input_shape], outputs=[output])
        
        
    def build_discriminator(self):
        
        filter_sizes = (512, 256, 128)
        kernel_sizes = (3, 5, 5)
        dropout_rates = (0.2, 0.2, 0.2)
        clf_dropout = 0.4
        
        
        input_img = tf.keras.layers.Input(self.target_shape, name='input-image')

        x = input_img
        for filters, kernels, dropout in zip(filter_sizes, kernel_sizes, dropout_rates):
            x = self.downsampling_module(x, filters, kernels, 1, dropout)
        
        d_src = self.classifier(x, 1, dropout=clf_dropout, name='src-output')
        d_aux_types = self.classifier(x, self.n_types+1, dropout=clf_dropout, name='aux-types-output')
        d_aux_shape = self.classifier(x, self.n_shapes, dropout=clf_dropout, name='aux-shape-output')
        
        return tf.keras.Model(inputs=[input_img], outputs=[d_src, d_aux_types, d_aux_shape])
        
    
gan = PokeGAN()