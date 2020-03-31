# taken from: https://colab.research.google.com/drive/1f2Ejlm3UmsthqQni9vFkGmTTcS_ndqXR#scrollTo=flShjqVg7Nu5

import tensorflow as tf

def l2_normalize(x, eps=1e-12):
  '''
  Scale input by the inverse of it's euclidean norm
  '''
  return x / tf.linalg.norm(x + eps)

POWER_ITERATIONS = 5

class SpectralNorm(tf.keras.constraints.Constraint):
    '''
    Uses power iteration method to calculate a fast approximation
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, n_iters=POWER_ITERATIONS):
        self.n_iters = n_iters

    def __call__(self, w):
      flattened_w = tf.reshape(w, [w.shape[0], -1])
      u = tf.random.normal([flattened_w.shape[0]])
      v = tf.random.normal([flattened_w.shape[1]])
      for i in range(self.n_iters):
        v = tf.linalg.matvec(tf.transpose(flattened_w), u)
        v = l2_normalize(v)
        u = tf.linalg.matvec(flattened_w, v)
        u = l2_normalize(u)
      sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
      return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}
