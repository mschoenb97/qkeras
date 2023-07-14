"""
Element-wise gradient scaling quantizer.

See https://arxiv.org/pdf/2104.00903.pdf
"""

import sys
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer


# update path so that we have access to qkeras
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class quantized_ewgs(Layer):
  """Quantizer for element-wise gradient scaling.
  
  Notes:
    - We're treating delta as a hyperparameter here
    - In the EWGS code, u and l are initialized based on these LOCs:
      https://github.com/cvlab-yonsei/EWGS/blob/main/ImageNet/custom_modules.py#L109-L117
      Note that the `x` value in this case is the input to the activation on the first pass,
      and the weight and activation data come from pre-trained full-precision 
      models. I believe it's best to compute these elsewhere and port them in here. 
      Also, see section 4 of https://arxiv.org/pdf/2104.00903.pdf for more info.
    - This needs to be a subclass of Layer so that the variables can be
      trainable.
    - Scale parameters are constants here, NOT per-channel tensors
  
  """
    
  def __init__(self,
                bits,
                u_initial_value,
                l_initial_value,
                delta,
                activation=False):
    
    super(quantized_ewgs, self).__init__()

    self.bits = bits
    self.activation = activation
    self.delta = delta

    # create weights for u and l with constant initial values
    self.u = self.add_weight(
      shape=(),
      initializer=tf.constant_initializer(u_initial_value), 
      trainable=True)
    self.l = self.add_weight(
      shape=(),
      initializer=tf.constant_initializer(l_initial_value), 
      trainable=True)

  def get_xn(self, x):

    scaled = (x - self.l) / (self.u - self.l)
    return K.clip(scaled, 0.0, 1.0)
  
  @tf.custom_gradient
  def get_xq(self, xn):

    xq = tf.round(xn * (2**self.bits - 1)) / (2**self.bits - 1)
    def grad(gxq):

      scale = 1 + self.delta * tf.sign(gxq) * (xn - xq)

      return gxq * scale
      
    return xq, grad

  def __call__(self, x):

    xn = self.get_xn(x)
    xq = self.get_xq(xn)

    if self.activation:
      res = xq
    else:
      res = 2 * (xq - 0.5)
    return res
  
  def max(self):

    return 1.0
  
  def min(self):

    return 0.0 if self.activation else -1.0
  

class QActivationEWGS(Layer, PrunableLayer):
  """
  Quantized activation layer with EWGS quantization.
  
  The main differences between this and QActivation are
    - This layer takes in EWGS quantizer parameters directly, rather than a 
      quantizer object/string
    - There is a trainable scale parameter for the output of the activation. The
      initialization logic for this parameter is here:
      https://github.com/cvlab-yonsei/EWGS/blob/main/ImageNet/custom_modules.py#L121
      Since we don't have access to the input to the activation on the first
      pass, and this value is approximately 1, we initialize it to 1. Can be more precise
      if necessary.
      
  """
  
  def __init__(self, *quantizer_args, **quantizer_kwargs):

    super(QActivationEWGS, self).__init__()

    quantizer_kwargs['activation'] = True
    self.quantizer = quantized_ewgs(*quantizer_args, **quantizer_kwargs)
    self.scale = self.add_weight(shape=(), initializer='ones', trainable=True)

  def call(self, inputs):

    return self.quantizer(inputs) * self.scale

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []
  

if __name__ == '__main__':

  from tensorflow import keras
  from tensorflow.keras import layers
  import qkeras as qk

  def create_gpu_strategy():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    return strategy

  strategy = create_gpu_strategy()

  with strategy.scope():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    test = False

    if test:
      # subsample 1% data
      x_train = x_train[:600]
      y_train = y_train[:600]
      x_test = x_test[:100]
      y_test = y_test[:100]

    def ewgs_quantizer():

      return quantized_ewgs(4, -1.0, 1.0, 0.1)

    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Rescaling(1./255)(inputs)
    x = qk.QConv2D(
        filters=32, kernel_size=3, kernel_quantizer=ewgs_quantizer(),
        bias_quantizer=ewgs_quantizer())(x)
    x = QActivationEWGS(4, 0.0, 1.0, 0.1)(x)
    x = layers.Flatten()(x)
    x = qk.QDense(10, kernel_quantizer=ewgs_quantizer(),
        bias_quantizer=ewgs_quantizer())(x)
    outputs = layers.Activation("softmax", name="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='rmsprop',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    print(model.summary())

    history = model.fit(x_train, y_train, epochs=1, batch_size=64, 
                        validation_data=(x_test, y_test))
  