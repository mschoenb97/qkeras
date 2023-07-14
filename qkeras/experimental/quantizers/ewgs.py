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

from qkeras import BaseQuantizer

class quantized_ewgs(BaseQuantizer):
  """Quantizer for element-wise gradient scaling.
  
  Notes:
    - We're treating delta as a hyperparameter here
    - In the EWGS code, u and l are initialized based on these LOCs:
      https://github.com/cvlab-yonsei/EWGS/blob/main/ImageNet/custom_modules.py#L109-L117
      Note that the `x` value in this case is the input to the activation on the first pass.
      Afaik, it's not possible to use this information for initialization in Keras.
      Tbd what the best way to initialize these values is.
  
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

    self.u = tf.Variable(u_initial_value, trainable=True)
    self.l = tf.Variable(l_initial_value, trainable=True)

  def get_xn(self, x):

    scaled = (x - self.l) / (self.u - self.l)
    return K.clip(scaled, 0, 1)
  
  @tf.custom_gradient
  def get_xq(self, xn):

    xq = tf.round(xn * (2**self.bits - 1)) / (2**self.bits - 1)
    def grad(gxq):

      scale = 1 + self.delta * tf.sign(gxq) * (xn - xq)

      return gxq * scale
      
    return xq, grad

  def call(self, x):

    xn = self.get_xn(x)
    xq = self.get_xq(xn)

    if self.activation:
      res = xq
    else:
      res = 2 * (xq - 0.5)
    return res
  

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
      pass, and this value is approximately 1, we initialize it to 1.
      
  """
  
  def __init__(self, *quantizer_args, **quantizer_kwargs):

    super(QActivationEWGS, self).__init__()

    self.quantizer = quantized_ewgs(*quantizer_args, **quantizer_kwargs, activation=True)
    self.scale = self.add_weight(
        name="scale",
        shape=(1,),
        initializer="ones",
        trainable=True)

  def call(self, inputs):
    return self.quantizer(inputs) * self.scale

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []
  