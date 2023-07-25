'''
  Frechet Inception Distance
  --------------------------
  GAN evaluation to determine the Fedility & Diversity
  > Statistical Parameters are
      > mean
      > covarience
'''


### importing the modules
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.applications import InceptionV3

class Frechet_Inception_Distance():
  '''
    Frechet Ineption Distance is a metric to evaluate the GANS.
    -----------------------------------------------------------
    Args: 
      model (default None) :  Any model which will extract features 
      feature_vectors (default True) : True/False giving a feature maps in a flatten way.
                                       If a feature_vectors is False and model is None then feautre_vectors can be extracted from IneptionV3 model.

      Input_shape : Default (256, 256, 3) // input shape when model is None and feature_vectors are False then u must need to pass the Input_shape or the default input shape will take cares.

  '''

  def __init__(self, model = None, feature_vetors = True, Input_shape = (256,256,3)):

    self.feature_vectors = feature_vectors
    self.model = model
    if(model is None and not feature_vectors):
      assert Input_shape is not None
      self.model = model  if model else self.inception_model(input_shape)


  def covarience(self, input_):
    covar = tfp.stats.covariance(input_)
    return covar

  def means(self, input_):
    mean_ = tf.math.reduce_mean(input_, axis=0)
    return mean_


  def square_root(self, m):
    out = scipy.linalg.sqrtm(m)

  def call(self, y_pred, y_true):
    ## calculating means
    if(not self.feature_vectors):
      y_pred = self.model.predict(y_pred)
      y_true = self.model.predict(y_true)
    

    y_pre_mean = self.means(y_pred)
    y_true_mean = self.means(y_true)

    ## Calculating variences
    y_pre_var = self.covarience(y_pred)
    y_true_var = self.covarience(y_true)

    ### getting all the shape of the intermediate values
    # print("The shape of the PREDICT : {}, TRUE : {}".format(y_pre_mean.shape, y_true_mean.shape))
    # print("The shape of the TRUES : {}, TRUE : {}".format(y_pre_var.shape, y_true_var.shape))
    # print( tf.linalg.trace(2*tf.sqrt( y_pre_var @ y_true_var)))
    fid = tf.square(tf.norm(y_pre_mean - y_true_mean)) + tf.linalg.trace(y_pre_var) +tf.linalg.trace( y_true_var) - tf.linalg.trace(2*tf.sqrt(y_pre_var @ y_true_var))
    return fid


  def inception_model(self, input_shape = (256, 256, 3)):
    base_model = tf.keras.applications.InceptionV3(
                                                  include_top=True,
                                                  weights="imagenet",
                                                  input_shape=input_shape
                                              )
    inputs = tf.keras.layers.Input(input_shape)
    x = base_model(inputs, training = False)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)

    self.model = tf.keras.Model(inputs = inputs, outputs = outputs)