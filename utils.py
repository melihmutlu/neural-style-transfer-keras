""" Error calculation funtions and some other utilities"""

from keras import backend as K
from keras.preprocessing.image import *

# Save image method in case of using old versions of Keras
def save_img(path, x, data_format=None, file_format=None, scale=True, **kwargs):
    img = array_to_img(x, data_format=data_format, scale=scale)
    img.save(path, format=file_format, **kwargs)

def gram_matrix(x):
    if  K.ndim(x) != 3 :
        raise Exception("Dimension error") 
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, generated, num_rows, num_cols):
    if not (K.ndim(style) == 3 and  K.ndim(generated) == 3):
        raise Exception("Dimension error")
    
    S = gram_matrix(style)
    G = gram_matrix(generated)
    channels = 3
    size = num_rows * num_cols

    return K.sum(K.square(S - G)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(content, generated):
    return K.sum(K.square(generated - content))

def total_loss(x, num_rows, num_cols):
    if not K.ndim(x) == 4:
        raise Exception("Dimension error")
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :num_rows - 1, :num_cols - 1] - x[:, :, 1:, :num_cols - 1])
        b = K.square(x[:, :, :num_rows - 1, :num_cols - 1] - x[:, :, :num_rows - 1, 1:])
    else:
        a = K.square(x[:, :num_rows - 1, :num_cols - 1, :] - x[:, 1:, :num_cols - 1, :])
        b = K.square(x[:, :num_rows - 1, :num_cols - 1, :] - x[:, :num_rows - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))
