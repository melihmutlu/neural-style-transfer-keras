from __future__ import print_function
import os
# Using Theano as backend. Comment this line if you want to user Tensorflow instead
os.environ["KERAS_BACKEND"] = "theano"
from keras.preprocessing.image import load_img, array_to_img, img_to_array
import numpy as np
import argparse
from keras.applications import vgg19
from keras import backend as K
import utils as utils

parser = argparse.ArgumentParser(description='CMPE 462 Term Project - Neural Style transfer')
parser.add_argument('content_image_path', metavar='base', type=str,
                    help='Path to the content image.')
parser.add_argument('style_image_path', metavar='ref', type=str,
                    help='Path to the style image')
parser.add_argument('result_file', metavar='res_prefix', type=str,
                    help='Result file names')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations')

args = parser.parse_args()
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_file = args.result_file
iterations = args.iter

# Weights
total_weight = 1.0
style_weight = 1.0
content_weight = 0.025

# Dimensions of images.
width, height = load_img(content_image_path).size
num_rows = 400
num_cols = int(width * num_rows / height)

# Tensor representations of input images
content_image = K.variable(proc.preprocess_image(content_image_path, num_rows, num_cols))
style_image = K.variable(proc.preprocess_image(style_image_path, num_rows, num_cols))

# Placeholder for utput image
if K.image_data_format() == 'channels_first':
    generated_image = K.placeholder((1, 3, num_cols, num_rows))
else:
    generated_image = K.placeholder((1, num_rows, num_cols, 3))

#VGG model for the network that uses Imagenet weights
model = vgg19.VGG19(input_tensor= K.concatenate([content_image, style_image, generated_image], axis=0), 
                    weights='imagenet', 
                    include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
content_image_features = layer_features[0, :, :, :]
generated_features = layer_features[2, :, :, :]
loss += content_weight * util.content_loss(content_image_features,
                                      generated_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    generated_features = layer_features[2, :, :, :]
    sl = util.style_loss(style_features, generated_features, num_rows, num_cols)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_weight * utils.total_variation_loss(generated_image, num_rows, num_cols)

# Gradients of the generated image wrt the loss
grads = K.gradients(loss, generated_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([generated_image], outputs)