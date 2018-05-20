from __future__ import print_function
import os
# Using Theano as backend. Comment this line if you want to user Tensorflow instead
os.environ["KERAS_BACKEND"] = "theano"
from keras.preprocessing.image import load_img, array_to_img, img_to_array
import numpy as np
import argparse
from keras.applications import vgg19
from keras import backend as K

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
total_variation_weight = 1.0
style_weight = 1.0
content_weight = 0.025

# Dimensions of images.
width, height = load_img(content_image_path).size
num_rows = 400
num_cols = int(width * img_nrows / height)

# Tensor representations of input images
content_image = K.variable(proc.preprocess_image(content_image_path, num_rows, num_cols))
style_image = K.variable(proc.preprocess_image(style_image_path, num_rows, num_cols))

# Placeholder for utput image
if K.image_data_format() == 'channels_first':
    generated_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    generated_image = K.placeholder((1, img_nrows, img_ncols, 3))

#VGG model for the network that uses Imagenet weights
model = vgg19.VGG19(input_tensor= K.concatenate([content_image, style_image, generated_image], axis=0), 
                    weights='imagenet', 
                    include_top=False)

