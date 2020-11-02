# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 10:00
# @Author  : lin

import argparse
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

from decoder import Decoder

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images', type=str, required=True, 
                    help='Image file or folder path.')

args = parser.parse_args()

def read_img_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (32, 120))
    return img


img = read_img_and_preprocess(args.images)
imgs = tf.expand_dims(img, 0)


from tensorflow.lite.python import interpreter as interpreter_wrapper
import numpy as np

# #crnn
# crnn_interpreter = interpreter_wrapper.Interpreter(model_path='models/crnn_vgg_ctc.tflite')
# crnn_interpreter.allocate_tensors()
# crnn_input_details = crnn_interpreter.get_input_details()
# crnn_output_details = crnn_interpreter.get_output_details()
# crnn_interpreter.resize_tensor_input(crnn_input_details[0]['index'], imgs.shape)
# crnn_interpreter.allocate_tensors()
# # print(crnn_input_details)

# crnn_interpreter.set_tensor(crnn_input_details[0]['index'], imgs)
# crnn_interpreter.invoke()
# y_pred = crnn_interpreter.get_tensor(crnn_output_details[0]['index'])
# print(y_pred)


#ctc
crnn_interpreter = interpreter_wrapper.Interpreter(model_path='models/crnn.tflite')
crnn_input_details = crnn_interpreter.get_input_details()
crnn_output_details = crnn_interpreter.get_output_details()
crnn_interpreter.resize_tensor_input(crnn_input_details[0]['index'], imgs.shape)
crnn_interpreter.allocate_tensors()
crnn_interpreter.set_tensor(crnn_input_details[0]['index'], imgs)
crnn_interpreter.invoke()
y_pred = crnn_interpreter.get_tensor(crnn_output_details[0]['index'])

ctc_interpreter = interpreter_wrapper.Interpreter(model_path='models/ctc_decoder_greedy.tflite')
ctc_input_details = ctc_interpreter.get_input_details()
ctc_output_details = ctc_interpreter.get_output_details()
ctc_interpreter.resize_tensor_input(ctc_input_details[0]['index'], y_pred.shape)
ctc_interpreter.allocate_tensors()
ctc_interpreter.set_tensor(ctc_input_details[0]['index'], y_pred)
ctc_interpreter.invoke()
output = ctc_interpreter.get_tensor(ctc_output_details[0]['index'])
print(output)

with open('vocabulary.txt', 'r') as f:
    inv_table = [char.strip() for char in f]

decoder = Decoder(inv_table)

print(decoder.map2string(output.astype(int)))



