# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 10:00
# @Author  : lin

import argparse
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images', type=str, required=True, 
                    help='Image file or folder path.')
# parser.add_argument('-t', '--table_path', type=str, required=True, 
#                     help='The path of table file.')


def read_img_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=args.img_channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (32, 100))
    return img


img = read_img_and_preprocess(args.images)
imgs = tf.expand_dims(img, 0)

# with open(args.table_path, 'r') as f:
#     inv_table = [char.strip() for char in f]

from tensorflow.lite.python import interpreter as interpreter_wrapper
import numpy as np

#crnn
crnn_interpreter = interpreter_wrapper.Interpreter(model_path='converted_model.tflite')
crnn_interpreter.allocate_tensors()
crnn_input_details = crnn_interpreter.get_input_details()
crnn_output_details = crnn_interpreter.get_output_details()
crnn_interpreter.resize_tensor_input(crnn_input_details[0]['index'], imgs.shape)
crnn_interpreter.allocate_tensors()
# print(crnn_input_details)

crnn_interpreter.set_tensor(crnn_input_details[0]['index'], imgs)
crnn_interpreter.invoke()
y_pred = crnn_interpreter.get_tensor(crnn_output_details[0]['index'])
print(y_pred.shape)


#ctc
ctc_interpreter = interpreter_wrapper.Interpreter(model_path='decoded.tflite')
ctc_interpreter.allocate_tensors()
ctc_input_details = ctc_interpreter.get_input_details()
ctc_output_details = ctc_interpreter.get_output_details()
ctc_interpreter.resize_tensor_input(ctc_input_details[0]['index'], y_pred.shape)
ctc_interpreter.allocate_tensors()

ctc_interpreter.set_tensor(ctc_input_details[0]['index'], y_pred)
ctc_interpreter.invoke()
output = ctc_interpreter.get_tensor(ctc_output_details[0]['index'])
print(output)