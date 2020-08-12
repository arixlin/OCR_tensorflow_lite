# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 18:00
# @Author  : lin

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

import math
import os
import os.path as osp
import time

import cv2
import glob
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm


def resize_image(image, image_short_side=640):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
    
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2
    
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=500, box_thresh=0.7):
    """
    _bitmap: single map with shape (H, W),
        whose values are binarized as {0, 1}
    """
    min_size = 1

    bitmap = np.squeeze(bitmap)
    assert len(bitmap.shape) == 2
    height, width = bitmap.shape
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), max_candidates)
    # boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
    boxes = []
    # scores = np.zeros((num_contours,), dtype=np.float32)
    scores = []
    rects = []
    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = box_score_fast(pred, contour)
        if box_thresh > score:
            continue

        box = unclip(points, unclip_ratio=1.5).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)

        if sside < min_size + 2:
            continue
        box = np.array(box)
        # box = np.array(contour_to_box(contour))
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()
        
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        # boxes[index, :, :] = box.astype(np.int16)
        boxes.append(box.astype(np.int16))
        scores.append(score)
    return boxes, scores, bitmap


def main():
    BOX_THRESH = 0.5
    mean = np.array([103.939, 116.779, 123.68])
    img_dir = 'datasets/test/input'
    img_names = os.listdir(img_dir)

    dbnet_interpreter = interpreter_wrapper.Interpreter(model_path='models/dbnet_mobilenetv3.tflite')
    dbnet_interpreter.allocate_tensors()
    dbnet_input_details = dbnet_interpreter.get_input_details()
    dbnet_output_details = dbnet_interpreter.get_output_details()

    for img_name in tqdm(img_names):
        img_path = osp.join(img_dir, img_name)
        image = cv2.imread(img_path)
        src_image = image.copy()
        h, w = image.shape[:2]
        image = resize_image(image)
        image = image.astype(np.float32)
        image -= mean
        image_input = np.expand_dims(image, axis=0)
        image_input_tensor = tf.convert_to_tensor(image_input)
        start_time = time.time()

        dbnet_interpreter.resize_tensor_input(dbnet_input_details[0]['index'], image_input_tensor.shape)
        dbnet_interpreter.allocate_tensors()
        dbnet_interpreter.set_tensor(dbnet_input_details[0]['index'], image_input_tensor)
        dbnet_interpreter.invoke()
        p = dbnet_interpreter.get_tensor(dbnet_output_details[0]['index'])[0]

        end_time = time.time()
        print("time: ", end_time - start_time)

        bitmap = p > 0.3
        boxes, scores, bitmap = polygons_from_bitmap(p, bitmap, w, h, box_thresh=BOX_THRESH)
        # print(boxes.shape)
        for box in boxes:
            cv2.polylines(src_image, np.int32([box]), 1, (0, 255, 0), 1)
        image_fname = osp.split(img_path)[-1]
        cv2.imwrite('datasets/test/output/' + image_fname, src_image)
        cv2.imwrite('datasets/test/output/bitmap' + image_fname, bitmap * 255)

if __name__ == '__main__':
    main()
