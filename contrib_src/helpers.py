import mxnet as mx
import cv2 as cv
import numpy as np
import os
from PIL import Image
import math
from collections import namedtuple

from mxnet.contrib.onnx import import_model

import cityscapes_labels



def preprocess(im, rgb_mean):
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    return test_img


def get_palette():
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def predict(imgs, result_shape, mod, im):
    # get input and output dimensions
    result_height, result_width = result_shape
    _, _, img_height, img_width = imgs.shape
    # set downsampling rate
    ds_rate = 8
    # set cell width
    cell_width = 2
    # number of output label classes
    label_num = 19

    # Perform forward pass
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([imgs]),is_train=False)
    labels = mod.get_outputs()[0].asnumpy().squeeze()

    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width),:int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])

    # get softmax output
    softmax = labels

    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results

    # comput confidence score
    confidence = float(np.max(softmax, axis=0).mean())

    # generate segmented image
    result_img = Image.fromarray(colorize(raw_labels)).resize(result_shape[::-1])

    # generate blended image
    blended_img = Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0))

    return confidence, result_img, blended_img, raw_labels


def get_model(ctx, model_path, im):
    # import ONNX model into MXNet symbols and params
    sym,arg,aux = import_model(model_path)
    # define network module
    mod = mx.mod.Module(symbol=sym, data_names=['data'], context=ctx, label_names=None)
    # bind parameters to the network
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, im.shape[0], im.shape[1]))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg, aux_params=aux,allow_missing=True, allow_extra=True)
    return mod
