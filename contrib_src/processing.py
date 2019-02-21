from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers import preprocess
import cv2
import numpy as np
import os
from skimage import transform as trans
from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json
import mxnet as mx

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # switches PIL to cv2
            self._im  = np.array(image)
            if len(self._im.shape) <= 2:
                raise IOError("Image format not supported for preprocessing.")
            # set output shape (same as input shape)
            self._result_shape = [self._im.shape[0],self._im.shape[1]]
            # set rgb mean of input image (used in mean subtraction)
            self._rgb_mean = cv2.mean(self._im)
            pre = preprocess(self._im, self._rgb_mean)
            return pre
        else:
            raise IOError("Image Type not supported for preprocessing.")

    def _preprocessAfterConversionToNumpy(self, npArr):
        mxArr = mx.nd.array(npArr)
        return mxArr
