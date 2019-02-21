import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
import mxnet as mx
import numpy as np
from helpers import get_model, predict

class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # get context - cpu
        self._ctx = mx.cpu()

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # # Run inference
        im =  self._imageProcessor._im
        result_shape = self._imageProcessor._result_shape
        # model loading needs to happen after input has been prosessed as
        # instantianting it requires the size of the input image.
        model = get_model(self._ctx, 'model/model.onnx', im)
        conf,result_img,blended_img,raw = predict(inputAsNpArr, result_shape, model, im)
        return np.array(result_img)
