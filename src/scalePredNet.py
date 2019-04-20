from src.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Dropout
from keras.engine import Input
from keras.models import Model

class scalePredNet(object):
    def __init__(self):
        return
    
    def createNet(self, modelPath='../models/cellSizePred_vgg16_model/checkpoints/weights.239-0.10.hdf5', imgShape = [64, 64]):
        vgg16_conv5_model = VGG16(include_top=False)
        inputImg = Input(shape=[imgShape[0], imgShape[1], 3], name = 'inputImg')

        vgg16_conv5_val = vgg16_conv5_model(inputImg)
        x = Flatten(name='flatten')(vgg16_conv5_val)
        x = Dense(4096, activation=None, name='fc1')(x)
        x = Dense(4096, activation=None, name='fc2')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        #Create your own model 
        pretrained_model = Model(inputImg, x)
        #pretrained_model.load_weights('../models/cellSizePred_vgg16_model/checkpoints/weights.1494-0.07.hdf5')
        pretrained_model.load_weights(modelPath)
        return pretrained_model
    
    