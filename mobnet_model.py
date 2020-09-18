from keras.applications.mobilenet import MobileNet
from keras.layers import Dense
from keras.initializers import constant
from keras.models import Model
from keras.layers import Dropout, Conv2D, UpSampling2D, Reshape, Softmax, BatchNormalization, Add, Dense, Concatenate, Input
from keras.backend import expand_dims
from keras.activations import relu
import numpy as np
import math
import params
def upsample_block(input, num_filt, res, drp):
    Up = UpSampling2D(size=(2, 2))(input)

    conv1 = Conv2D(num_filt, (3, 3), activation=relu, padding='same')(Up)
    drop1 = BatchNormalization()(conv1)
    if drp == 1:
        drop1 = Dropout(0.5)(drop1)
    conv2 = Conv2D(num_filt, (3, 3), activation=relu, padding='same')(drop1)
    drop2 = BatchNormalization()(conv2)
    if drp == 1:
        drop2 = Dropout(0.5)(drop2)

    shortcut = Conv2D(num_filt, (1, 1), activation=relu, padding='same')(Up)
    shortcut = BatchNormalization()(shortcut)

    if res == 1:
        x = Add()([drop2, shortcut])
    else:
        x = drop2
    return x


def output_block(input, drp):
    if drp == 1:
        input = Dropout(0.5)(input)
    conv_out = Conv2D(1, (3, 3), activation=relu, padding='same')(input)
    reshape_1 = Reshape((1, input.shape[1] * input.shape[2]))(conv_out)
    sm = Softmax()(reshape_1)
    reshape_2 = Reshape((input.shape[1], input.shape[2]))(sm)

    return reshape_2

def mask_block(input, size, std):
    mask = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            mask[i,j] = math.exp(-((size//2 - i)**2 + (size//2 - j)**2 )/std)
    mask = mask/np.sum(np.sum(mask))
    input = Reshape((input.shape[1],input.shape[2],1))(input)
    conv_out = Conv2D(1,(size,size), padding='same', use_bias=False, kernel_initializer=constant(mask), trainable=False)(input)
    conv_out = Reshape((input.shape[1],input.shape[2]))(conv_out)
    return conv_out

def dense_hp_model(input):
    d1 = Dense(512)(input)
    drop1 = Dropout(0.5)(d1)
    d2 = Dense(672)(drop1)
    drop2 = Dropout(0.5)(d2)
    out = Reshape((3,7,32))(drop2)
    length = 6
    return out, length


def ups_model(input):
        conv1 = Conv2D(512, (3, 3), activation=relu, padding='same')(input)
        drop1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.5)(drop1)
        conv2 = Conv2D(512, (3, 3), activation=relu, padding='same')(drop1)
        drop2 = BatchNormalization()(conv2)
        drop2 = Dropout(0.5)(drop2)
# decoder network
        Up1 = upsample_block(drop2, 256, res=1, drp=1)
        Up2 = upsample_block(Up1, 128, res=1, drp=1)
        Up3 = upsample_block(Up2, 64, res=1, drp=1)
        Up4 = upsample_block(Up3, 32, res=1, drp=1)
        Up5 = upsample_block(Up4, 16, res=1, drp=1)

## outputs
        out_1 = output_block(drop2, drp=0)
        out_2 = output_block(Up1, drp=0)
        out_3 = output_block(Up2, drp=0)
        out_4 = output_block(Up3, drp=0)
        out_5 = output_block(Up4, drp=0)
        out_6 = output_block(Up5, drp=0)

        return [out_1, out_2, out_3, out_4, out_5, out_6]

def mask_output(outs):
        masked = []
        size = [3, 5, 9, 19, 31, 63]
        std = [1, 2, 10, 50, 100, 300]
        for i, out in enumerate(outs):
            mask_out = mask_block(out, size=size[i], std=std[i])
            masked.append(mask_out)
        return masked

class model(object):
    def __init__(self):
        self.len_hpencoder = 0
        self.len_eyeencoder = 0
        if params.model.eye_en:
            model = MobileNet(weights=None, input_shape=params.data.IMG_SHAPE, include_top=False)
            self.len_eyeencoder = len(model.layers)
            resout = model.output

        if params.model.hp_en:
            in2 = Input(shape=(6,))
            model2, self.len_hpencoder = dense_hp_model(in2)

        if params.model.hp_en and params.model.eye_en:
            encout = Concatenate()([model2, resout])
            model_in = [model.input,in2]
        elif params.model.hp_en:
            encout = model2
            model_in = in2
        elif params.model.eye_en:
            encout = resout
            model_in = model.input
        else:
            print('At least one of the hp and eye branch should be enabled')
            return -1

        outs = ups_model(encout)
        model_bare = Model(inputs=model_in, outputs=outs)
        self.len_decoder = len(model_bare.layers) - self.len_eyeencoder - self.len_hpencoder
        self.model_out = outs
        masked = mask_output(outs)
        self.model_bare = model_bare
        self.model = Model(inputs=model_in, outputs=masked)


    def freeze_ups(self):
        for lr in range(self.len_eyeencoder + self.len_hpencoder, self.len_decoder):
            self.model.layers[lr].trainable = False

    def unfreeze_ups(self):
        for lr in range(self.len_eyeencoder + self.len_hpencoder,self.len_decoder):
            self.model.layers[lr].trainable = True

    def freeze_down(self):
        for lr in range(0, self.len_eyeencoder + self.len_hpencoder):
            self.model.layers[lr].trainable = False

    def unfreeze_down(self):
        for lr in range(0, self.len_eyeencoder + self.len_hpencoder):
            self.model.layers[lr].trainable = True
