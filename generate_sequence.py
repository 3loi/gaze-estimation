# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:15:38 2020

@author: sxj146830
"""
# data generator for continuous data from 10 training subjects
# from generate_data_cont_class import DataGenerator
from keras.utils import Sequence
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os
import math
from scipy.ndimage import convolve
import params

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def mask_mat(size, std):
    mask = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            mask[i,j] = math.exp(-((size//2 - i)**2 + (size//2 - j)**2 )/std)
    mask = mask/np.sum(np.sum(mask))
    return mask

class data_generator(Sequence):

    def __init__(self, x_set, x2_set, bb_set, y_set, batch_size):
        self.x, self.x2, self.y, self.bb = x_set, x2_set, y_set, bb_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        imsize1 = params.data.IMG_SHAPE[1]
        imsize2 = params.data.IMG_SHAPE[0]

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_bb = self.bb[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = np.zeros((len(batch_x), imsize2, imsize1, 3))
        batch_hp = np.zeros((len(batch_x),6))
        batch_angles = [None]*6
        mask_size = [3, 5, 9, 19, 31, 63]
        mask_std = [1, 2, 10, 50, 100, 300]
        for op in range(6):
            batch_angles[op] = np.zeros((len(batch_y), 3*(2**op), 7*(2**op)))
        for i in range(0, len(batch_x)):
            if params.model.eye_en:
                drop_eye = rand(0,1)<0.2
                im_or = Image.open(batch_x[i])
                r_margin = rand(10, 25)
                bb = batch_bb[i]
                bb_tuple = tuple(bb[0] - r_margin) + tuple(bb[1] + r_margin)
                im = im_or.crop(bb_tuple)

                iw, ih = im.size
                scale = min(imsize1 / iw, imsize2 / ih)
                r_scale = rand(.7, 1.3)

                nw = int(iw * scale * r_scale)
                nh = int(ih * scale * r_scale)
                im = im.resize((nw, nh), Image.BICUBIC)
                new_im = Image.new('RGB', (imsize1, imsize2), (128, 128, 128))
                # im = im.resize((imsize1,imsize2))
                new_im.paste(im, ((imsize1 - nw) // 2, (imsize2 - nh) // 2))
                new_im = np.array(new_im).astype(np.float32) / 255

                # distort image
                hue = 0.1
                sat = 1.5
                val = 1.5
                hue = rand(-hue, hue)
                sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
                val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
                x = rgb_to_hsv(new_im)
                x[..., 0] += hue
                x[..., 0][x[..., 0] > 1] -= 1
                x[..., 0][x[..., 0] < 0] += 1
                x[..., 1] *= sat
                x[..., 2] *= val
                x[x > 1] = 1
                x[x < 0] = 0
                new_im = hsv_to_rgb(x)  # numpy array, 0 to 1
                if drop_eye == 0:
                    batch_imgs[i, :, :, :] = new_im
            if params.model.hp_en:
                batch_hp[i] = batch_x2[i]
            angles = batch_y[i]
            for op in range(params.data.OUT_Layers):
                mask = mask_mat(mask_size[op], mask_std[op])
                indv = int(angles[1] * (params.data.OUT_SHAPE_1[0]*(2**op))/ 1080)
                indh = int(angles[0] * (params.data.OUT_SHAPE_1[1]*(2**op)) / 1920)
                target = np.zeros((params.data.OUT_SHAPE_1[0]*(2**op) + mask_size[op] - 1, params.data.OUT_SHAPE_1[1]*(2**op) + mask_size[op] -1))
                target[indv:indv+mask_size[op],indh:indh+mask_size[op]] = mask
                batch_angles[op][i] = target[(mask_size[op]-1)//2:-(mask_size[op]-1)//2,(mask_size[op]-1)//2:-(mask_size[op]-1)//2]
        if params.model.hp_en and params.model.eye_en:
            model_in = [batch_imgs,batch_hp]
        elif params.model.hp_en:
            model_in = batch_hp
        elif params.model.eye_en:
            model_in = batch_imgs
        else:
            print('At least one of the hp and eye branch should be enabled')
            return -1
        return model_in, batch_angles
