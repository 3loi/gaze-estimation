# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:15:38 2020

@author: sxj146830
"""
# data generator for continuous data from 10 training subjects
import numpy as np
import os
import pickle
import scipy.io as sio
import math
import params

def subject_data(data_subj='20190522001'):
    face_fold = '/home/sumit/continuous_7/extracted_cont_face/'
    label_fold = '/home/sumit/continuous_7/extracted_cont_april/'
    hp_fold = '/home/sumit/hp_extract_cont/calibrated_hp/'
    tr_list = os.listdir(face_fold + data_subj)
    # angles range
    # max_h = 1.5
    # min_h = -1.5
    # max_v = 0.4
    # min_v = -1.0
    # # angles range
    # max_h = 1.0
    # min_h = -0.6
    # max_v = 0.125
    # min_v = -0.050

    imgs = []
    eye_bb = []
    angles = []
    headpose = []
    hom_rd2bk = sio.loadmat('/home/sumit/hp_extract_cont/road2back_ref')['hom_rd2bk']
    with open(face_fold + data_subj + '.pkl', 'rb') as f:
        fan_out = pickle.load(f)
    # read data.txt
    for subject in tr_list:
        label_path = label_fold + data_subj + '/' + subject + '.csv'
        hp_path = hp_fold + data_subj + '/' + subject + '.mat'
        hpose = sio.loadmat(hp_path)['hp']
        op_list = np.genfromtxt(label_path)
        breaks = np.where(op_list[:, 0] == 2222)
        prev = 0
        count = 1
        for line in breaks[0]:
            detected = op_list[prev:line]
            prev = line
            tag = np.where((detected[:, 0] == 100) | (detected[:, 0] == 300))
            if tag[0].shape[0] != 0:
                img_loc = face_fold + data_subj + '/' + subject + '/filename' + str(count).zfill(4) + '.jpg'
                img_key = 'D:\\extracted_cont_face\\' + data_subj + '\\' + subject + '\\filename' + str(
                    count).zfill(4) + '.jpg'
                if os.path.exists(img_loc) and fan_out[img_key] and np.sum(np.isnan(hpose[count-1]))==0:
                    lm = fan_out[img_key]
                    # if more than one face is detected find the one closest to the pxl 1200,600
                    # fix this later to be more robust and make sure only the driver is selected
                    idx = np.argmin(np.linalg.norm(np.mean(lm, axis=1) - np.array([1200, 600]), axis=1))
                    lm = lm[idx]
                    bb = [np.min(lm[36:48, :], axis=0), np.max(lm[36:48, :], axis=0)]
                    loc_road = [-detected[tag[0], 4],detected[tag[0], 5],-detected[tag[0], 3],1]
                    loc_back = np.dot(loc_road,hom_rd2bk)
                    head_loc = hpose[count-1,0:3]
                    gaze_vec = loc_back[0:3] - head_loc
                    norm_vec = gaze_vec/np.linalg.norm(gaze_vec)
                    theta = math.atan2(norm_vec[0],-norm_vec[2])
                    phi = math.atan2(norm_vec[1],np.linalg.norm([gaze_vec[0],gaze_vec[2]]))

                    hori = 1920 * (theta - params.data.min_h) / (params.data.max_h - params.data.min_h)
                    if hori < 0:
                        hori = 0
                    if hori > 1919:
                        hori = 1919
                    vert = 1080 * (phi - params.data.min_v) / (params.data.max_v - params.data.min_v)
                    if vert < 0:
                        vert = 0
                    if vert > 1079:
                        vert = 1079
                    # x = 960 - 1000 * (detected[tag[0], 4] / detected[tag[0], 3])
                    # y = 540 - 1000 * (detected[tag[0], 5] / detected[tag[0], 3])
                    if ~np.isnan(hori) and ~np.isnan(vert):
                        imgs.append(img_loc)
                        eye_bb.append(bb)
                        angles.append(np.array([hori,vert]))
                        headpose.append(hpose[count-1])
            count = count + 1

    return imgs, eye_bb, angles, headpose


def list_of_data():
    va_subj = 1

    list_of_subjects = [
        '20190522001',
        '20190530001',
        '20190611001',
        '20190614001',
        '20190621001',
        '20190709001',
        '20190710001',
        '20191003001',
        '20191120001',
        '20191125001'
    ]

    train_images = []
    train_bb = []
    train_angles = []
    train_hp = []
    for subject in list_of_subjects[0:-va_subj]:
        img, bb, ang, hp = subject_data(data_subj=subject)
        train_images = train_images + list(img)
        train_bb = train_bb + list(bb)
        train_angles = train_angles + list(ang)
        train_hp = train_hp + list(hp)
        print(subject + " added")

    val_images = []
    val_bb = []
    val_angles = []
    val_hp = []
    for subject in list_of_subjects[-va_subj:]:
        img, bb, ang, hp = subject_data(data_subj=subject)
        val_images = val_images + list(img)
        val_bb = val_bb + list(bb)
        val_angles = val_angles + list(ang)
        val_hp = val_hp + list(hp)
        print(subject + " added")

    return train_images, train_bb, train_angles, train_hp, val_images, val_bb, val_angles, val_hp

from PIL import Image
def return_batch(x_set, x2_set, bb_set, y_set, batch_size, idx):
    imsize2 = params.data.IMG_SHAPE[0]
    imsize1 = params.data.IMG_SHAPE[1]
    batch_x = x_set[idx * batch_size:(idx + 1) * batch_size]
    batch_x2 = x2_set[idx * batch_size:(idx + 1) * batch_size]
    batch_y = y_set[idx * batch_size:(idx + 1) * batch_size]
    batch_bb = bb_set[idx * batch_size:(idx + 1) * batch_size]
    batch_imgs = np.zeros((len(batch_x), imsize2, imsize1, 3))
    batch_hp = np.zeros((len(batch_x), 6))
    batch_angles = [None] * params.data.OUT_Layers
    for op in range(params.data.OUT_Layers):
        batch_angles[op] = np.zeros((len(batch_y), params.data.OUT_SHAPE_1[0] * (2 ** op), params.data.OUT_SHAPE_1[1] * (2 ** op)))

    for i in range(len(batch_x)):
        im_or = Image.open(batch_x[i])
        r_margin = 15
        bb = batch_bb[i]
        bb_tuple = tuple(bb[0] - r_margin) + tuple(bb[1] + r_margin)
        im = im_or.crop(bb_tuple)

        iw, ih = im.size
        scale = min(imsize1 / iw, imsize2 / ih)
        r_scale = 1

        nw = int(iw * scale * r_scale)
        nh = int(ih * scale * r_scale)
        im = im.resize((nw, nh), Image.BICUBIC)
        new_im = Image.new('RGB', (imsize1, imsize2), (128, 128, 128))
        # im = im.resize((imsize1,imsize2))
        new_im.paste(im, ((imsize1 - nw) // 2, (imsize2 - nh) // 2))
        new_im = np.array(new_im).astype(np.float32) / 255

        batch_hp[i] = batch_x2[i]
        batch_imgs[i, :, :, :] = new_im
        angles = batch_y[i]
        for op in range(params.data.OUT_Layers):
            indv = int(angles[1] * (params.data.OUT_SHAPE_1[0] * (2 ** op) - 1) / 1080)
            indh = int(angles[0] * (params.data.OUT_SHAPE_1[1] * (2 ** op) - 1) / 1920)
            batch_angles[op][i, indv:indv + 2, indh:indh + 2] = 0.25 * np.ones((2, 2))
    if params.model.hp_en and params.model.eye_en:
        model_in = [batch_imgs,batch_hp]
    elif params.model.hp_en:
        model_in = batch_hp
    elif params.model.eye_en:
        model_in = batch_imgs
    else:
        print('At least one of the hp and eye branch should be enabled')
        return -1
    return model_in, batch_angles, batch_y

def rep_data_for_balance(train_images, train_bb, train_angles, train_hp):
    print('repeating less frequent data to balance')
    angle_arr = np.array(train_angles)
    new_train_image = []
    new_train_bb = []
    new_train_angles = []
    new_train_hp = []
    hist1, x_bins, y_bins = np.histogram2d(angle_arr[:,0],angle_arr[:,1],10)
    for im, bb, ang, hp in zip(train_images, train_bb, train_angles, train_hp):
        x_loc = np.sum(x_bins < ang[0]) - 1
        y_loc = np.sum(y_bins < ang[1]) - 1
        hist_val = np.max(hist1)/hist1[x_loc,y_loc]
        for rep in range(int(round(hist_val))):
            new_train_angles.append(ang)
            new_train_image.append(im)
            new_train_bb.append(bb)
            new_train_hp.append(hp)
    return new_train_image, new_train_bb, new_train_angles, new_train_hp


def rep_marker_data(train_images, train_bb, train_angles, train_hp, train_images_mark, train_bb_mark, train_angles_mark, train_hp_mark):
    cont_len = len(train_images)
    mark_len = len(train_images_mark)
    ratio_cm = cont_len // mark_len
    train_images_final = train_images_mark*ratio_cm + train_images
    train_bb_final = train_bb_mark*ratio_cm + train_bb
    train_angles_final = train_angles_mark*ratio_cm + train_angles
    train_hp_final = train_hp_mark*ratio_cm + train_hp
    return train_images_final, train_bb_final, train_angles_final, train_hp_final


