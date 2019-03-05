#!usr/bin/env python2
#!coding=utf-8

# class test():
#     def aa(self):
#         print("aa")
#
# A = test()
# A.bb = 4
# A.aa()
# print(A.bb)


import numpy as np
import os
import keras as K
# def judge_pic_cate(pic_path):
#     pic_cate_path = pic_path.split('/')[-3].split('_')[0]
#     if pic_cate_path == 'HMB':
#         return 'HMB'
#     else:
#         return 'GOPR'


def cal_loss(predict, pic_path):
    label_path = pic_path.split('/')[:-2]
    label_path = '/'.join(label_path)
    # judge the picture is HMB or GOPR
    pic_cate_path = pic_path.split('/')[-3].split('_')[0]
    if pic_cate_path == 'HMB':
        pic_cate = 'HMB'
    else:
        pic_cate = 'GOPR'

    if pic_cate == 'HMB':
        steerings_filename = os.path.join(label_path, 'sync_steering.txt')
        ground_truth_list = np.loadtxt(steerings_filename, usecols=0, delimiter=',', skiprows=1)
        pic_list = os.listdir(os.path.join(label_path, "images"))
        pic_list.sort()
        index = pic_list.index(pic_path.split('/')[-1])
        steer_ground_truth = ground_truth_list[index]
        steer_predict = predict[0]
        steer_loss = K.square(steer_predict - steer_ground_truth)
        return steer_loss

    elif pic_cate == 'GOPR':
        labels_filename = os.path.join(label_path, 'labels.txt')
        ground_truth_list = np.loadtxt(labels_filename, usecols=0)
        index = int(pic_path.split('/')[-1].split('_')[-1].split('.')[0])
        coll_ground_truth = ground_truth_list[index]
        coll_predict = predict[1]
        coll_loss = K.binary_crossentropy(coll_ground_truth, coll_predict)
        return coll_loss

    else:
        raise ValueError('wrong picture category!')

# pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/GOPR0272/images/frame_00027.jpg"
pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/HMB_4/images/1479425730681186395.png"
cal_loss(pic_path)




