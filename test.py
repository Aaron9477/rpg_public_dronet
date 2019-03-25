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


# import numpy as np
# import os
# import keras as K
# def judge_pic_cate(pic_path):
#     pic_cate_path = pic_path.split('/')[-3].split('_')[0]
#     if pic_cate_path == 'HMB':
#         return 'HMB'
#     else:
#         return 'GOPR'


# def cal_loss(predict, pic_path):
#     label_path = pic_path.split('/')[:-2]
#     label_path = '/'.join(label_path)
#     # judge the picture is HMB or GOPR
#     pic_cate_path = pic_path.split('/')[-3].split('_')[0]
#     if pic_cate_path == 'HMB':
#         pic_cate = 'HMB'
#     else:
#         pic_cate = 'GOPR'
#
#     if pic_cate == 'HMB':
#         steerings_filename = os.path.join(label_path, 'sync_steering.txt')
#         ground_truth_list = np.loadtxt(steerings_filename, usecols=0, delimiter=',', skiprows=1)
#         pic_list = os.listdir(os.path.join(label_path, "images"))
#         pic_list.sort()
#         index = pic_list.index(pic_path.split('/')[-1])
#         steer_ground_truth = ground_truth_list[index]
#         steer_predict = predict[0]
#         steer_loss = K.square(steer_predict - steer_ground_truth)
#         return steer_loss
#
#     elif pic_cate == 'GOPR':
#         labels_filename = os.path.join(label_path, 'labels.txt')
#         ground_truth_list = np.loadtxt(labels_filename, usecols=0)
#         index = int(pic_path.split('/')[-1].split('_')[-1].split('.')[0])
#         coll_ground_truth = ground_truth_list[index]
#         coll_predict = predict[1]
#         coll_loss = K.binary_crossentropy(coll_ground_truth, coll_predict)
#         return coll_loss
#
#     else:
#         raise ValueError('wrong picture category!')
#
# # pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/GOPR0272/images/frame_00027.jpg"
# pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/HMB_4/images/1479425730681186395.png"
# cal_loss(pic_path)






# coding=utf-8
import cv2
import numpy as np

# pic_id = '1479425099304331898'
#
# origin = cv2.imread("/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/HMB_2/images/{id}.png".format(id=pic_id))
# origin = cv2.resize(origin, (320,320))
# img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (3, 3), 0)
#
# canny0 = cv2.Canny(img, 50, 150)
# # canny1 = cv2.Canny(img, 50, 200)
# # canny2 = cv2.Canny(img, 50, 250)
# # canny3 = cv2.Canny(img, 50, 300)
# # canny4 = cv2.Canny(img, 50, 350)
# # canny5 = cv2.Canny(img, 100, 300)
# # canny6 = cv2.Canny(img, 150, 350)
# # out = cv2.merge([origin, canny0])
# # print(out)
# # exit()
#
# cv2.imwrite("/home/zq610/WYZ/graduation_project/picture/{id}.jpg".format(id=pic_id), origin)
# cv2.imwrite("/home/zq610/WYZ/graduation_project/picture/{id}_edge.jpg".format(id=pic_id), canny0)
# exit()
#
# cv2.imshow('img', origin)
# cv2.imshow('Canny0', canny0)
# # cv2.imshow('Canny1', canny1)
# # cv2.imshow('Canny2', canny2)
# # cv2.imshow('Canny3', canny3)
# # cv2.imshow('Canny4', canny4)
# # cv2.imshow('Canny5', canny5)
# # cv2.imshow('Canny6', canny6)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


train_name = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val_name = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
test_name = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

def read_print(file_name):
    with open(file_name) as file:
        a = file.readlines()
        print(len(a))
read_print(train_name)
read_print(val_name)
read_print(test_name)


