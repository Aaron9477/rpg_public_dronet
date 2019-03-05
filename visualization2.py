#!usr/bin/env python2
#!coding=utf-8
# from keras.applications.vgg16 import (
#     VGG16, preprocess_input, decode_predictions)
from keras.applications.resnet50 import (
    ResNet50, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from keras.models import Model

import utils
from evaluation_flags import FLAGS
import os



# category_index是各个类的预测概率
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(320, 320))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='res5c_branch2c'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    # K.learning_phase()是学习阶段标志，一个布尔张量（0 = test，1 = train）
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = ResNet50(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, layer_name, pic_path):
    model_loss = Lambda(cal_loss, output_shape=(1,),
        arguments={'pic_path': pic_path, })(input_model.output)
    model = Model(inputs=input_model.input, outputs=model_loss)
    # model.summary()
    loss = K.sum(model.output)

    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    # 将计算图编译为具体的函数
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    # 下面这块不是很理解
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # 对各个通道的特征图进行取平均，得到单个通道的均值
    weights = np.mean(grads_val, axis=(0, 1))
    # why initialize with cam with ones
    # cam = np.ones(output.shape[0: 2], dtype=np.float32)
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    # resize
    cam = cv2.resize(cam, (320, 320), interpolation=cv2.INTER_CUBIC)
    # 取正值
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    # all colol map to the original image
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    # unit8 to save memory
    return np.uint8(cam), heatmap


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
        pic_list = os.listdir(os.path.join(label_path, "images"))
        pic_list.sort()
        index = pic_list.index(pic_path.split('/')[-1])
        coll_ground_truth = ground_truth_list[index-1]
        coll_predict = predict[1]
        # print(coll_ground_truth)
        # print(coll_predict)
        # exit()
        coll_loss = K.binary_crossentropy([[coll_ground_truth]], coll_predict)
        # coll_loss = K.square(coll_ground_truth - coll_predict)

        return coll_loss

    else:
        raise ValueError('wrong picture category!')


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    layer_visualize = "conv2d_10"
    # pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/GOPR0300/images/frame_00392.jpg"
    pic_path = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/training/HMB_2/images/1479425040343968790.png"
    preprocessed_input = load_image(pic_path)
    # preprocessed_input = load_image(sys.argv[1])

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    weight_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)

    # json_model_path = '/home/zq610/WYZ/deeplearning/network/rpg_public_dronet/model/model_struct.json'
    # weight_path = '/home/zq610/WYZ/deeplearning/network/rpg_public_dronet/model/model_weights.h5'

    model = utils.jsonToModel(json_model_path)
    weights_load_path = weight_path
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    cam, heatmap = grad_cam(model, preprocessed_input, layer_visualize, pic_path)
    while(cv2.waitKey(27)):
        cv2.imshow("WindowNameHere", cam)

    cv2.imwrite("gradcam.jpg", cam)

# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model, layer_visualize)
# saliency = saliency_fn([preprocessed_input, 0])
# # np.newaxis加一个维度
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
