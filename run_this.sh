#!/usr/bin/env bash

# evaluation just uses the file.
#python3 evaluation.py --experiment_rootdir='./model' --weights_fname='model_weights.h5' --test_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation'
#python3 evaluation.py

# original train file
#python cnn.py

# darknet file train file
# python darknet_cnn.py

python3 mobilenet_cnn.py >> output.txt

#python3 evaluation.py --experiment_rootdir='result/darknet14_3.7' --weights_fname='weights_040.h5' \
#--test_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation' --img_mode='rgbe'

#python3 evaluation.py --experiment_rootdir='result/darknet14_3.6' --weights_fname='weights_022.h5' \
#--test_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation' --img_mode='rgbe'

#python3 evaluation.py --experiment_rootdir='result/darknet14_3.4' --weights_fname='weights_029.h5' \
#--test_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation' --img_mode='rgb'

#python3 evaluation.py --experiment_rootdir='result/mobilenet_3.11' --weights_fname='weights_027.h5' \
#--test_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/validation' --img_mode='rgb'

