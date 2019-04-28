#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: run.py
# Brown CSCI 1430 assignment
# Created by Aaron Gokaslan

import numpy as np
import argparse
import os
import cv2
import sys
from glob import glob
from tensorpack.dataflow import imgaug as iaa
import random

from tensorpack import *
from tensorpack.tfutils.sessinit import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow

from vgg_model import VGGModel
from your_model import YourModel
import hyperparameters as hp

"""
TASK 1: To train from scratch (on CPU) and validate:
    python run.py --task 1 --gpu -1

TASK 2: To fine tune the VGG-16 model on CPU (will error until you make changes):
    python run.py --task 2 --gpu -1

On Google Cloud Platform, you will set up the facility
to use a powerful GPU. There, you can change the value passed to the flag:

    --gpu -1  runs in CPU mode.
    --gpu 0   runs on the first machine GPU,  
    --gpu 1   runs on the second machine GPU (should you be so lucky; GCP will only give you 1 GPU)

More advanced functions (see __main__):
    python run.py --gpu -1,0,1 --data datadir --load vgg16.npy --task 1,2
"""


"""
 
"""


# def my_data_loader(inpt):
#     # load data from somewhere with Python, and yield them
#     for k in range(100):
#         yield [inpt[0][k], inpt[1][k]]

class MyDataFlow(DataFlow):

    def __init__(self, inpt):
        self.inpt = inpt
        np.random.shuffle(self.inpt)

    def __iter__(self):
        # load data from somewhere with Python, and yield them
        for k in range(x , 100):
            yield [self.inpt[0][k], self.inpt[1][k]]

    def __len__(self):
        return 100




class Scene15(RNGDataFlow):

    def __init__(self, dir, name, img_size, meta_dir=None,
                 shuffle=None, dir_structure=None):

        assert name in ['train', 'test'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        # For each category, add up to the self-enforced limit on the number of training/test examples
        #
        self.imglist = []
        for catname in glob('%s/%s/*' % (dir, name)):
            catlist = glob('%s/*' % catname)
            c = 0
            for fname in catlist:
                self.imglist.append( (fname, os.path.basename(os.path.dirname(fname))) )
                c = c+1
                if name == 'train' and c >= hp.num_train_per_category:
                    break
                if name == 'test' and c >= hp.num_test_per_category:
                    break

        # Compact variant with no limits; just read all the data
        # We don't do this for speed reasons
        # self.imglist2 = [(fname, os.path.basename(os.path.dirname(fname))) for fname in glob('%s/%s/*/*' % (dir, name))]

        self.label_lookup = dict()
        for label in sorted(set(i[1] for i in self.imglist)):
            self.label_lookup[label] = len(self.label_lookup)

        self.imglist = [(fname, self.label_lookup[dirname]) for fname, dirname in self.imglist]

        
        idxs = np.arange(len(self.imglist))

        # Load images into numpy array
        ## VVV: let's make sure this is how we want the data to be loaded and normalized
        self.imgs = np.zeros( (img_size, img_size, 3, len(self.imglist) ), dtype=np.float )
        for k in idxs:
            fname, label = self.imglist[k]
            img = cv2.resize(cv2.imread(fname), (img_size, img_size) )
            # data normalization
            img = (img - np.mean(img, axis = (0, 1, 2))) / np.std(img, axis = (0, 1, 2))
            self.imgs[:,:,:,k] = img


    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [self.imgs[:,:,:,k], label]


"""
Convenience function to load the 15 Scene database.
This is where you would place any potential data augmentations.
"""
def get_data(datadir, train_or_test):
    isTrain = train_or_test == 'train'
    img_size = hp.img_size
    ds = Scene15(datadir, train_or_test, img_size)
    if isTrain:
        augmentors = [
            # VVV: lets do our own augmentors!!!!
            #################################################
            # TASK 1: Add data augmentations
            #
            # An example (that is duplicated work).
            # In the Scene15 class, we resize each image to 
            # 64x64 pixels as a preprocess. You then perform
            # standardization over the images in Task 1.
            #
            # However, if we wanted to skip standardization, 
            # we could use an augmentation to resize the image
            # whenever it is needed:
            #
            # Please use the same syntax to write more useful 
            # augmentations. Read the documentation on the 
            # TensorPack image augmentation library and experiment!
            #################################################
            imgaug.Affine(rotate_max_deg=45),
            imgaug.GaussianNoise(1, True),
        ]
    else:
        # Validation/test time augmentations
        augmentors = [
            # imgaug.Resize( (img_size, img_size) ) 
        ]
    # TensorPack: Add data augmentations
    ds = AugmentImageComponent(ds, augmentors)
    # TensorPack: How to batch the data
    # VVV: apparently... adds another dimension to the data points? 
    ds = BatchData(ds, hp.batch_size, remainder=not isTrain)
    if isTrain:
        # TensorPack: Perform clever image fetching, e.g., multithreaded
        # These numbers will depend on your particular machine.
        # Note: PrefetchData seems to be broken on Windows : /
        if not sys.platform.lower().startswith('win'):
            ds = PrefetchData(ds, 4, 2)
    return ds


"""
Program argument parsing, data setup, and training
"""
#VV: change the TASKS SHITS
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--task',
    #     required=True,
    #     choices=['1', '2'],
    #     help='Which task of the assignment to run - training from scratch (1), or fine tuning VGG-16 (2).')
    # # Set GPU to -1 to not use a GPU.
    # parser.add_argument('--gpu', help='Comma-separated list of GPU(s) to use.')
    # parser.add_argument(
    #     '--load',
    #     # Location of pre-trained model
    #     # - As a relative path to the student distribution
    #     #VVV: soooo this will be our dataset??? but is our dataset in the same format
    #     default='vgg16.npy',
    #     # - As an absolute path to the location on the Brown CS filesystem
    #     #default='/course/cs1430/pretrained_weights/vgg16.npy',
    #     help='Load VGG-16 model.')
    # parser.add_argument(
    #     '--data',
    #     # Location of 15 Scenes dataset
    #     # - As a relative path to the student distribution
    #     default=os.getcwd() + '/../data/',
    #     # - As an absolute path to the location on the Brown CS filesystem
    #     #default='/course/cs1430/datasets/15SceneData/',
    #     help='Location where the dataset is stored.')

    # args = parser.parse_args()

    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # logger.auto_set_dir()

    # dataset_train = get_data(args.data, 'train')
    # dataset_test = get_data(args.data, 'test')
    train_info = []
    train_img = []
    for i in range(10000):
        a = np.zeros([8,8])
        y = random.randint(0,7)
        x = random.randint(0,7)
        h = random.randint(0,7-y)
        w = random.randint(0,7-x)
        for t in range(w):
            for k in range(h):
                a[y+k][x+t] = 1
        train_info.append(x)
        # train_info.append([x,y,w,h])
        train_img.append(a)
    train = np.array(train_img)
    dataset_train = train

    test_info = []
    test_img = []
    for i in range(100):
        a = np.zeros([8,8])
        y = random.randint(0,7)
        x = random.randint(0,7)
        h = random.randint(0,7-y)
        w = random.randint(0,7-x)
        for t in range(w):
            for k in range(h):
                a[y+k][x+t] = 1
        test_info.append(x)
        # test_info.append([x,y,w,h])
        test_img.append(a)
    test = np.array(test_img)   
    dataset_test = test



    # TensorPack: Training configuration
    config = TrainConfig(
        model=YourModel(),
        # dataflow=dataset_train,
        dataflow = MyDataFlow((dataset_train)),
        callbacks=[
            # Callbacks are performed at the end of every epoch.
            #
            # For instance, we can save the current model
            ModelSaver(checkpoint_dir="/home/adam_shelby/save"),
            # Evaluate the current model and print out the loss
            InferenceRunner(MyDataFlow((dataset_test)),
                            [ScalarStats('cost'), ClassificationError()])
            #
            # You can put other callbacks here to change hyperparameters,
            # etc...
            #
        ],
        max_epoch=hp.num_epochs,
        nr_tower=max(get_nr_gpu(), 1),
        session_init=None# if args.task == '1' else get_model_loader(args.load)
    )
    # TensorPack: Training with simple one at a time feed into batches
    launch_train_with_config(config, SimpleTrainer())
