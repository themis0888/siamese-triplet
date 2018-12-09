# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:28:42 2018
@author: Baek
"""

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from os import listdir
from os.path import join
from PIL import Image
import random
import matplotlib.pyplot as plt
import imageio
import cv2
import numpy as np
import torch
from skimage import color
from torch.utils import data
import os
import pickle
from tqdm import tqdm
import time
def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".hdr"])

def load_img(filepath):
    img = Image.open(filepath)
    #img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, args, name='DIV2K' ,train=True):
        super(DatasetFromFolder, self).__init__()
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale[0]
        self.bin = self.args.ext.find('sep') >=0 or self.args.ext.find('bin') >=0
        self.LR_filenames, self.HR_filenames = self._get_filenames(name)
        if args.n_colors == 4:
            self.LR_labelnames, self.HR_labelnames = self._get_filenames('cityscapes/gtFine')
        if self.bin:
            self.LR_filenames, self.HR_filenames = self._get_bin(name, self.LR_filenames, self.HR_filenames)
            if args.n_colors == 4:
                self.LR_labelnames, self.HR_labelnames = self._get_bin('cityscapes/gtFine', self.LR_labelnames, self.HR_labelnames)
        self.patch_size = args.patch_size
        
    
    def __getitem__(self, index):
        time1 = time.time()
        LR_filename, HR_filename = self.LR_filenames[index], self.HR_filenames[index]
        LR, HR = self._open_file(LR_filename, HR_filename)
        time2 = time.time()
        if self.args.n_colors == 4 and self.name.find('bin') == -1:
            LR_labelname, HR_labelname = self.LR_labelnames[index], self.HR_labelnames[index]
            LRlabel, HRlabel = self._open_file(LR_labelname, HR_labelname)
            time3 = time.time()
            #LRlabel, HRlabel = LRlabel * self.args.rgb_range / 33, HRlabel * self.args.rgb_range / 33
            #LRlabel, HRlabel = np.expand_dims(LRlabel, axis=2), np.expand_dims(HRlabel, axis=2) 
            #LR, HR = np.concatenate((LR, LRlabel), axis = 2), np.concatenate((HR, HRlabel), axis = 2)
        time4 = time.time()
        if self.train:
            if self.args.n_colors == 4 and self.name.find('bin') == -1:
                LR, HR, LRlabel, HRlabel = self._random_crop(LR, HR, LRlabel, HRlabel)
            else:
                
                LR, HR = random_crop(LR, HR, patch_size = self.patch_size, scale = self.scale)
            LR, HR = augment(LR, HR)
        
        LR, HR = np2tensor(LR, HR)
        
        filename = HR_filename
        time5 = time.time()
        #if time3:
            #print(time2-time1, time3-time2, time4-time3, time5-time4)
        #else:
            #print(time2-time1, time4-time2, time5-time4)
        
        if self.args.n_colors == 4 and self.name.find('bin') == -1:
            LRlabel, HRlabel = torch.Tensor(LRlabel), torch.Tensor(HRlabel)
            return (LR, LRlabel), (HR, HRlabel), filename
        else:
            return LR, HR, filename


    def __len__(self):
        return len(self.HR_filenames)

    def get_patch(self, lr, hr):
        scale = self.args.scale
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(scale > 1),
                input_large=False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr
        
    def _open_file(self, LR_filename, HR_filename):
        if self.bin:
            with open(HR_filename, 'rb') as _f: HR = pickle.load(_f)
        else:
            HR = load_img(HR_filename)
            HR = np.asarray(HR)

        if LR_filename:
            if self.bin:
                with open(LR_filename, 'rb') as _f: LR = pickle.load(_f)
            else:
                LR = load_img(LR_filename)
                LR = np.asarray(LR)
        else:
            HR = np.asarray(HR)
            size = np.shape(HR)
            h, w = size(0), size(1)
            h2, w2 = (h//2)*2, (w//2)*2
            HR = HR[0:h2, 0:w2, :]
            LR = cv2.resize(HR, None, fx = 0.5, fy = 0.5,interpolation = cv2.INTER_CUBIC)
        
        return LR, HR

    def _get_filenames(self, name):
        root_dir = join(self.args.dir_data, name)
        if name == 'DIV2K':
            LR_dir = join(root_dir, 'DIV2K_train_LR_bicubic')
            LR_dir = join(LR_dir, 'X'+ str(self.args.scale[0]))

            HR_dir = join(root_dir, 'DIV2K_train_HR')
            r = self.args.data_range.split('/')
            if self.train:
                data_range = r[0].split('-')
            elif self.args.test_only:
                data_range = r[0].split('-')
            else:    
                data_range = r[1].split('-')

            HR_names = sorted(listdir(HR_dir))
            HR_names = HR_names[int(data_range[0])-1:int(data_range[1])]
            LR_names = sorted(listdir(LR_dir))
            LR_names = LR_names[int(data_range[0])-1:int(data_range[1])]
            LR_filenames = [join(LR_dir, x) for x in LR_names]
            HR_filenames = [join(HR_dir, x) for x in HR_names]
        elif name == 'cityscapes/leftImg8bit' or name == 'cityscapes/gtFine':
            if self.train:
                LR_dir = join(root_dir, 'train_LR_bicubic')
                HR_dir = join(root_dir, 'train_HR')
            else:
                LR_dir = join(root_dir, 'val_LR_bicubic')
                HR_dir = join(root_dir, 'val_HR')
            LR_dir = join(LR_dir, 'X'+ str(self.args.scale[0]))

            HR_names = sorted(listdir(HR_dir))
            LR_names = sorted(listdir(LR_dir))

            LR_filenames = [join(LR_dir, x) for x in LR_names]
            HR_filenames = [join(HR_dir, x) for x in HR_names]
        elif name == 'CUB200':
            image_dir = join(root_dir, 'images', 'images')
            mid_dir = [join(image_dir, x) for x in listdir(image_dir)]
            class_dir = sorted(mid_dir)
            class_dir = class_dir[200:400]

            HR_filenames = []
            classes = []
            for idx in range(len(class_dir)):
                image_names = listdir(class_dir[idx])
                image_names = sorted(image_names)
                image_names = image_names[len(image_names)//2: len(image_names)]

                HR_filenames += [join(class_dir[idx], x) for x in image_names]
                
                classes += [idx for x in range(len(image_names))]

            self.classes = classes
            LR_filenames = None

        return LR_filenames, HR_filenames

    def _get_bin(self, name, LR_filenames , HR_filenames):
        if name.find('cityscapes/leftImg8bit') >= 0:
            image_path = 'leftImg8bit/'
            bin_path = 'leftImg8bit/bin/'
        elif name.find('cityscapes/gtFine') >=0:
            image_path = 'gtFine/'
            bin_path = 'gtFine/bin/'
        elif name.find('DIV2K') >=0:
            image_path = 'DIV2K/'
            bin_path = 'DIV2K/bin/'


        bin_hr = [x.replace(image_path, bin_path) for x in HR_filenames]
        bin_lr = [x.replace(image_path, bin_path) for x in LR_filenames]
        bin_hr = [x.replace('png', 'pt') for x in bin_hr]
        bin_lr = [x.replace('png', 'pt') for x in bin_lr]

        if self.args.ext.find('sep') >= 0:  
            for idx, (img_path, bin_path) in enumerate(tqdm(zip(HR_filenames, bin_hr), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


            for idx, (img_path, bin_path) in enumerate(tqdm(zip(LR_filenames, bin_lr), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


        return bin_lr, bin_hr

    def _load_and_make(self, img_path, bin_path):
        image = load_img(img_path)
        bin = np.asarray(image)
        with open(bin_path, 'wb') as _f: pickle.dump(bin, _f)
    def _random_crop(self, LR, HR, LRlabel, HRlabel):
        h, w, c = np.shape(LR)

        crop_w = self.patch_size//self.scale
        crop_h = self.patch_size//self.scale
        i = random.randint(0, h- crop_h)
        j = random.randint(0, w - crop_w)
        #print(i//scale, (i+crop_h)//scale, j//scale, (j+crop_w)//scale)
        LR = LR[i:(i+crop_h), j:(j+crop_w),:]
        HR = HR[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
        LRlabel = LRlabel[i:(i+crop_h), j:(j+crop_w),:]
        HRlabel = HRlabel[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
    
        return LR, HR, LRlabel, HRlabel

class TripletDataset(data.Dataset):
    def __init__(self, args, name='CUB' ,train=True):
        super(DatasetFromFolder, self).__init__()
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale[0]
        self.bin = self.args.ext.find('sep') >=0 or self.args.ext.find('bin') >=0
        self.anchors, self.poses, self.negs = self._get_filenames(name)

        self.patch_size = args.patch_size
        
    
    def __getitem__(self, index):
        time1 = time.time()
        anchor, pos, neg = self.anchors[index], self.poses[index], self.negs[index]
        LR, HR = self._open_file(anchor_name, pos_name, neg_name)
        time2 = time.time()
        if self.args.n_colors == 4 and self.name.find('bin') == -1:
            LR_labelname, HR_labelname = self.LR_labelnames[index], self.HR_labelnames[index]
            LRlabel, HRlabel = self._open_file(LR_labelname, HR_labelname)
            time3 = time.time()
            #LRlabel, HRlabel = LRlabel * self.args.rgb_range / 33, HRlabel * self.args.rgb_range / 33
            #LRlabel, HRlabel = np.expand_dims(LRlabel, axis=2), np.expand_dims(HRlabel, axis=2) 
            #LR, HR = np.concatenate((LR, LRlabel), axis = 2), np.concatenate((HR, HRlabel), axis = 2)
        time4 = time.time()
        if self.train:
            if self.args.n_colors == 4 and self.name.find('bin') == -1:
                LR, HR, LRlabel, HRlabel = self._random_crop(LR, HR, LRlabel, HRlabel)
            else:
                
                LR, HR = random_crop(LR, HR, patch_size = self.patch_size, scale = self.scale)
            LR, HR = augment(LR, HR)
        
        LR, HR = np2tensor(LR, HR)
        
        filename = HR_filename
        time5 = time.time()
        #if time3:
            #print(time2-time1, time3-time2, time4-time3, time5-time4)
        #else:
            #print(time2-time1, time4-time2, time5-time4)
        
        if self.args.n_colors == 4 and self.name.find('bin') == -1:
            LRlabel, HRlabel = torch.Tensor(LRlabel), torch.Tensor(HRlabel)
            return (LR, LRlabel), (HR, HRlabel), filename
        else:
            return LR, HR, filename


    def __len__(self):
        return len(self.HR_filenames)

    def get_patch(self, lr, hr):
        scale = self.args.scale
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(scale > 1),
                input_large=False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr
        
    def _open_file(self, LR_filename, HR_filename):
        if self.bin:
            with open(HR_filename, 'rb') as _f: HR = pickle.load(_f)
        else:
            HR = load_img(HR_filename)
            HR = np.asarray(HR)

        if LR_filename:
            if self.bin:
                with open(LR_filename, 'rb') as _f: LR = pickle.load(_f)
            else:
                LR = load_img(LR_filename)
                LR = np.asarray(LR)
        else:
            HR = np.asarray(HR)
            size = np.shape(HR)
            h, w = size(0), size(1)
            h2, w2 = (h//2)*2, (w//2)*2
            HR = HR[0:h2, 0:w2, :]
            LR = cv2.resize(HR, None, fx = 0.5, fy = 0.5,interpolation = cv2.INTER_CUBIC)
        
        return LR, HR

    def _get_filenames(self, name):
        root_dir = join(self.args.dir_data, name)
        if name == 'DIV2K':
            LR_dir = join(root_dir, 'DIV2K_train_LR_bicubic')
            LR_dir = join(LR_dir, 'X'+ str(self.args.scale[0]))

            HR_dir = join(root_dir, 'DIV2K_train_HR')
            r = self.args.data_range.split('/')
            if self.train:
                data_range = r[0].split('-')
            elif self.args.test_only:
                data_range = r[0].split('-')
            else:    
                data_range = r[1].split('-')

            HR_names = sorted(listdir(HR_dir))
            HR_names = HR_names[int(data_range[0])-1:int(data_range[1])]
            LR_names = sorted(listdir(LR_dir))
            LR_names = LR_names[int(data_range[0])-1:int(data_range[1])]
            LR_filenames = [join(LR_dir, x) for x in LR_names]
            HR_filenames = [join(HR_dir, x) for x in HR_names]
        elif name == 'cityscapes/leftImg8bit' or name == 'cityscapes/gtFine':
            if self.train:
                LR_dir = join(root_dir, 'train_LR_bicubic')
                HR_dir = join(root_dir, 'train_HR')
            else:
                LR_dir = join(root_dir, 'val_LR_bicubic')
                HR_dir = join(root_dir, 'val_HR')
            LR_dir = join(LR_dir, 'X'+ str(self.args.scale[0]))

            HR_names = sorted(listdir(HR_dir))
            LR_names = sorted(listdir(LR_dir))

            LR_filenames = [join(LR_dir, x) for x in LR_names]
            HR_filenames = [join(HR_dir, x) for x in HR_names]
        elif name == 'CUB200':
            image_dir = join(root_dir, 'images', 'images')
            mid_dir = [join(image_dir, x) for x in listdir(image_dir)]
            class_dir = sorted(mid_dir)
            class_dir = class_dir[200:400]

            HR_filenames = []
            classes = []
            for idx in range(len(class_dir)):
                image_names = listdir(class_dir[idx])
                image_names = sorted(image_names)
                image_names = image_names[len(image_names)//2: len(image_names)]

                HR_filenames += [join(class_dir[idx], x) for x in image_names]
                
                classes += [idx for x in range(len(image_names))]

            self.classes = classes
            LR_filenames = None

        return LR_filenames, HR_filenames

    def _get_bin(self, name, LR_filenames , HR_filenames):
        if name.find('cityscapes/leftImg8bit') >= 0:
            image_path = 'leftImg8bit/'
            bin_path = 'leftImg8bit/bin/'
        elif name.find('cityscapes/gtFine') >=0:
            image_path = 'gtFine/'
            bin_path = 'gtFine/bin/'
        elif name.find('DIV2K') >=0:
            image_path = 'DIV2K/'
            bin_path = 'DIV2K/bin/'


        bin_hr = [x.replace(image_path, bin_path) for x in HR_filenames]
        bin_lr = [x.replace(image_path, bin_path) for x in LR_filenames]
        bin_hr = [x.replace('png', 'pt') for x in bin_hr]
        bin_lr = [x.replace('png', 'pt') for x in bin_lr]

        if self.args.ext.find('sep') >= 0:  
            for idx, (img_path, bin_path) in enumerate(tqdm(zip(HR_filenames, bin_hr), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


            for idx, (img_path, bin_path) in enumerate(tqdm(zip(LR_filenames, bin_lr), ncols=80)):
                dir_path, basename = os.path.split(bin_path)
                os.makedirs(dir_path, exist_ok=True)
                self._load_and_make(img_path, bin_path)
                #print('Making binary files ' + bin_path)


        return bin_lr, bin_hr

    def _load_and_make(self, img_path, bin_path):
        image = load_img(img_path)
        bin = np.asarray(image)
        with open(bin_path, 'wb') as _f: pickle.dump(bin, _f)
    def _random_crop(self, LR, HR, LRlabel, HRlabel):
        h, w, c = np.shape(LR)

        crop_w = self.patch_size//self.scale
        crop_h = self.patch_size//self.scale
        i = random.randint(0, h- crop_h)
        j = random.randint(0, w - crop_w)
        #print(i//scale, (i+crop_h)//scale, j//scale, (j+crop_w)//scale)
        LR = LR[i:(i+crop_h), j:(j+crop_w),:]
        HR = HR[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
        LRlabel = LRlabel[i:(i+crop_h), j:(j+crop_w),:]
        HRlabel = HRlabel[i*self.scale:(i+crop_h)*self.scale, j*self.scale:(j+crop_w)*self.scale, :]
    
        return LR, HR, LRlabel, HRlabel


def get_train_loader(args):
    #train_set = DatasetFromFolder(args, 'cityscapes/gtFine', train=True)
    train_set = DatasetFromFolder(args, args.data_train[0], train=True)
    train_loader = data.DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=0,
                              pin_memory=False,
                              shuffle=True)

    return train_loader
def get_val_loader(args):
    #val_set = DatasetFromFolder(args, 'cityscapes/gtFine', train=False)
    val_set = DatasetFromFolder(args, args.data_test[0], train=False)
    val_loader = data.DataLoader(dataset=val_set,
                              batch_size=1,
                              num_workers=0,
                              pin_memory=False,
                              shuffle=False)
    return val_loader

def random_crop(LR, HR, patch_size = 96, scale = 2):
    h, w, c = np.shape(LR)
    crop_w = patch_size//scale
    crop_h = patch_size//scale
    i = random.randint(0, h- crop_h)
    j = random.randint(0, w - crop_w)
    #print(i//scale, (i+crop_h)//scale, j//scale, (j+crop_w)//scale)
    LR = LR[i:(i+crop_h), j:(j+crop_w),:]
    HR = HR[i*scale:(i+crop_h)*scale, j*scale:(j+crop_w)*scale, :]

    return LR, HR

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):

        if hflip: 
            img = img[:, ::-1, :]
        if vflip: 
            img = img[::-1, :, :]
        #if rot90: img = img.transpose(1,0,2)
        if rot90: 
            img = np.rot90(img, 1, (0,1))
        return img

    return [_augment(a) for a in args]

def np2tensor(*args):
    def _trans(img):
        img = torch.from_numpy((img.transpose([2, 0, 1])).copy())
        img = img.type(torch.FloatTensor)
        return img

    return [_trans(a) for a in args]