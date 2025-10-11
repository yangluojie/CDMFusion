import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random

class Fusion_Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True):

        if self.config.data.data_dir_for_train == None or 'None': train_path =self.config.data.data_dir_for_test;train_filelist=self.config.data.filelist_for_test
        else: train_path = self.config.data.data_dir_for_train;train_filelist=self.config.data.filelist_for_train
        train_dataset = Fusion_Dataset(train_path,
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          filelist=train_filelist,
                                          parse_patches=parse_patches,
                                          datasets=self.config.data.dataset)
        val_dataset = Fusion_Dataset(self.config.data.data_dir_for_test,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=self.config.data.filelist_for_test,
                                        parse_patches=parse_patches,
                                        datasets=self.config.data.dataset)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class Fusion_Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True, datasets=None):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip() for i in input_names] #.split('_')[0]+'_GT.png'

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.datasets = datasets

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        if self.datasets == 'RSFB':
            input_img = PIL.Image.open(os.path.join(self.dir,'vi', input_name)).convert('RGB') if self.dir else PIL.Image.open(input_name).convert('RGB')
            try:
                gt_img = PIL.Image.open(os.path.join(self.dir,'gt_vi', input_name.split('_')[0]+'.png')).convert('RGB') if self.dir else PIL.Image.open(input_name).convert('RGB')
            except:
                gt_img = PIL.Image.open(os.path.join(self.dir,'gt_vi', input_name.split('_')[0]+'.png')).convert('RGB') if self.dir else \
                    PIL.Image.open(gt_name).convert('RGB')
            try:
                ir_img = PIL.Image.open(os.path.join(self.dir,'ir', input_name.split('_')[0]+'.png')).convert('L') if self.dir else PIL.Image.open(input_name).convert('L')
            except:
                ir_img = PIL.Image.open(os.path.join(self.dir,'ir', input_name.split('_')[0]+'.png')).convert('L') if self.dir else \
                    PIL.Image.open(input_name).convert('L')
        else:
            input_img = PIL.Image.open(os.path.join(self.dir,'vi', input_name)).convert('RGB') if self.dir else PIL.Image.open(input_name).convert('RGB')
            try:
                gt_img = PIL.Image.open(os.path.join(self.dir,'vi', input_name)).convert('RGB') if self.dir else PIL.Image.open(input_name).convert('RGB')
            except:
                gt_img = PIL.Image.open(os.path.join(self.dir,'vi', input_name)).convert('RGB') if self.dir else \
                    PIL.Image.open(gt_name).convert('RGB')
            try:
                ir_img = PIL.Image.open(os.path.join(self.dir,'ir', input_name)).convert('L') if self.dir else PIL.Image.open(input_name).convert('L')
            except:
                ir_img = PIL.Image.open(os.path.join(self.dir,'ir', input_name)).convert('L') if self.dir else \
                    PIL.Image.open(input_name).convert('L')
        

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            ir_img = self.n_random_crops(ir_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i]), self.transforms(ir_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            maxsize = 640
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > maxsize:
                wd_new = int(np.ceil(wd_new * maxsize / ht_new))
                ht_new = maxsize
            elif ht_new <= wd_new and wd_new > maxsize:
                ht_new = int(np.ceil(ht_new * maxsize / wd_new))
                wd_new = maxsize
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            ir_img = ir_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img), self.transforms(ir_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
