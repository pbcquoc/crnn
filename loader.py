import torchvision
import torch
from torchvision.transforms import ToTensor
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import argparse
import time
from multiprocessing import cpu_count
import uuid

def img_loader(path):
    return Image.open(path).convert('L')

def target_loader(path):
    label = open(path).read().rstrip()
    return label

def default_flist_reader(flist):
    imlist = []
    with open(flist) as rf:
        for line in rf.readlines():
            impath = line.strip()
            if impath.endswith('.jpg') or impath.endswith('.png'):
                imlabel = os.path.splitext(impath)[0] + '.txt'
                imlist.append((impath, imlabel))
                                    
    return imlist

class resizeNormalize(object):
    def __init__(self, width, height):
        self.expected_size = width, height
        self.toTensor = ToTensor()

    def __call__(self, img):
        desired_w, desired_h = self.expected_size #(width, height)
        img_w, img_h = img.size  # old_size[0] is in (width, height) format
        ratio = 1.0*img_w/img_h
        new_size = min(desired_w, int(np.floor(desired_h*ratio))), desired_h
        img = img.resize(new_size, Image.ANTIALIAS)

        # padding image
        new_img = Image.new("L", (desired_w, desired_h), color=255)
        new_img.paste(img, (0, 0))
        new_img = self.toTensor(new_img)
        return new_img

class ImageFileList(data.Dataset):
    def __init__(self, root, flist, transform, target_transform,
        flist_reader=default_flist_reader):
        self.root   = root
        self.imlist = flist_reader(flist)             
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        impath, targetpath = self.imlist[index]
        imgpath = os.path.join(self.root,impath)
        targetpath = os.path.join(self.root,targetpath)

        img = self.transform(imgpath)
        target = self.target_transform(targetpath)

        return img, target

    def __len__(self):
        return len(self.imlist)

class alignCollate(object):

    def __init__(self, imgW, imgH):
        self.imgH = imgH
        self.imgW = imgW
    
    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize(imgW, imgH)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


class DatasetLoader(object):
    def __init__(self, root, train_file, test_file, imgW, imgH):
        self.root = root
        self.train_file = os.path.join(root, train_file)
        self.test_file = os.path.join(root, test_file)
        self.imgW = imgW
        self.imgH = imgH

        self.train_dataset = ImageFileList(root, self.train_file, transform=img_loader, target_transform=target_loader) 

        self.test_dataset = ImageFileList(root, self.test_file, transform=img_loader, target_transform=target_loader)


    def train_loader(self, batch_size, num_workers=4, shuffle=True):

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=alignCollate(self.imgW, self.imgH)
        )

        return train_loader

    def test_loader(self, batch_size, num_workers=4, shuffle=True):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=alignCollate(self.imgW, self.imgH)
        )

        return test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='path to root folder')
    parser.add_argument('--train', required=True, help='path to train list')
    parser.add_argument('--val', required=True, help='path to test list')
    opt = parser.parse_args()

    loader = DatasetLoader(opt.root, opt.train, opt.val, 1024, 64)
    for _ in range(100):
        train_loader = iter(loader.train_loader(64, num_workers=cpu_count()))
        i = 0
        while i < len(train_loader):
            start_time = time.time()
            X_train, y_train = next(train_loader)
            elapsed_time = time.time() - start_time

            i += 1
            print(i, elapsed_time, X_train.size())

if __name__ == '__main__':
    main()
