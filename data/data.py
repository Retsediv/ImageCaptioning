import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from data_loader import get_loader
import torchvision
import argparse
import os
import numpy as np
import torch
from PIL import Image


class Flickr8k(object):
    def __init__(self, root, transform, text_transform):
        self.root = root
        self.transform = transform

        self.imgs = list(sorted(os.listdir(os.path.join(root, "image/train"))))

    def __getitem__(self, idx):
        img, target= None, None

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def transform(img, target):
    return img, target

def trantext_transform(target):
    return traget


def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    print(args)
    main(args)