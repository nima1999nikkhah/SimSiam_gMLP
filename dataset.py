from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
# from decord import VideoReader
# from decord import cpu, gpu
import glob

from icecream import ic

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10(Dataset):

    def __init__(self,
                 root_dir='cifar-10-batches-py/',
                 resize_height=32,
                 resize_width=32,
                 transform=None,
                 offset=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = []
        for i in range(5):

            train_file_path = os.path.join(root_dir, 'data_batch_{}'.format(i+1))
            # ic(root_dir, train_file_path)
            dict = unpickle(train_file_path)
            self.data.extend(dict[b'data'])
        # self.labels = dict[b'fine_labels']



        self.resize_height = resize_height
        self.resize_width = resize_width


        # ic(len(self.training_vlist))
        # self.root_dir = root_dir
        self.transform = self.get_simclr_pipeline_transform(size=resize_height, s=1)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_simclr_pipeline_transform(size=256, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = ColorJitter(0.3, 0.3, 0.3, 0.2)

        data_transforms = transforms.Compose([ToTensor(),
                                              Normalize(),
                                              RandomResizedCrop(size=size, scale=(0.7, 0.9)),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              # RandomGrayscale(p=0.2),
                                              RandomHorizontalFlip(p=0.5),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              ])
        return data_transforms


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx].reshape(3, 32, 32)

        sample = {'imgs1': img,
                  'imgs2': img,
                  # 'video3': video1,
                  # 'video4': video2
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample



class Normalize(object):
    def __init__(self):
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        # return {"imgs1": self.transform(sample["imgs1"]/255.),
        #         "imgs2": self.transform(sample["imgs2"]/255.)}
        return {'imgs1': sample['imgs1']/255.,
                'imgs2': sample['imgs2']/255.,
                # 'imgs3': sample['imgs3']/255.,
                # 'imgs4': sample['imgs4']/255.
                }

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        # imgs1_, imgs2_ = sample['imgs3'], sample['imgs4']
        # swap color axis because
        # numpy img: H x W x C
        # torch image: C X H X W
        # print(imgs.shape)
        imgs1 = np.array(imgs1).astype(np.float32)
        imgs2 = np.array(imgs2).astype(np.float32)
        # imgs1_ = np.array(imgs1_).transpose((0, 3, 1, 2)).astype(np.float32)
        # imgs2_ = np.array(imgs2_).transpose((0, 3, 1, 2)).astype(np.float32)
        return {'imgs1': torch.from_numpy(imgs1),
                'imgs2': torch.from_numpy(imgs2),
                # 'imgs3': torch.from_numpy(imgs1_),
                # 'imgs4': torch.from_numpy(imgs2_)
                }

class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        imgs1_ = super().__call__(imgs1)
        imgs2_ = super().__call__(imgs2)
        return {'imgs1': imgs1_,
                'imgs2': imgs2_,
                # 'imgs3': sample['imgs3'],
                # 'imgs4': sample['imgs4']
                }

class RandomGrayscale(transforms.RandomGrayscale):
    def __init__(self, p):
        super(RandomGrayscale, self).__init__(p)

    def __call__(self, sample):
        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        imgs1_ = super().__call__(imgs1)
        imgs2_ = super().__call__(imgs2)
        return {'imgs1': imgs1_,
                'imgs2': imgs2_,
                # 'imgs3': sample['imgs3'],
                # 'imgs4': sample['imgs4']
                }

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, sample):
        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        imgs1_ = super().__call__(imgs1)
        imgs2_ = super().__call__(imgs2)
        return {'imgs1': imgs1_,
                'imgs2': imgs2_,
                # 'imgs3': sample['imgs3'],
                # 'imgs4': sample['imgs4']
                }

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale):
        super(RandomResizedCrop, self).__init__(size=size, scale=scale, ratio=(0.99, 1.))

    def __call__(self, sample):
        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        imgs1_ = super().__call__(imgs1)
        imgs2_ = super().__call__(imgs2)
        return {'imgs1': imgs1_,
                'imgs2': imgs2_,
                # 'imgs3': sample['imgs3'],
                # 'imgs4': sample['imgs4']
                }

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        # self.pil_to_tensor = transforms.ToTensor()
        # self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, sample):

        imgs1, imgs2 = sample['imgs1'], sample['imgs2']
        imgs1 = imgs1.unsqueeze(0)
        imgs2 = imgs2.unsqueeze(0)
        # img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            imgs1_ = self.blur(imgs1)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            imgs2_ = self.blur(imgs2)
            # img = img.squeeze()

        # img = self.tensor_to_pil(img)

        return {'imgs1': imgs1_.squeeze(0),
                'imgs2': imgs2_.squeeze(0),
                # 'imgs3': sample['imgs3'],
                # 'imgs4': sample['imgs4']
                }



class MyDataLoader:
    def __init__(self,
                 data_dir='cifar-10-batches-py/',
                 resize_height=32,
                 resize_width=32,
                 offset=0,
                 ):
        # self.my_transforms = transforms.Compose([ToTensor(), Normalize()])
        self.trainDataLoad = CIFAR10(root_dir=data_dir,
                                      resize_height=resize_height,
                                      resize_width=resize_width,
                                      transform=None,
                                      offset=offset)

        # self.testDataLoad = SomethingSomething10Classes(
        #                                 json_file='reduced_validation_file.json',
        #                                 root_dir='/data2/something-something/',
        #                                 resize_height=256,
        #                                 resize_width=256,
        #                                 transform=self.my_transforms)

    def dataloader(self, batch_size, num_workers, train):
        if train:
            shuffle=True
            return DataLoader(self.trainDataLoad, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
        # else:
        #     shuffle = False
        #     return DataLoader(self.testDataLoad, batch_size=batch_size,
        #                         shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    loader = MyDataLoader()
    train_ds = loader.dataloader(10, 10, True)

    for sample in train_ds:
        ic(sample['imgs1'].shape)
        exit()
