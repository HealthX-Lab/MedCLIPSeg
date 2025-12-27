import torch
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

import random
import cv2

# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2

# Normalization used by CLIP models
CLIP_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                                      
def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic, dtype=np.uint8)
    return torch.from_numpy(pic).long()

def random_rotate(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle)
    label = label.rotate(angle)
    return image, label

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (width, height)

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        # Resize
        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        # To tensor and normalize
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        # Update sample
        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text  # unchanged

        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (width, height)

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if random.random() > 0.5:
            image, mask = random_rotate(image, mask)

        # Resize
        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        # To tensor and normalize
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        # Update sample
        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text  # unchanged

        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class DatasetSegmentation(Dataset):

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: list,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224
    ) -> None:

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'label')
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name

        self.data_pairs = [
            (row['Image'], row['Ground Truth'], row['Description'])
            for row in row_text
        ]

        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x: x 

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):

        image_filename, mask_filename, text = self.data_pairs[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask[mask < 127] = 0
        mask[mask >= 127] = 1

        image, mask = correct_dims(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        inputs = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text,
            "dataset_name": self.task_name
        }

        if self.joint_transform:
            inputs = self.joint_transform(inputs)

        return inputs