import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib


class MoDA(Dataset):
    """ MoDA Dataset """

    def __init__(self, base_dir=None, split='train',  num=None, transform=None, num_classes=2):
        self._base_dir = base_dir
        self.transform = transform

        train_path = self._base_dir+'/train.txt'
        train_target_path = self._base_dir+'/train_target.txt'
        test_path = self._base_dir+'/val.txt'
        self.num_classes = num_classes

        if split == 'train':
            with open(train_path, 'r') as f:
                self.name_list = [a.strip() for a in f.readlines()]
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.name_list = [a.strip() for a in f.readlines()]
        self.file_list = []

        for row in self.name_list:
            t = os.path.join(self._base_dir,"training_source",row+'_ceT1.nii.gz')
            label = os.path.join(self._base_dir,"training_source",row+'_Label.nii.gz')
            # mask = os.path.join(self._base_dir,'training_target',row[0]+'_GIFoutput.nii.gz')

            self.file_list.append({'image':t,"label":label})

        print("total labeled {} samples".format(len(self.file_list)))
        
        if split == 'semi-train':
            with open(train_target_path, 'r') as f:
                self.name_list2 = [a.strip() for a in f.readlines()]
            for row in self.name_list2:
                t = os.path.join(self._base_dir,"training_target",row+'_hrT2.nii.gz')
                label = os.path.join(self._base_dir,"training_target",row+'_Label.nii.gz')
                self.file_list.append({'image':t,"label":label})
        
        if num is not None:
            self.file_list = self.file_list[:num]
        print("total {} samples".format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    
    def __getitem__(self, idx):
        s = self.file_list[idx]
        image = nib.load(s['image']).get_fdata()
        # label = np.max(nib.load(s['label']).get_fdata(), axis=-1)
        # label = np.stack([label==num_class for num_class in range(self.num_classes)], axis=-1)
        if os.path.exists(s['label']):
            label = nib.load(s['label']).get_fdata().astype(np.uint8)
        else:
            label = np.zeros_like(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
