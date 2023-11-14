import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as trans


class Honercornb_dataset(Dataset):
    def __init__(self, transform=None):
        super(Honercornb_dataset, self).__init__()
        self.image_path = '../data/images/' + 'img.txt'
        self.label_path = '../data/masks/' + 'mask.txt'
        f1 = open(self.image_path, 'r')
        data_image = f.readlines()
        imgs = []
        for line in data_image:
            line = line.rstrip()
            imgs.append(os.path.join('../data/images' + '/img', line))
        f1.close()

        f2 = open(self.label_path, 'r')
        data_label = f2.readlines()
        labels = []
        for line in data_label:
            line = line.rstrip()
            labels.append(os.path.join('../data/masks' + '/mask', line))
        f2.close()

        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        img = Image.open(img).convert('1')
        img = trans1(img)
        if self.transforms is not None:
            img = self.transform(img)
        label = Image.open(label).convert('1')
        label = trans1(label)
        return img, label




