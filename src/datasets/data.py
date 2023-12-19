from torch.utils.data import Dataset
import glob
import os
from skimage.transform import rescale
from skimage import io
import torch

class ComphyDataset(Dataset):

    def __init__(
            self,
            split='train',
            data_root='',
            down_sz=64,
            ):
        self.down_sz = down_sz
        self.files = glob.glob(f'{data_root}/**/*.png', recursive=True)
        self.files.sort()
        split_index = int(0.95 * len(self.files))
        if split=='train':
            self.files = self.files[:split_index]
        elif split=='val':
            self.files = self.files[split_index:]
        print('{} items loaded for the {}_set'.format(len(self.files),split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        imgpath = self.files[idx]
        scaled_img = self.rescale_img(io.imread(imgpath))
        img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
        return img

    def rescale_img(self,img):
        H,W,C = img.shape
        down = rescale(img, [self.down_sz/H], order=3, mode='reflect', multichannel=True)
        return down

def build_comphyImg_dataset(args):
    train_dataset = ComphyDataset(split='train',**args)
    val_dataset = ComphyDataset(split='val',**args)
    return train_dataset, val_dataset
