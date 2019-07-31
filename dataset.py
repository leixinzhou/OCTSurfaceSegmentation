from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from AugSurfSeg import *
from smooth1D import smooth

SLICE_per_vol = 60

class OCTDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, img_np, label_np, surf, vol_list=None, transforms=None, col_len=512, sigma=50, Window_size=101):
        """
        Args:
            img_np (string): Path to the image numpy file.
            label_np (string): Path to the label numpy file.
            vol_list (list): sample list.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sf = surf
        self.image = np.load(img_np)
        self.label = np.load(label_np)
        self.vol_list = vol_list
        self.trans = transforms
        self.LT = OCTDataset.LT_gen(col_len, sigma)
        self.window_size = Window_size

    def __len__(self):
        if self.vol_list is None:
            return self.image.shape[0]
        else:
            return len(self.vol_list)*SLICE_per_vol

    def __getitem__(self, idx):
        if self.vol_list is None:
            real_idx = idx
        else:
            real_idx = self.vol_list[int(idx/SLICE_per_vol)]*SLICE_per_vol + idx % SLICE_per_vol
        image = np.swapaxes(self.image[real_idx, ], 0, 1)
        label = np.swapaxes(self.label[real_idx,:, self.sf], 0, 1)
        # print(self.label.shape, label.shape)
        img_gt = {"img": image.astype(np.float64), "gt": label}
        if self.trans is not None:
            img_gt = self.trans(img_gt)
        gt_g = self.LT[img_gt["gt"].astype(np.int32)]
        gt_g = np.transpose(gt_g, (2, 0, 1))
        # ONLY WORK ON 1D
        # print("image_gt shape: ", img_gt["gt"].shape)
        gt_d = smooth(img_gt["gt"][:-1, 0] - img_gt["gt"][1:, 0], self.window_size, 'flat')
        # print(image.shape, gt_g.shape)
        image_gt_ts = {"img": torch.from_numpy(img_gt["img"].astype(np.float32)).unsqueeze(0),
                        "gt": torch.from_numpy(img_gt["gt"].astype(np.float32).reshape(-1, order='F')),
                        "gt_g": torch.from_numpy(gt_g.astype(np.float32)),
                        "gt_d": torch.from_numpy(gt_d.astype(np.float32))}
        
        return image_gt_ts
       
    @staticmethod
    def Softmax(x_array):
        return np.exp(x_array) / np.sum(np.exp(x_array), axis=0)
    @staticmethod
    def G_PDF(x_arry, mean, sigma, A=1.):
        pdf_array = np.empty_like(x_arry, dtype=np.float16)
        for i in range(x_arry.shape[0]):
            pdf_array[i] = A * np.exp((-(x_arry[i] - mean)**2)/(2*sigma**2))
        return pdf_array
    @staticmethod
    def LT_gen(col_len, sigma):
        # lookup table
        lk_tab = np.zeros((col_len, col_len), dtype=np.float16)
        x_range = np.arange(col_len).astype(np.float16)
        for i in range(col_len):
            prob = OCTDataset.G_PDF(x_range, i, sigma)
            prob = OCTDataset.Softmax(prob)
            lk_tab[i,] = prob
        return lk_tab

if __name__ == "__main__":
    """
    test the class
    """
    aug_dict = {
                "saltpepper": SaltPepperNoise(sp_ratio=0.05), 
                "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
                "cropresize": RandomCropResize(crop_ratio=0.9), 
                "circulateud": CirculateUD(),
                "mirrorlr":MirrorLR()}
    rand_aug = RandomApplyTrans(trans_seq=[aug_dict[key] for key, _ in aug_dict.items()],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])

    vol_list = [10,3]
    vol_index = 1
    slice_index = 1
    patch_dir = "/home/leizhou/Documents/OCT/60slice/split_data_2D_400/test/patch.npy"
    truth_dir = "/home/leizhou/Documents/OCT/60slice/split_data_2D_400/test/truth.npy"
    dataset = OCTDataset(img_np=patch_dir, label_np=truth_dir, surf=[0], vol_list=vol_list, transforms=rand_aug)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    test_patch = np.load(patch_dir, mmap_mode='r')
    test_truth = np.load(truth_dir, mmap_mode='r')
    _, axes = plt.subplots(1,4)
    axes[0].imshow(np.transpose(test_patch[vol_list[vol_index]*SLICE_per_vol+slice_index,].astype(np.float32)), cmap='gray')
    axes[0].plot(test_truth[vol_list[vol_index]*SLICE_per_vol+slice_index,:,0])
    for i, batch in enumerate(loader):
        if i == vol_index*SLICE_per_vol+slice_index:
            img = batch['img'].squeeze().numpy()
            gt = batch['gt'].squeeze(0).numpy()
            gt_g = batch['gt_g'].squeeze(0).numpy()
            gt_d = batch['gt_d'].squeeze(0).numpy()
            print(gt_g.shape)
            break
    axes[1].imshow(img, cmap='gray')
    axes[1].plot(gt[:400])
    axes[2].imshow(gt_g[:,:,0])
    axes[2].plot(gt[:400])
    axes[3].plot(gt_d)
    
    plt.show()
    

