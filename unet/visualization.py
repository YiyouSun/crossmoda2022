import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np

base_path = '../model/MoDA/Mean_Teacher_t2_180/Prediction/'
test_path = '../data/unet/test.txt'
#name = 'crossmoda2022_etz_101'
with open(test_path, 'r') as f:
    name_list = [a.strip() for a in f.readlines()]

for name in name_list:
    #name = 'crossmoda2021_ldn_100'
    label_path = os.path.join(base_path, f'{name}_lab.nii.gz')
    img_path = os.path.join(base_path, f'{name}_img.nii.gz')
    prediction_path = os.path.join(base_path, f'{name}_pred.nii.gz')

    img = nib.load(img_path).get_fdata().transpose(1,2,0)
    label = nib.load(label_path).get_fdata().transpose(1,2,0)
    pred = nib.load(prediction_path).get_fdata().transpose(1,2,0)

    plt.figure()
    plt.subplot(1,3,1)
    plt.title("Synthesized T2")
    #plt.title("Original T1")
    plt.imshow(img[:,:,2], cmap='gray')
    plt.subplot(1,3,2)
    plt.title("segmentation map")
    plt.imshow(np.max(label[:,:],axis=2)*50, cmap='gray')
    plt.subplot(1,3,3)
    plt.title("predication map")
    plt.imshow(np.max(pred[:,:],axis=2)*50, cmap='gray')
    plt.savefig(os.path.join(base_path, "%s.png"%(name)))