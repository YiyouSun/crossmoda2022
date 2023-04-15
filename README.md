# crossmoda2022
medical imaging benchmark of unsupervised cross-modality Domain Adaptation approaches (from contrast-enhanced T1 to high-resolution T2)

Project Page:https://crossmoda2022.grand-challenge.org/ .

Implementation is based algorithm came up with by ne2e group which has best performance on segmentation task. Paper: https://arxiv.org/abs/2109.14219 .  
Two deeplearning models are required to be trained to generate segmentation maps on non-labeled hrT2 images. 


### Dataset structure:
```
data/
├── train_source
├── train_target
├── validation
├── unet

├── ...
```
t1 scans are saved in train_source, while t2 scans are saved in train_target for training NiceGAN. Validation set is used for testing.  
sythesized hrT2 and hrT2 are saved in unet for training and testing segmentation model

### build environments
python version=3.8
torch=2.0.0
```
pip install -r requirements.txt
```

## NiceGAN:
Pixel alignment transfers the scans in ceT1 to that in hrT2, helping to reduce domain shift in the training segmentation model. Backbone of NiceGAN model is introduced in https://arxiv.org/abs/2003.00273 .


### train NiceGAN(extension version of cycled GAN)
```
python trainGAN.py 
```
### output 20000 iteration, best performed model
![Image text](https://github.com/YiyouSun/crossmoda2022/blob/main/results/figures/t1_0020000_3sample.png)  
* 10 samples are splited from train_source as validation(3 of 10 are showed here), top images show cropped, resized, flipped 2-D slice of original t1 scans.   
* Middle images show output of t1_to_t2 generator of NiceGAN, due to the limitation of input image size and reduction of parameters and flops of model, transferred images' quality is not as good as original t1 scans.   
* Bottom images show output of t1_to_t2_t1 generator of NiceGAN, these images are used to calculate cycle loss between original t1 and transfered back t1 in model.   


## semi-supervised Unet:
Self-training adapts the decision boundary of the segmentation network to fit the distribution of hrT2 scans. Refence code: https://github.com/HiLab-git/SSL4MIS

### train Unet
```
cd unet/code
./semi_train3d.sh
```

### semi-superviesed learning on original t1 scans and labels: predication on validation t1 2-D slice
  
![Image text](https://github.com/YiyouSun/crossmoda2022/blob/main/results/figures/unet_t1.png)  

### semi-superviesed learning on Synthesized t2 scans, labels and original t2 scans: prediaction on validation t2 2-D slice

