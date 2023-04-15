# crossmoda2022
medical imaging benchmark of unsupervised cross-modality Domain Adaptation approaches (from contrast-enhanced T1 to high-resolution T2)

Project Page:https://crossmoda2022.grand-challenge.org/ .

Implementation is based algorithm came up with by ne2e group which has best performance on segmentation task. Paper: https://arxiv.org/abs/2109.14219 .  
Two deeplearning models are required to be trained to generate segmentation maps on non-labeled hrT2 images. 


###Dataset structure:
```
data/
├── train_source
├── train_target
├── validation
├── ...
```
t1 scans are saved in train_source, while t2 scans are saved in train_target for training. Validation set is used for testing.

##NiceGAN:
Pixel alignment transfers the scans in ceT1 to that in hrT2, helping to reduce domain shift in the training segmentation model. Backbone of NiceGAN model is introduced in https://arxiv.org/abs/2003.00273 .

### build environments
python version=3.8
torch=2.0.0
```
pip install -r requirements.txt
```
### train NiceGAN(extension version of cycled GAN)
```
python trainGAN.py 
```
10 samples are splited from train_source as validation, top images show  cropped, resized, flipped 2-D slice of original t1 scans. 
Middle images show output of t1_to_t2 generator of NiceGAN, due to the limitation of input image size and reduction of parameters and flops of model, transferred images' quality is not as good as original t1 scans. 
Bottom images show output of t1_to_t2_t1 generator of NiceGAN, these images are used to calculate cycle loss between original t1 and transfered back t1 in model. 
![Image text](https://raw.githubusercontent.com/hongmaju/light7Local/master/img/productShow/20170518152848.png)

##nn-Unet:
Self-training adapts the decision boundary of the segmentation network to fit the distribu- tion of hrT2 scans. nn-Unet is introduced in https://arxiv.org/abs/1904.08128 .


