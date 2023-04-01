# crossmoda2022
medical imaging benchmark of unsupervised cross-modality Domain Adaptation approaches (from contrast-enhanced T1 to high-resolution T2)

Project Page:https://crossmoda2022.grand-challenge.org/

Implementation is based algorithm came up with by ne2e group which has best performance on segmentation task. Paper: https://arxiv.org/abs/2109.14219 .  
Two deeplearning models are required to be trained to generate segmentation maps on non-labeled hrT2 images. 

NiceGAN:
Pixel alignment transfers the scans in ceT1 to that in hrT2, helping to reduce domain shift in the training segmentation model. Backbone of NiceGAN model is introduced in https://arxiv.org/abs/2003.00273 .

nn-Unet:
Self-training adapts the decision boundary of the segmentation network to fit the distribu- tion of hrT2 scans. nn-Unet is introduced in https://arxiv.org/abs/1904.08128 .
