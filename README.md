# crossmoda2022
medical imaging benchmark of unsupervised cross-modality Domain Adaptation approaches (from contrast-enhanced T1 to high-resolution T2)

Project Page:https://crossmoda2022.grand-challenge.org/

Implementation is based on: https://arxiv.org/abs/2109.14219 which has best performance on segmentation task. The alogorithm is divided into two parts of training. 

Pixel alignment transfers the scans in ceT1 to that in hrT2, helping to reduce domain shift in the training segmentation model. Backbone of NiceGAN model is introduced in https://arxiv.org/abs/2003.00273 with github link https://github.com/alpc91/NICE-GAN-pytorch. 

Self-training adapts the decision boundary of the segmentation network to fit the distribu- tion of hrT2 scans. Training is based on nn-unet which is introduced in https://arxiv.org/abs/1904.08128 with github link https://github.com/MIC-DKFZ/nnUNet.
