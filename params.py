import os, torch
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import random_split
from time import strftime
import time
import itertools
import argparse
import logging
from matplotlib import pyplot as plt
import monai
from monai.transforms import (
    Compose,
    LoadImaged,   
    AddChanneld,
    ScaleIntensityd,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    CenterSpatialCropd,
    Resized,
    ToTensord,
)
from monai.networks.layers import Norm

from monai.data import NiftiSaver

from networks import *
from utils import *

from thop import profile
from thop import clever_format
from monai.inferers import sliding_window_inference
from monai.inferers import Inferer
from monai.inferers import SlidingWindowInferer

from monai.data import NiftiSaver
from monai.transforms import SaveImage
import nibabel as nib
from monai.data import write_nifti

class params:
    def __init__(self,parser):
        
        parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")
        parser.add_argument("--split", type=str, default="./data/infos_source_training.csv", help="path to CSV file that defines"
                                                                                         " training, validation and"
                                                                                         " test datasets")
        
        parser.add_argument('--light', type=str2bool, default=False, help='[NICE-GAN full version / NICE-GAN light version]')
        parser.add_argument('--iteration', type=int, default=300000, help='The number of training iterations')
        parser.add_argument("--train_batch_size", type=int, default=1, help="batch size of the forward pass")
        parser.add_argument("--initial_learning_rate", type=float, default=1e-4, help="learning rate at first epoch")
        parser.add_argument('--print_freq', type=int, default=100, help='The number of image print freq')
        parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
        parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
        parser.add_argument('--num_workers', type=int, default=1, help='The number of parallel workers for batch training')

        

        parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
        parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
        parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
        parser.add_argument('--recon_weight', type=int, default=10, help='Weight for Reconstruction')

        parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
        parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
        parser.add_argument('--n_dis', type=int, default=7, help='The number of discriminator layer')

        parser.add_argument('--img_size', type=int, default=256, help='The size of image')
        parser.add_argument('--img_ch', type=int, default=32, help='The size of image channel, e.g RGB 2-D image channel=3')
        
        parser.add_argument(
            "--cuda", type=str, default="cuda:1", help="gpu id"
        )


        parser.add_argument(
            "--results_folder_name", type=str, default="temp", help="name of results folder"
        )   


        args = parser.parse_args()
        self.data_root = "./data"
        self.root = "./"
        self.num_workers = args.num_workers
        
        
    
        self.debug = args.debug
        self.split_csv = args.split
        self.results_foler_name = args.results_folder_name
        

        self.target_split_csv = "./data/infos_target_training.csv"
        self.validation_split_csv = "./data/infos_target_validation.csv"
        
        if self.debug:
            self.split_csv = "./data/debug_training.csv"
            self.target_split_csv="./data/debug_target_training.csv"
            self.validation_split_csv = "./data/debug_target_validation.csv"
            
        
        """parameters for training"""
        self.light = args.light
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.val = 0.05 # validation set percentage
        if self.debug:
            self.val = 0.3

        self.train_batch_size = args.train_batch_size
        self.lr = args.initial_learning_rate
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        


        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight
        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        #self.img_size = args.img_size
        self.img_ch = args.img_ch
        

        # print(self.train_batch_size)
        # generate the results
        
        
        self.sptial_shape_t1 = [256, 256, 64]
        self.sptial_shape_t2 = [256, 256, 64]
        #### this model would reach over 200GB with params and flops
        #self.pad_crop_shape = [384, 384, 64]
        #self.pad_crop_shape_test = [384, 384, 64] 
        
        
        self.pad_crop_shape = [128, 128, 32]
        
        self.img_ch = 32

        #self.pad_crop_shape = [256, 256, 32]
        
        
        if self.debug:
            self.pad_crop_shape = [128, 128, 32]
            self.pad_crop_shape_test = [128, 128, 32]
        
        self.img_size = self.pad_crop_shape[0]
        
        self.results_folder_path = os.path.join(self.root, "results/", args.results_folder_name)
        #if self.debug:
            #self.results_folder_path = os.path.join(self.root, "results", "debug")
    
        
        self.logs_path = os.path.join(self.results_folder_path, "logs")
        self.model_path = os.path.join(self.results_folder_path, "model")
        self.figures_path = os.path.join(self.results_folder_path, "figures")
        
        self.inference_path =  os.path.join(self.results_folder_path, "inference")
        self.ht2_path =  os.path.join(self.inference_path, "ht2")
        self.plabel_path =  os.path.join(self.inference_path, "plabel")

        self.device = torch.device(args.cuda)
        
        self.train_flag = False
    
    
    def create_results_folders(self):
        # create results folders for logs, figures and model
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path, exist_ok=False)
            os.chmod(self.logs_path, 0o777)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=False)
            os.chmod(self.model_path, 0o777)
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path, exist_ok=False)
            os.chmod(self.figures_path, 0o777)
        if not os.path.exists(self.inference_path):
            os.makedirs(self.inference_path, exist_ok=False)
            os.chmod(self.inference_path, 0o777)
        if not os.path.exists(self.ht2_path):
            os.makedirs(self.ht2_path, exist_ok=False)
            os.chmod(self.ht2_path, 0o777)
        if not os.path.exists(self.plabel_path):
            os.makedirs(self.plabel_path, exist_ok=False)
            os.chmod(self.plabel_path, 0o777)
    
    def set_up_logger(self, log_file_name):
        # logging settings
        
        self.logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.logs_path, log_file_name), mode="w")
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s        %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Created " + log_file_name)
        
        return self.logger
    
    def load_t1_files(self):
        logger = self.logger
        logger.info("loading source training ser")
        with open(self.split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            next(csvReader)
            train_source = []
            for row in csvReader:
                t1 = os.path.join(self.data_root,'training_source',row[0]+'_ceT1.nii.gz')
                label = os.path.join(self.data_root,'training_source',row[0]+'_Label.nii.gz')
                mask = os.path.join(self.data_root,'training_source_GIF',row[0]+'_GIFoutput.nii.gz')
                
                train_source.append({'t1':t1,"label":label,"mask":mask})
                
        #check if all files exist
        for file_dict in train_source:
            assert (os.path.isfile(file_dict['t1'])), f" {file_dict['t1']} is not a file"
            assert (os.path.isfile(file_dict['label'])), f" {file_dict['label']} is not a file"
            assert (os.path.isfile(file_dict['mask'])), f" {file_dict['mask']} is not a file"
        
        logger.info("Number of images in training source set   = {}".format(len(train_source)))
        logger.info("training source set   = {}".format(train_source))
        self.train_source = train_source
        return train_source
    

    def load_t2_files(self):
        logger = self.logger
        logger.info("loading target training set")
        with open(self.target_split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            next(csvReader)
            train_target = []
            for row in csvReader:
                t2 = os.path.join(self.data_root,'training_target',row[0]+'_hrT2.nii.gz')
                train_target.append({"t2":t2})
        
        with open(self.validation_split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            next(csvReader)
            val_target = []
            for row in csvReader:
                val_t2 = os.path.join(self.data_root,'validation',row[0]+'_hrT2.nii.gz')
                val_target.append({"t2":val_t2})
                
        for file_dict in train_target + val_target:
            assert (os.path.isfile(file_dict['t2'])), f" {file_dict['t2']} is not a file"
            
        logger.info("Number of images in train target set       = {}".format(len(train_target)))
        logger.info("Number of images in validation target set = {}".format(len(val_target)))
        
        logger.info("train target set       = {}".format(train_target))
        logger.info("validation set = {}".format(val_target))
        
        self.train_target =train_target
        self.val_target = val_target
        return train_target, val_target 

    def get_transforms_niceGAN(self):
        self.logger.info("Getting transforms...")
        # Setup transforms of data sets
        # for nice training , make cropping random_center = TRUE
        self.train_source_transforms = Compose(
            [
                #LoadNiftid(keys=["t1"]),
                LoadImaged(keys=["t1"]),
                AddChanneld(keys=["t1"]),
                Orientationd(keys=["t1"], axcodes="RAS"),
                #ScaleIntensityd(keys=['t1'],minv=0.0, maxv=255.0),
                NormalizeIntensityd(keys=["t1"]),
                SpatialPadd(keys=["t1"], spatial_size=self.sptial_shape_t1),
                RandSpatialCropd( keys=["t1"], roi_size=self.sptial_shape_t1,random_center=True, random_size=False),
                Resized(keys=['t1'],spatial_size = self.pad_crop_shape,mode='area'),
                RandFlipd(keys=["t1"], prob=0.5, spatial_axis=2),              
                ToTensord(keys=["t1"]),
            ]
        )
        self.train_target_transforms = Compose(
            [
                #LoadNiftid(keys=["t2"]),
                LoadImaged(keys=["t2"]),
                AddChanneld(keys=["t2"]),
                Orientationd(keys=["t2"], axcodes="RAS"),
                #ScaleIntensityd(keys=['t2'],minv=0.0, maxv=255.0),
                NormalizeIntensityd(keys=["t2"]),
                SpatialPadd(keys=["t2"], spatial_size=self.sptial_shape_t2),
                RandSpatialCropd( keys=["t2"], roi_size=self.sptial_shape_t2,random_center=True,random_size=False),
                Resized(keys=['t2'],spatial_size = self.pad_crop_shape,mode='area'),
                RandFlipd(keys=["t2"], prob=0.5, spatial_axis=2),           
                ToTensord(keys=["t2"]),
            ]
        )
        
        self.val_source_transforms = Compose(
            [
                #LoadNiftid(keys=["t1"]),
                LoadImaged(keys=["t1"]),
                ToTensord(keys=["t1"]),
            ]
        )    
        
        self.val_target_transforms = Compose(
            [
                #LoadNiftid(keys=["t2"]),
                LoadImaged(keys=["t2"]),
         
                ToTensord(keys=["t2"]),
            ]
        )    
        return self.train_source_transforms,self.train_target_transforms,self.val_source_transforms,self.val_target_transforms  
  
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))
    
   
    def check_transforms_t1_t2(self, modality, train, train_transforms):
        logger = self.logger
        slice_index = 40
        # check the transforms on the first validation set image
        check_ds = monai.data.Dataset(data=train, transform=train_transforms)  # apply transform
        check_loader = DataLoader(check_ds, batch_size=2)
        check_data = monai.utils.misc.first(check_loader)  # gets the first item from an input iterable
        
        logger.info("-" * 10)
        
        logger.info(
            "Length of check_data = {}".format(len(check_data))
        )  # this dictionary also contains all the nifti header info
        if modality == "t1":
            logger.info("Check the transforms on the first train source set image")
            logger.info("check_data['image'].shape = {}".format(check_data["t1"].shape))
        else :
            logger.info("Check the transforms on the first train target set image")
            logger.info("check_data['image'].shape = {}".format(check_data["t2"].shape))
        
        image = check_data['t1']
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(image[0,0, slice_index, :, :], cmap="gray", interpolation="none")
        plt.subplot(1,3,2)
        plt.imshow(image[0,0,:, slice_index, :], cmap="gray", interpolation="none")
        plt.subplot(1,3,3)
        plt.imshow(image[0,0,:,:,20], cmap="gray", interpolation="none")
        
        plt.savefig(os.path.join(self.figures_path, "hiiiiiii4.png"))
        


    def build_loader(self):
        self.logger.info("Building training and validation source dataloader...")
        # Define CacheDataset and DataLoader for training and validation
        
        # seprate validation set from train source set
        train_source, val_source = random_split(self.train_source,[1-self.val,self.val],generator=torch.Generator().manual_seed(0))
        # Use cacheDataset if GPU storage is sufficient
        train_source_ds = monai.data.Dataset(data=train_source, transform=self.train_source_transforms)
        val_source_ds = monai.data.Dataset(data=val_source, transform=self.train_source_transforms)
        
        self.train_source_loader = DataLoader(
            train_source_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            #sampler = BatchSampler(SequentialSampler(train_source_ds), batch_size=self.ch, drop_last=False),
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )

        self.val_source_loader = DataLoader(
            val_source_ds,
            batch_size=self.train_batch_size,
            shuffle=False,
            #sampler = BatchSampler(SequentialSampler(train_source_ds), batch_size=self.ch, drop_last=False),
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )

        self.logger.info("Building training and validation target dataloader...")
        # Define CacheDataset and DataLoader for training and validation
        
        # seprate validation set from train set
        # no need to keep same seed since from source and target there is no pair relationship
        train_target, val_target = random_split(self.train_target,[1-self.val,self.val],generator=torch.Generator().manual_seed(0))
        train_target_ds = monai.data.Dataset(data=train_target, transform=self.train_target_transforms)
        val_target_ds = monai.data.Dataset(data=val_target, transform=self.train_target_transforms)

        self.train_target_loader = DataLoader(
            train_target_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )

        self.val_target_loader = DataLoader(
            val_target_ds,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )
        
        return self.train_source_loader,self.val_source_loader,self.train_target_loader,self.val_target_loader
        '''
        self.logger.info("Caching test target data set...")
        # Define CacheDataset and DataLoader for training and validation
        val_target_ds = monai.data.CacheDataset(
            data=self.val_target, transform=self.val_target_transforms, cache_rate=1.0, num_workers=self.num_workers
        )
        self.val_target_loader = DataLoader(
            val_target_ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )
        '''
        
    def plot_slice(self):
        logger = self.logger
        check_data = monai.utils.misc.first(self.train_source_loader)  # gets the first item from an input iterable
        #check_data = monai.utils.misc.first(self.train_target_loader)  # gets the first item from an input iterable
        slice_idx = 20
        image = check_data['t1']
        image = image.view(image.shape[-3:])
        logger.info("Plot one slice of the image and the label")
        logger.info("image shape: {}, slice = {}".format(image.shape, slice_idx))
        # plot the slice [:, :, slice]
        plt.figure("check", (18, 6))
        plt.title("image")
        plt.subplot(1,3,1)
        plt.imshow(image[:, :, slice_idx], cmap="gray", interpolation="none")
        
        image2 = check_data['t1'].permute(0,1,4,3,2).squeeze(4)
        
        plt.subplot(1,3,2)
        plt.imshow(image2[0,0,slice_idx, :, :], cmap="gray", interpolation="none")
        
        '''
        view() cannot work nice, see in check_image6.png
        image3 = check_data['t1'].view([32,128,128])
        plt.subplot(1,3,3)
        plt.imshow(image3[slice_idx, :, :], cmap="gray", interpolation="none")
        '''
        image3 = check_data['t1'].permute(0,1,4,2,3).squeeze(4)
        plt.subplot(1,3,3)
        plt.imshow(image3[0,0,slice_idx, :, :], cmap="gray", interpolation="none")
        
        plt.savefig(os.path.join(self.figures_path, "check_image8.png"))

    def build_model(self):

        
        logger = self.logger
        logger.info("Building NiceGAN model, loss functions, and optimizers..")
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light =self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.pad_crop_shape[0], self.pad_crop_shape[1]]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        
        logger.info('[Network {}] Total number of parameters:  {}'.format("disA",params))
        logger.info('[Network {}] Total number of FLOPs: {} '.format("disA",macs))
        logger.info('-----------------------------------------------')
        
        
        _,_, _,  _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        
        logger.info('[Network {}] Total number of parameters:  {} '.format("gen2B",params))
        logger.info('[Network {}] Total number of FLOPs: {} '.format("gen2B",macs))
        logger.info('-----------------------------------------------')
        

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(),self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        
    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, 'params_%07d.pt' % step))
        
    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.model_path, path_g))

    def generate_NiceGAN(self,modality, data_loader,step):
        """
        This function is used to check sample slices from unseen validation 
        set from training set that are transferred from t1 to t2 then back to t1 
        """
        logger = self.logger
        slice_idx = 20
        
        plt.figure("check top:t1, middle:transferd t1, bottom faket1", (3*len(data_loader), 6))
        plt.title("%s"%(modality))
        logger.info("Generating map from given {} images, with {} samples".format(modality,len(data_loader)))
        for i, batch_data in enumerate(data_loader):
            logger = self.logger
            real = batch_data[modality].permute(0,1,4,2,3).squeeze(1)
            real = real.to("cpu")
            _,  _,  _, _, real_z = self.disA(real) 
            fake = self.gen2B(real_z)
            fake = fake.detach()
            
            _, _, _, _, fake_z = self.disB(fake)  
            fake_t1 = self.gen2A(fake_z)
            fake_t1 = fake_t1.detach()

            logger.info("image shape: {}, slice = {}".format(fake.shape, slice_idx))
            
            plt.subplot(3,len(data_loader),i+1)
            
            plt.imshow(real[0,slice_idx, :, :], cmap="gray", interpolation="none") 
            
            plt.subplot(3,len(data_loader),i+len(data_loader)+1)
            plt.imshow(fake[0,slice_idx, :, :], cmap="gray", interpolation="none") 
            
            plt.subplot(3,len(data_loader),i+2*len(data_loader)+1)
            plt.imshow(fake_t1[0,slice_idx, :, :], cmap="gray", interpolation="none") 
            
            plt.savefig(os.path.join(self.figures_path, "%s_%07d.png"%(modality,step)))

    def train(self):
        logger = self.logger
        logger.info("Start training..")
        torch.cuda.empty_cache()

        # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

        self.start_iter = 1
        
        # training loop
        testnum = 4
        logger.info("self.start_iter.{}".format(self.start_iter))
        
        start_time = time.time()
        
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # real_A source data, real_B target data
            try:
                real_A = next(trainA_iter)
            except:
                trainA_iter = iter(self.train_source_loader)
                real_A= next(trainA_iter)

            try:
                real_B = next(trainB_iter)
            except:
                trainB_iter = iter(self.train_target_loader)
                real_B= next(trainB_iter)
                
            #print(real_A)
                
            #logger.info("Number of iteration :{}".format(step))
            #logger.info("The shape of batch_data from dataloader:{}".format(real_A['t1'].shape))    
                
            real_A = real_A['t1'].permute(0,1,4,2,3).squeeze(1)
            real_B = real_B['t2'].permute(0,1,4,2,3).squeeze(1)
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            #logger.info("The shape of batch_data after transposing:{}".format(real_A.shape))
            #logger.info("Cuda memory allocated after loading data {}".format(convert_size(torch.cuda.memory_allocated(self.device))))
         
            # Update D
            self.D_optim.zero_grad()

            real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(real_A)
            real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)


            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))            
            D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
            D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()
            # writer.add_scalar('D/%s' % 'loss_A', D_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('D/%s' % 'loss_B', D_loss_B.data.cpu().numpy(), global_step=step)  

            #logger.info("Cuda memory allocated after interation of discriminators {}".format(convert_size(torch.cuda.memory_allocated(self.device))))


            # Update G
            self.G_optim.zero_grad()

            _,  _,  _, _, real_A_z = self.disA(real_A)
            _,  _,  _, _, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(fake_A2B)

            fake_B2A2B = self.gen2B(fake_A_z)
            fake_A2B2A = self.gen2A(fake_B_z)

            #logger.info("Cuda memory allocated after foward propagation of generators {}".format(convert_size(torch.cuda.memory_allocated(self.device))))

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
            G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

            G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_z)
            fake_B2B = self.gen2B(real_B_z)

            G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B)


            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()
            # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)  

            # move this logger here could print step information for every patient
            logger.info("{}/{}  time: {:4.2f} d_loss: {:.4f}, g_loss: {:.4f}".format(step, self.iteration, time.time() - start_time, Discriminator_loss.cpu().detach().numpy().item(), Generator_loss.cpu().detach().numpy().item()))

            #logger.info("{}/{}  time: {:4.2f} d_loss: {:.4f}, g_loss: {:.4f}".format(step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))


            if step % self.save_freq == 0:
                self.save(os.path.join(self.model_path), step)

            if step % self.print_freq == 0:
                print('current D_learning rate:{}'.format(self.D_optim.param_groups[0]['lr']))
                print('current G_learning rate:{}'.format(self.G_optim.param_groups[0]['lr']))
                self.save_path("_params_latest.pt",step)

                self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

                self.gen2B,self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
                self.disA,self.disB = self.disA.to('cpu'), self.disB.to('cpu')

                #self.generate_NiceGAN("t1","train",self.train_source_loader,step)
                self.generate_NiceGAN("t1",self.val_source_loader,step)

                #  numpy does not support GPU, so remember to direct model back to GPU(self.device)
                self.gen2B,self.gen2A = self.gen2B.to(self.device), self.gen2A.to(self.device)
                self.disA,self.disB = self.disA.to(self.device), self.disB.to(self.device)
                self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()
        
        
        self.train_flag = True

    def load(self):
        params = torch.load(os.path.join(self.model_path, '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter'] 
    
    def predictor_t1(self, batch_data):
        
        _,  _,  _, _, real_I_z = self.disA(batch_data)
        fake_I  = self.gen2B(real_I_z)

        return fake_I

    def inference(self):
        '''
        run self.load_t1_files() before use this 
        if model is not trained, run self.load()
        this function is used to create synthesized hrT2 scans from ceT1 scans in datalist self.train_source
        '''
        
        logger = self.logger
        if self.train_flag==False:
            logger.info("loading the paramaters for model")
            self.load()

        logger.info("infering the train source set- t1 scans")
        
        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

        self.gen2B,self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
        self.disA,self.disB = self.disA.to('cpu'), self.disB.to('cpu')


        transforms = Compose(
            [
                #LoadNiftid(keys=["t2"]),
                LoadImaged(keys=["t1","label"]),
                AddChanneld(keys=["t1","label"]),
                NormalizeIntensityd(keys=["t1"]),
                SpatialPadd(keys=["t1","label"], spatial_size=self.sptial_shape_t1),
                RandSpatialCropd( keys=["t1","label"], roi_size=self.sptial_shape_t1,random_center=True),
                Resized(keys=['t1',"label"],spatial_size = self.pad_crop_shape,mode='area'),      
                ToTensord(keys=["t1","label"]),
            ]
        )    
        
        train_source_ds = monai.data.Dataset(data=self.train_source, transform=transforms)      
        train_source_loader = DataLoader(
            train_source_ds,
            batch_size=1,
            shuffle=False,  
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )
        slice_idx = 5
        for i, batch_data in enumerate(train_source_loader):
            logger.info("infering {} ".format( os.path.split(self.train_source[i]['t1'])[1]))
            # size after permute 1 32 128 128 size before permute: 1 128 128 32
            real_I = batch_data['t1'].permute(0,1,4,2,3).squeeze(1)
            real_I = real_I.to("cpu")
            print(real_I.shape)
            _,  _,  _, _, real_I_z = self.disA(real_I)
            output  = self.gen2B(real_I_z)

            print("hahahjieguo",output.shape)
            output = output.permute(0,2,3,1)
            output = output.detach()

            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(batch_data['t1'].permute(0,1,4,2,3).squeeze(1)[0,slice_idx,:,:],cmap="gray", interpolation="none")
            plt.subplot(2,2,2)
            plt.imshow(output[0,:,:,slice_idx],cmap="gray", interpolation="none")
            plt.subplot(2,2,3)
            plt.imshow(batch_data['label'].permute(0,1,4,2,3).squeeze(1)[0,slice_idx,:,:], cmap='gray', interpolation="none")
            #print(np.count_nonzero(batch_data['label'].permute(0,1,4,2,3).squeeze(1)[0,0:10,:,:]>0))
            data_image_name = os.path.split(self.train_source[i]['t1'])[1]
            plt.savefig(os.path.join(self.figures_path, "check_image_%s.png"%(data_image_name)))

            data_image_name = os.path.split(self.train_source[i]['t1'])[1]
            write_nifti(output, os.path.join(self.ht2_path,"%s"%(data_image_name)))

            data_label_name = os.path.split(self.train_source[i]['label'])[1]
            write_nifti(batch_data['label'].squeeze(1).to('cpu'), os.path.join(self.plabel_path,"%s"%(data_label_name)))
            
            plt.figure()
        
            
  
        self.gen2B,self.gen2A = self.gen2B.to(self.device), self.gen2A.to(self.device)
        self.disA,self.disB = self.disA.to(self.device), self.disB.to(self.device)                
                
                
    


            
            






