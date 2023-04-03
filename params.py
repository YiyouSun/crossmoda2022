import os, torch
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
from time import strftime
import time
import itertools
import argparse
import logging
from matplotlib import pyplot as plt
import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
)
from monai.networks.layers import Norm

from monai.data import NiftiSaver

from networks import *
from thop import profile
from thop import clever_format


class params:
    def __init__(self,parser):
        
        parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")
        parser.add_argument("--split", type=str, default="./data/infos_source_training.csv", help="path to CSV file that defines"
                                                                                         " training, validation and"
                                                                                         " test datasets")
        
        parser.add_argument('--light', type=str2bool, default=False, help='[NICE-GAN full version / NICE-GAN light version]')
        parser.add_argument('--iteration', type=int, default=1000, help='The number of training iterations')
        parser.add_argument("--train_batch_size", type=int, default=1, help="batch size of the forward pass")
        parser.add_argument("--initial_learning_rate", type=float, default=1e-4, help="learning rate at first epoch")
        parser.add_argument('--print_freq', type=int, default=10, help='The number of image print freq')
        parser.add_argument('--save_freq', type=int, default=100, help='The number of model save freq')
        parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

        

        parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
        parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
        parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
        parser.add_argument('--recon_weight', type=int, default=10, help='Weight for Reconstruction')

        parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
        parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
        parser.add_argument('--n_dis', type=int, default=7, help='The number of discriminator layer')

        parser.add_argument('--img_size', type=int, default=256, help='The size of image')
        parser.add_argument('--img_ch', type=int, default=1, help='The size of image channel')
        
        parser.add_argument(
            "--cuda", type=str, default="cuda:1", help="name of results folder"
        )   


        parser.add_argument(
            "--results_folder_name", type=str, default="temp", help="name of results folder"
        )   


        args = parser.parse_args()
        self.data_root = "./data"
        self.root = "./"
        self.num_workers = 4
        self.epochs_with_const_lr = 100
        
    
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
        self.val = 0.15 # validation set percentage
        if self.debug:
            self.val = 0.3

        self.train_batch_size = args.train_batch_size
        self.lr = args.initial_learning_rate
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        if self.debug:
            self.ch = 32


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
        
        
        self.sptial_shape_t1 = [512, 512, 120]
        self.sptial_shape_t2 = [448, 448, 80]
        self.pad_crop_shape = [384, 384, 64]
        if self.debug:
            self.pad_crop_shape = [128, 128, 32]
        self.pad_crop_shape_test = [384, 384, 64]
        if self.debug:
            self.pad_crop_shape_test = [128, 128, 32]
        self.img_size = self.pad_crop_shape[0]
        
        self.results_folder_path = os.path.join(self.root, "results/", args.results_folder_name)
        #if self.debug:
            #self.results_folder_path = os.path.join(self.root, "results", "debug")
        if self.results_foler_name == "TRUE":
            self.results_folder_path = os.path.join(self.root, "results", "temp" + strftime("%Y%m%d%H%M%S"))
        self.inference_path =  os.path.join(self.results_folder_path, "inference")
        self.logs_path = os.path.join(self.results_folder_path, "logs")
        self.model_path = os.path.join(self.results_folder_path, "model")
        self.figures_path = os.path.join(self.results_folder_path, "figures")

        self.device = torch.device(args.cuda)
    
    
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
        self.train_source_transforms = Compose(
            [
                LoadNiftid(keys=["t1"]),
                AddChanneld(keys=["t1"]),
                Orientationd(keys=["t1"], axcodes="RAS"),
                NormalizeIntensityd(keys=["t1"]),
                SpatialPadd(keys=["t1"], spatial_size=self.pad_crop_shape),
                RandSpatialCropd(
                    keys=["t1"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                ),
                
                RandFlipd(keys=["t1"], prob=0.5, spatial_axis=0),              
                ToTensord(keys=["t1"]),

            ]
        )
        self.train_target_transforms = Compose(
            [
                LoadNiftid(keys=["t2"]),
                AddChanneld(keys=["t2"]),
                Orientationd(keys=["t2"], axcodes="RAS"),
                NormalizeIntensityd(keys=["t2"]),
                SpatialPadd(keys=["t2"], spatial_size=self.pad_crop_shape),
                RandSpatialCropd(
                    keys=["t2"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                ),
                
                RandFlipd(keys=["t2"], prob=0.5, spatial_axis=0),           
                ToTensord(keys=["t2"]),
            ]
        )
        
        self.val_source_transforms = Compose(
            [
                LoadNiftid(keys=["t1"]),
                
                ToTensord(keys=["t1"]),
            ]
        )    
        
        self.val_target_transforms = Compose(
            [
                LoadNiftid(keys=["t2"]),
         
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
        
        image2 = image.transpose(1,2)
        image2 = image2.transpose(0,1)
        plt.subplot(1,3,2)
        plt.imshow(image2[slice_idx, :, :], cmap="gray", interpolation="none")
        
        '''
        view() cannot work nice, see in check_image6.png
        image3 = check_data['t1'].view([32,128,128])
        plt.subplot(1,3,3)
        plt.imshow(image3[slice_idx, :, :], cmap="gray", interpolation="none")
        '''
        image3 = check_data['t1'].permute(4,0,2,3,1).squeeze(4)
        plt.subplot(1,3,3)
        plt.imshow(image3[slice_idx,0, :, :], cmap="gray", interpolation="none")
        
        plt.savefig(os.path.join(self.figures_path, "check_image8.png"))

    def build_model(self):

        self.logger.info("Building NiceGAN model, loss functions, and optimizers..")
        
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light =self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.pad_crop_shape[0], self.pad_crop_shape[1]]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        _,_, _,  _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

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
        logger = self.logger
        slice_idx = 20
        
        plt.figure("check top:t1, middle:transferd t1, bottom faket1", (3*len(data_loader), 6))
        plt.title("%s"%(modality))
        logger.info("Generating map from given {} images, with {} samples".format(modality,len(data_loader)))
        for i, batch_data in enumerate(data_loader):
            logger = self.logger
            real = batch_data[modality].permute(4,0,2,3,1).squeeze(4)
            real = real.to("cpu")
            _,  _,  _, _, real_z = self.disA(real) 
            fake = self.gen2B(real_z)
            fake = fake.detach()
            
            _, _, _, _, fake_z = self.disB(fake)  
            fake_t1 = self.gen2A(fake_z)
            fake_t1 = fake_t1.detach()

            logger.info("image shape: {}, slice = {}".format(fake.shape, slice_idx))
            
            plt.subplot(3,len(data_loader),i+1)
            
            plt.imshow(real[slice_idx,0, :, :], cmap="gray", interpolation="none") 
            
            plt.subplot(3,len(data_loader),i+len(data_loader)+1)
            plt.imshow(fake[slice_idx,0, :, :], cmap="gray", interpolation="none") 
            
            plt.subplot(3,len(data_loader),i+2*len(data_loader)+1)
            plt.imshow(fake_t1[slice_idx,0, :, :], cmap="gray", interpolation="none") 
            
            plt.savefig(os.path.join(self.figures_path, "%s_%07d.png"%(modality,step)))

    def train(self):
        logger = self.logger
        logger.info("Start training..")

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
            
            for batch_source, batch_target in zip(self.train_source_loader,self.train_target_loader):
                
                
                real_A = batch_source['t1'].permute(4,0,2,3,1).squeeze(4)
                real_B = batch_target['t2'].permute(4,0,2,3,1).squeeze(4)
                
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                #print(real_A.shape,real_B.shape)

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

            logger.info("{}/{}  time: {:4.2f} d_loss: {:.4f}, g_loss: {:.4f}".format(step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))


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

    def load(self):
        params = torch.load(os.path.join(self.model_path, 'params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter'] 
    
    def inference(self):
        '''
        run self.load_t1_files() before use this 
        if model is not trained, run self.load()
        this function is used to create synthesized hrT2 scans from ceT1 scans in datalist self.train_source
        '''
        logger = self.logger
        self.load()
        logger.info("inferencing on the train source set")
        transforms =  Compose(
            [
                LoadNiftid(keys=["t1","label"]),
                AddChanneld(keys=["t1","label"]),
                Orientationd(keys=["t1","label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["t1","label"]),
                SpatialPadd(keys=["t1","label"], spatial_size=self.pad_crop_shape),
                RandSpatialCropd(
                    keys=["t1","label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                ),                   
                ToTensord(keys=["t1","label"]),

            ]
        )

        train_source_ds = monai.data.Dataset(data=self.train_source, transform=transforms)
        
        train_source_loader = DataLoader(
            train_source_ds,
            batch_size=self.train_batch_size,
            shuffle=False,
            #sampler = BatchSampler(SequentialSampler(train_source_ds), batch_size=self.ch, drop_last=False),
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )

        for i, batch_data in enumerate(train_source_loader):
            # size after permute 32 1 128 128(debug)
            real_I = batch_data['t1'].permute(4,0,2,3,1).squeeze(4)
            real_I = real_I.to("cpu")
            _,  _,  _, _, real_I_z = self.disA(real_I)

            fakeI = self.gen2B(real_I_z)
            fakeI = fakeI.permute(1,2,3,0)
            fakeI = fakeI.detach()

            # generate persudo labels in hrT2 domain as well
            real_L = batch_data['label'].permute(4,0,2,3,1).squeeze(4)
            real_L = real_L.to("cpu")
            _,  _,  _, _, real_L_z = self.disA(real_L)

            fakeL = self.gen2B(real_L_z)
            fakeL = fakeL.permute(1,2,3,0)
            fakeL = fakeL.detach()


            
            






