import argparse
import monai
from params import params
import monai
import numpy as np
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    SplitChanneld,
    ToTensord,
)
from monai.transforms import (
    LoadNifti,
    AddChannel,
    NormalizeIntensity,
    SpatialPad,
    RandFlip,
    RandSpatialCrop,
    Orientation,
    SplitChannel,
    ToTensor,
)


parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = params(parser)

# create folders
p.create_results_folders()

# set up logger
logger = p.set_up_logger("training_log.txt")


train_source = p.load_t1_files()
train_target,val_target = p.load_t2_files()


# define the transforms
train_source_transforms,train_target_transforms,val_source_transforms,val_target_transforms = p.get_transforms_niceGAN()
# check transforms
#p.check_transforms_on_first_source_train_image_and_label(train_source, train_source_transforms)

p.check_transforms_t1_t2("t1",train_source, train_source_transforms)
p.check_transforms_t1_t2("t2",train_target, train_target_transforms)
p.check_transforms_t1_t2("t2",val_target, val_target_transforms)

monai.utils.set_determinism(seed=0)
# build dataloader
p.build_loader()
#p.plot_slice()

#build model
p.build_model()
p.train()
'''

#load data list

train_source, train_target,val_target = p.load_t1_t2_files()
# Set deterministic training for reproducibility
train_transforms,val_transforms = p.get_transforms_niceGAN()

train_transforms = Compose(
            [
                LoadNifti(image_only =True), 
                
                AddChannel(),
                Orientation( axcodes="RAS"),
                NormalizeIntensity(),
                
                ToTensor(),
            ]
        )

p.check_transforms_t1_t2("t1",train_source, train_transforms)
p.check_transforms_t1_t2("t2",train_target, train_transforms)
p.check_transforms_t1_t2("t2",val_target, val_transforms)



'''