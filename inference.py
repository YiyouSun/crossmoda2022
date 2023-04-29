import argparse
from params import params

parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = params(parser)

# create folders
p.create_results_folders()
logger = p.set_up_logger("inference_log.txt")

train_source = p.load_t1_files()
train_target,val_target = p.load_t2_files()
p.build_model()
p.load()

p.inference()