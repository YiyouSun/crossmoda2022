
cp -r model/MoDA/Mean_Teacher_189/unet_3D/unet_3D_best_model.pth model/MoDA/Mean_Teacher_190/unet_3D 
cp -r data/training_target/. data/unet/train_target
cp -r results/temp1/inference/ht2/. data/unet/training_source
cp -r results/temp1/inference/plabel/. data/unet/training_source
cp -r data/validation/. data/unet/train_source
nohup python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data --max_iterations 30000 --exp MoDA/Mean_Teacher --base_lr 0.01 &

nohup python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data/unet --max_iterations 30000 --exp MoDA/Mean_Teacher --base_lr 0.1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data --max_iterations 100000 --exp MoDA/Mean_Teacher_484 --base_lr 0.01
python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data --max_iterations 100000 --exp MoDA/Mean_Teacher_484 --base_lr 0.01 

# transferred command
python -u test_3D.py --root_path ../data/unet --num_classes 3 --exp MoDA/Mean_Teacher_189 --model unet_3D
# origianl t1 test command
python -u test_3D.py --root_path ../data --num_classes 3 --exp MoDA/Mean_Teacher_190 --model unet_3D

python visualization.py
python -u test_3D.py --root_path ../data/unet --num_classes 3 --exp MoDA/Mean_Teacher__190 --model unet_3D


## supervised training for synthesized t2
python train_mean_teacher_3D.py --labeled_num 180 --num_classes 3 --root_path ../data/unet --max_iterations 100000 --exp MoDA/Mean_Teacher_t2 --base_lr 0.01  --cuda 4
nohup python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data/unet --max_iterations 100000 --exp MoDA/Mean_Teacher_t2 --base_lr 0.01  --cuda "cuda:7"
python -u test_3D.py --root_path ../data/unet --num_classes 3 --exp MoDA/Mean_Teacher_t2_190 --model unet_3D
python visualization.py

## semi-supervised training for synthesized t2
nohup python train_mean_teacher_3D.py --labeled_num 190 --num_classes 3 --root_path ../data/unet --max_iterations 10000 --exp MoDA/Mean_Teacher_semi-t2 --base_lr 0.01  --cuda "cuda:7" --semi True --resume True