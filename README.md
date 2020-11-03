# CenterNet-CondInst
[CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850) + [CondInst: Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664) 

## Installation
Please refer to [CnterNet INSTALL.md](readme/INSTALL.md) for installation instructions.

## Training
```bash
## note : seg_weight default setting is 1. You can set it to other value to get better performance.
cd src
python main.py ctseg --exp_id coco_dla_1x --batch_size 4 --lr 1.25e-4 --gpus 0 --num_workers 4 --num_epochs 70
```
## Eval
```bash
## not support flip test and multi scale test
cd src
python test.py ctseg --exp_id coco_dla_1x --resume
```
## Visualization
```bash
cd src
python demo.py ctseg --exp_id coco_dla_1x --load_model ../exp/ctseg/coco_dla_1x/epoch_75.pth --demo ../images/
```

## Result
(TODO)

## Reference
1. [CenterNet](https://github.com/xingyizhou/CenterNet)
2. [CondInst](https://github.com/Epiphqny/CondInst)
3. [CondInst-AdelaiDet](https://github.com/aim-uofa/AdelaiDet/)
4. [CenterNet-CondInst](https://github.com/CaoWGG/CenterNet-CondInst)