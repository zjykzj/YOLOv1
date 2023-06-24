
# README

## YOLOv1

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch yolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "31226" main_amp.py --arch yolov1 -b 256 --workers 4 --opt-level O1 --resume weights/yolov1_224/model_best.pth.tar --evaluate ./imagenet/
 * Prec@1 69.886 Prec@5 89.272
```

## YOLOv1 Extended Version 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch yolov1_s14 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "31226" main_amp.py --arch yolov1_s14 -b 256 --workers 4 --opt-level O1 --resume weights/yolov1_s14_224/model_best.pth.tar --evaluate ./imagenet/
 * Prec@1 73.828 Prec@5 91.314
```

## FastYOLOv1

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch fastyolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "36126" main_amp.py --arch fastyolov1 -b 256 --workers 4 --opt-level O1 --resume weights/fastyolov1_224/model_best.pth.tar --evaluate ./imagenet/
 * Prec@1 63.046 Prec@5 83.796
```

## FastYOLOv1 Extended Version 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch fastyolov1_s14 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port "36126" main_amp.py --arch fastyolov1_s14 -b 256 --workers 4 --opt-level O1 --resume weights/fastyolov1_s14_224/model_best.pth.tar --evaluate ./imagenet/
 * Prec@1 66.530 Prec@5 86.468
```