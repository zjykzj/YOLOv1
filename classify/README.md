
# README

## YOLOv1

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -a yolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 90 --opt-level O1 ../dataset/imagenet/ 
```

## FastYOLOv1

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31266" main_amp.py -a fast-yolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 90 --opt-level O1 ../dataset/imagenet/
```