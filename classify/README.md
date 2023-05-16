
# README

## YOLOv1

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch yolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```text
 * Prec@1 73.768 Prec@5 91.314
```

## FastYOLOv1

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py --arch fastyolov1 -b 256 --workers 4 --lr 0.1 --weight-decay 1e-5 --epochs 120 --opt-level O1 ./imagenet/
```

```text
 * Prec@1 66.590 Prec@5 86.248
```
