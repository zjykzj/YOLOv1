<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/YOLOv1"><img align="center" src="./imgs/YOLOv1.png" alt=""></a></div>

<p align="center">
  «YOLOv1» reproduced the paper "You Only Look Once"
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

* Train using the `VOC07+12 trainval` dataset and test using the `VOC2007 Test` dataset with an input size of `448x448`. give the result as follows

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zkss{background-color:#FFF;border-color:inherit;color:#333;text-align:center;vertical-align:top}
.tg .tg-chko{background-color:#FFF;color:#1F2328;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-fr9f{background-color:#FFF;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-y5w1{background-color:#FFF;border-color:inherit;color:#00E;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-d5y0{background-color:#FFF;color:#1F2328;text-align:center;vertical-align:top}
.tg .tg-9y4h{background-color:#FFF;border-color:inherit;color:#1F2328;text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-fr9f"></th>
    <th class="tg-fr9f"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-y5w1">Original (darknet)</th>
    <th class="tg-y5w1">abeardear/pytorch-YOLO-v1</th>
    <th class="tg-baqh">zjykzj/YOLOv1(This)</th>
    <th class="tg-baqh">zjykzj/YOLOv1(This)</th>
    <th class="tg-baqh">zjykzj/YOLOv1(This)</th>
    <th class="tg-baqh">zjykzj/YOLOv1(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fr9f">ARCH</td>
    <td class="tg-zkss">YOLOv1</td>
    <td class="tg-zkss">FastYOLOv1</td>
    <td class="tg-zkss">ResNet_YOLOv1</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">YOLOv1(S=14)</span></td>
    <td class="tg-chko">FastYOLOv1(S=14)</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">YOLOv1</span></td>
    <td class="tg-d5y0">FastYOLOv1</td>
  </tr>
  <tr>
    <td class="tg-fr9f">VOC AP[IoU=0.50]</td>
    <td class="tg-zkss">63.4</td>
    <td class="tg-9y4h">52.7</td>
    <td class="tg-9y4h">66.5</td>
    <td class="tg-baqh">71.71</td>
    <td class="tg-baqh">60.38</td>
    <td class="tg-baqh">66.85</td>
    <td class="tg-baqh">52.89</td>
  </tr>
</tbody>
</table>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Latest News](#latest-news)
- [Background](#background)
- [Prepare Data](#prepare-data)
  - [Pascal VOC](#pascal-voc)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Container](#container)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Demo](#demo)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Latest News

* ***[2023/06/26][v0.3.2](https://github.com/zjykzj/YOLOv1/releases/tag/v0.3.2). Refactor data module.***
* ***[2023/05/16][v0.3.1](https://github.com/zjykzj/YOLOv1/releases/tag/v0.3.1). Add `IGNORE_THRESH` in YOLOv1Loss and reset `lambda_*` based on the YOLOv1 paper.***
  * **In this version, the test results on the VOC dataset have exceeded the paper implementation.**
* ***[2023/05/16][v0.3.0](https://github.com/zjykzj/YOLOv1/releases/tag/v0.3.0). Expand receptive field and use F.cross_entropy for class loss.***
* ***[2023/05/14][v0.2.0](https://github.com/zjykzj/YOLOv1/releases/tag/v0.2.0). Update VOC dataset training results for YOLOv1 and FastYOLOv1.***

## Background

YOLOv1 is the beginning of the YOLO series, which establishes the basic architecture of the YOLO target detection network. In this repository, I plan to reimplement YOLOv1 to help better understand the YOLO architecture

## Prepare Data

### Pascal VOC

Use this script [voc2yolov5.py](https://github.com/zjykzj/vocdev/blob/master/py/voc2yolov5.py)

```shell
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-train -l trainval-2007 trainval-2012
python voc2yolov5.py -s /home/zj/data/voc -d /home/zj/data/voc/voc2yolov5-val -l test-2007
```

Then softlink the folder where the dataset is located to the specified location:

```shell
ln -s /path/to/voc /path/to/YOLOv1/../datasets/voc
```

## Installation

### Requirements

See [NVIDIA/apex](https://github.com/NVIDIA/apex)

### Container

Development environment (Use nvidia docker container)

```shell
docker run --gpus all -it --rm -v </path/to/YOLOv1>:/app/YOLOv1 -v </path/to/voc>:/app/datasets/voc nvcr.io/nvidia/pytorch:22.08-py3
```

## Usage

### Train

* One GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main_amp.py -c configs/yolov1_s14_voc.cfg --opt-level=O1 ../datasets/voc
```

* Multi-GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "36121" main_amp.py -c configs/yolov1_s14_voc.cfg --opt-level=O1 ../datasets/voc
```

### Eval

```shell
python eval.py -c configs/yolov1_s14_voc.cfg -ckpt outputs/yolov1_s14_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.7277
AP for bicycle = 0.8156
AP for bird = 0.7018
AP for boat = 0.5847
AP for bottle = 0.4280
AP for bus = 0.7849
AP for car = 0.7739
AP for cat = 0.8371
AP for chair = 0.5432
AP for cow = 0.7970
AP for diningtable = 0.7196
AP for dog = 0.8270
AP for horse = 0.8401
AP for motorbike = 0.7996
AP for person = 0.7258
AP for pottedplant = 0.4511
AP for sheep = 0.7157
AP for sofa = 0.7383
AP for train = 0.8082
AP for tvmonitor = 0.7221
Mean AP = 0.7171
python eval.py -c configs/yolov1_voc.cfg -ckpt outputs/yolov1_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.6916
AP for bicycle = 0.7539
AP for bird = 0.6359
AP for boat = 0.5363
AP for bottle = 0.3216
AP for bus = 0.7710
AP for car = 0.7297
AP for cat = 0.8380
AP for chair = 0.4568
AP for cow = 0.7125
AP for diningtable = 0.6579
AP for dog = 0.7984
AP for horse = 0.7886
AP for motorbike = 0.7398
AP for person = 0.6630
AP for pottedplant = 0.4048
AP for sheep = 0.6586
AP for sofa = 0.6916
AP for train = 0.8208
AP for tvmonitor = 0.6996
Mean AP = 0.6685
python eval.py -c configs/fastyolov1_s14_voc.cfg -ckpt outputs/fastyolov1_s14_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.6090
AP for bicycle = 0.7262
AP for bird = 0.5349
AP for boat = 0.4699
AP for bottle = 0.2417
AP for bus = 0.7292
AP for car = 0.7069
AP for cat = 0.7192
AP for chair = 0.3803
AP for cow = 0.6386
AP for diningtable = 0.6300
AP for dog = 0.7174
AP for horse = 0.7696
AP for motorbike = 0.7248
AP for person = 0.6621
AP for pottedplant = 0.3198
AP for sheep = 0.6093
AP for sofa = 0.5662
AP for train = 0.7128
AP for tvmonitor = 0.6071
Mean AP = 0.6038
python eval.py -c configs/fastyolov1_voc.cfg -ckpt outputs/fastyolov1_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.5515
AP for bicycle = 0.6446
AP for bird = 0.4649
AP for boat = 0.3989
AP for bottle = 0.1817
AP for bus = 0.6707
AP for car = 0.6120
AP for cat = 0.6896
AP for chair = 0.2574
AP for cow = 0.5105
AP for diningtable = 0.5809
AP for dog = 0.6595
AP for horse = 0.7308
AP for motorbike = 0.6273
AP for person = 0.5519
AP for pottedplant = 0.2394
AP for sheep = 0.4869
AP for sofa = 0.5197
AP for train = 0.6974
AP for tvmonitor = 0.5022
Mean AP = 0.5289
```

### Demo

```shell
python demo.py -ct 0.2 configs/yolov1_s14_voc.cfg outputs/yolov1_s14_voc/model_best.pth.tar --exp voc assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [zjykzj/YOLOv2](https://github.com/zjykzj/YOLOv2)
* [zjykzj/anchor-boxes](https://github.com/zjykzj/anchor-boxes)
* [zjykzj/vocdev](https://github.com/zjykzj/vocdev)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/YOLOv1/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj