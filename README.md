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
    <td class="tg-baqh">66.06</td>
    <td class="tg-baqh">54.81</td>
    <td class="tg-baqh">63.06</td>
    <td class="tg-baqh">49.89</td>
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
AP for aeroplane = 0.6778
AP for bicycle = 0.7818
AP for bird = 0.6459
AP for boat = 0.5422
AP for bottle = 0.3457
AP for bus = 0.7388
AP for car = 0.7438
AP for cat = 0.8009
AP for chair = 0.4762
AP for cow = 0.6693
AP for diningtable = 0.6399
AP for dog = 0.7767
AP for horse = 0.8256
AP for motorbike = 0.7572
AP for person = 0.6953
AP for pottedplant = 0.3835
AP for sheep = 0.6109
AP for sofa = 0.6699
AP for train = 0.7631
AP for tvmonitor = 0.6684
Mean AP = 0.6606
python eval.py -c configs/yolov1_voc.cfg -ckpt outputs/yolov1_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.6621
AP for bicycle = 0.7497
AP for bird = 0.6207
AP for boat = 0.5305
AP for bottle = 0.2737
AP for bus = 0.7349
AP for car = 0.6867
AP for cat = 0.8092
AP for chair = 0.4193
AP for cow = 0.6807
AP for diningtable = 0.6257
AP for dog = 0.7545
AP for horse = 0.7848
AP for motorbike = 0.7172
AP for person = 0.6324
AP for pottedplant = 0.3107
AP for sheep = 0.6070
AP for sofa = 0.6538
AP for train = 0.7643
AP for tvmonitor = 0.5951
Mean AP = 0.6306
python eval.py -c configs/fastyolov1_s14_voc.cfg -ckpt outputs/fastyolov1_s14_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.6351
AP for bicycle = 0.6876
AP for bird = 0.4754
AP for boat = 0.3916
AP for bottle = 0.2111
AP for bus = 0.6681
AP for car = 0.6473
AP for cat = 0.6638
AP for chair = 0.3328
AP for cow = 0.5368
AP for diningtable = 0.5210
AP for dog = 0.6372
AP for horse = 0.7270
AP for motorbike = 0.6959
AP for person = 0.6057
AP for pottedplant = 0.2159
AP for sheep = 0.5438
AP for sofa = 0.5415
AP for train = 0.6783
AP for tvmonitor = 0.5458
Mean AP = 0.5481
python eval.py -c configs/fastyolov1_voc.cfg -ckpt outputs/fastyolov1_voc/model_best.pth.tar ../datasets/voc
VOC07 metric? Yes
AP for aeroplane = 0.5922
AP for bicycle = 0.6198
AP for bird = 0.4091
AP for boat = 0.3628
AP for bottle = 0.1442
AP for bus = 0.6122
AP for car = 0.5751
AP for cat = 0.6963
AP for chair = 0.2481
AP for cow = 0.4858
AP for diningtable = 0.5206
AP for dog = 0.6302
AP for horse = 0.7140
AP for motorbike = 0.6067
AP for person = 0.5111
AP for pottedplant = 0.1767
AP for sheep = 0.4401
AP for sofa = 0.5081
AP for train = 0.6637
AP for tvmonitor = 0.4619
Mean AP = 0.4989
```

### Demo

```shell
python demo.py -c 0.4 configs/yolov1_s14_voc.cfg outputs/yolov1_s14_voc/model_best.pth.tar --exp voc assets/voc2007-test/
```

<p align="left"><img src="results/voc/000237.jpg" height="240"\>  <img src="results/voc/000386.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1)
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