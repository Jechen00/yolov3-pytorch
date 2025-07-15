# A PyTorch Implementation of YOLOv3
This project uses the PyTorch framework to implement the YOLOv3 architecture, along with a training pipeline that incorporates various techniques (e.g. multiscale training, mosaic augmentations) to improve generalization.

## Project Overview
<p align = 'center'>
  <img src = 'images/yolov3_sheep_demo.png' alt = 'YOLOv3 Sheep Demo' width = '46%'/>
  <img src = 'images/yolov3_duck_demo.png' alt = 'YOLOv3 Duck Demo' width = '52%'/>
</p>

The third iteration of the _You Only Look Once_ model, YOLOv3, was proposed by [Redmon & Farhadi (2018)](#references) and presents several architectural improvements over its predecessors: YOLOv2 [(Redmon & Farhadi 2016)](#references) and YOLOv1 [(Redmon et al. 2016)](#references). It introduces a more powerful backbone called _DarkNet-53_, which is comprised of deeper layers with batch normalization and interwoven residual connections for better feature extraction. Moreover, YOLOv3's detector now includes a _feature pyramid network (FPN)_, enabling predictions at three different scales — large, medium, and small — to improve detection across a range of object sizes.

For _object localization,_ YOLOv3 follows YOLOv2 in adopting an anchor-based strategy that constructs bounding boxes by applying predicted offsets to a fixed set of anchor boxes (these act as priors, typically determined using k-means clustering). These predictions are treated as a standard regression task that is optimized using _mean squared error (MSE)_. For _class prediction_, YOLOv3 treats each class as an independent logistic regression problem as opposed to using a single softmax over all classes. This allows the model to handle multi-label situations where an object may belong to multiple classes simultaneously. In a similar manner, the _objectness confidence_ is now handled as a logistic regression problem, with the prediction representing the probability that an object exists in a given anchor box (`Pr(object)`) instead of the product of objectness and localization quality (`Pr(object) * IoU`) seen in earlier versions of YOLO [(Redmon & Farhadi 2018)](#references).

The architecture used in this project is a faithful reproduction of YOLOv3 by [Redmon & Farhadi (2018)](#references), built using the official configuration files from the [DarkNet GitHub repository](https://github.com/pjreddie/darknet) and pretrained weights from [pjreddie.com](https://pjreddie.com/). This configuration-based approach was inspired by a previous implementation of the YOLOv3 detector done by [Kathuria 2018](#references). A key distinction lies in the training pipeline, which incorporates several optional techniques used by [Zhang et al. 2019](#references) and [Ge et al. 2021](#references) to improve generalization and accuracy. These include:
-  IoU-based coordinate loss (IoU, GIoU, DIoU, and CIoU)
-  Focal loss for object confidence
-  Softmax for class prediction in single-label scenarios
-  Label smoothing for classification
-  Cosine annealing learning rate scheduler
-  Multi-scale training (also used in the original paper)
-  Multi-image augmentations (mosaic and mix-up)
-  Exponential moving average (EMA) of model weights

## Recommended Installation Instructions
### 1) Create a New Python Environment
This environment should use **Python >= 3.10**.
### 2) Clone the `yolov3-pytorch` Repository
```
git clone git@github.com:Jechen00/yolov3-pytorch.git
```
### 3) Install Required Packages
Navigate to the `yolov3-pytorch` directory and run:
```
pip install -r requirements.txt
```
Alternatively, you may install the packages manually:
```
pip install matplotlib==3.10.3
pip install numpy==2.2.6
pip install pillow==11.2.1
pip install pyyaml==6.0.2
pip install seaborn==0.13.2
pip install torch==2.7.0
pip install torchvision==0.22.0
pip install torchmetrics==1.7.2
```

## Training Instructions
### 1) Modify Configurations
Edit the `config.yaml` file or create another YAML file following its structure.
This allows for configuring _most_ settings, such as:
- Device (CPU, MPS, or CUDA)
- Model Architecture
- EMA model tracking
- Dataloader (Pascal VOC or COCO)
- Loss Function 
- Optimizer (SGD)
- Learning Rate Scheduler (Cosine Annealing)
- Training/Evaluation Settings
- Checkpoint Settings

By default, the configuration assumes a standard setup that includes a DarkNet-53 backbone, an SGD optimizer, and a cosine annealing learning rate scheduler. If you wish to customize any of these components, you'll need to modify the `run_training.py` script accordingly.

### 2) Run Training Script
To start training, run:
```
python run_training.py -cf config.yaml
```
If you've created a custom configuration file, replace `config.yaml` with the appropriate file name.


## References
WIP
