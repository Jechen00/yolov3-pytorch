#####################################
# Imports
#####################################
import torch


#####################################
# General Constants
#####################################
# Setup device and multiprocessing context
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    MP_CONTEXT = None
    PIN_MEM = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    MP_CONTEXT = 'forkserver'
    PIN_MEM = False
else:
    DEVICE = torch.device('cpu')
    MP_CONTEXT = None
    PIN_MEM = False

BOLD_START = '\033[1m'
BOLD_END = '\033[0m'


#####################################
# YOLOv3 Constants
#####################################
# This is the default resolution (H, W) that the anchors of config files are relative to
BASE_YOLO_SIZE = (416, 416)


#####################################
# Loss and Evaluation Constants
#####################################
LOSS_NAMES = {
    'total': 'Total',
    'class': 'Class',
    'coord': 'Coord',
    'conf': 'Conf',
    'obj_conf': 'Obj',
    'noobj_conf': 'NoObj'
}
EVAL_NAMES = {
    'map': 'mAP',
    'map_50': 'mAP@[IoU=0.50]',
    'map_75': 'mAP@[IoU=0.75]'
}