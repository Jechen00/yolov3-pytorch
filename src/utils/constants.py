#####################################
# Imports
#####################################
import torch


#####################################
# General Constants
#####################################
# Default device to train on, based on what is available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

BOLD_START = '\033[1m'
BOLD_END = '\033[0m'


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
    'map_large': 'mAP (L)',
    'map_medium': 'mAP (M)',
    'map_small': 'mAP (S)',
    'map_50': 'mAP@[IoU=0.50]',
    'map_75': 'mAP@[IoU=0.75]'
}