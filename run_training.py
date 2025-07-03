#####################################
# Imports & Dependencies
#####################################
import torch
from torch import optim

import os
import yaml
import argparse

from src.models import builder
from src.utils import constants, misc
from src.data_setup import dataloader_utils
from src import loss, schedulers, engine



#####################################
# Functions
#####################################
def load_configs():
    # Set configuration file as a hyperparameter
    parser = argparse.ArgumentParser(description = 'Train YOLOv3 model')
    parser.add_argument('-cf', '--config-file', 
                        help = 'Path to the configuration YAML file.',
                        type = str, 
                        default = 'configs.yaml')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f'Config file not found: {args.config_file}')

    with open(args.config_file, 'r') as f:
        configs = yaml.safe_load(f)

    return configs


#####################################
# Training Code
#####################################
if __name__ == '__main__':
    misc.set_seed(0) # Set seed for reproducibility
    configs = load_configs()
    device = torch.device(configs['device']) if configs['device'] is not None else constants.DEVICE
    
    
    # ---------------------------
    # Model
    # ---------------------------
    model_cfgs = configs['model']

    # Using DarkNet53 backbone, as per YOLOv3 paper
    darknet53_backbone = builder.DarkNet53Backbone(cfg_file = model_cfgs['backbone_cfgs'])
    darknet53_backbone.load_weights_file(weights_file = model_cfgs['backbone_weights'], 
                                        input_shape = tuple(model_cfgs['input_shape']))

    model = builder.YOLOv3(backbone = darknet53_backbone, detector_cfgs = model_cfgs['detector_cfgs'])

    # Device will be CUDA or MPS if they are avaliable (Change if needed)
    if device.type == 'cuda':
        model.compile(dynamic = True)
    model = model.to(device)

    scale_anchors, strides, _ = model.infer_scale_info(model_cfgs['input_shape'])


    # ---------------------------
    # Dataloader (Pascal VOC)
    # ---------------------------
    train_builder, val_builder = dataloader_utils.get_dataloaders(
        scale_anchors = scale_anchors,
        strides = strides,
        default_input_size = model_cfgs['input_shape'][-1],
        return_builders = True,
        device = device,
        **configs['dataloader']
    )


    # ---------------------------
    # Loss, Optimizer, Scheduler
    # ---------------------------
    loss_fn = loss.YOLOv3Loss(
        scale_anchors = scale_anchors,
        strides = strides,
        **configs['loss_fn']
    )

    optimizer = optim.SGD(
        model.parameters(),
        **configs['optimizer']
    )

    scheduler = schedulers.WarmupCosineAnnealingLR(
        optimizer,
        **configs['scheduler']
    )


    # ---------------------------
    # Data Class Configs
    # ---------------------------
    # Training/Evaluation Configs
    te_cfgs = engine.TrainEvalConfigs(
        scale_anchors = scale_anchors,
        strides = strides,
        **configs['train_eval']
    )

    # Checkpoint Configs
    ckpt_cfgs = engine.CheckpointConfigs(
        **configs['checkpoint']
    )


    # ---------------------------
    # Run Training
    # ---------------------------
    train_losses, val_losses, eval_history = engine.train(
        model = model,
        train_builder = train_builder,
        val_builder = val_builder,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        te_cfgs = te_cfgs,
        ckpt_cfgs = ckpt_cfgs,
        device = device
    )
    