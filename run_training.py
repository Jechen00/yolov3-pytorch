#####################################
# Imports & Dependencies
#####################################
import torch
from torch import optim

import os
import yaml
import argparse
import math 

from src.models import builder, ema_model
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
    # Base and EMA Model
    # ---------------------------
    base_model_cfgs = configs['base_model']
    ema_cfgs = configs['ema']

    # Using DarkNet53 backbone, as per YOLOv3 paper
    darknet53_backbone = builder.DarkNet53Backbone(cfg_file = base_model_cfgs['backbone_cfgs'])
    if base_model_cfgs['backbone_weights'] is not None:
        darknet53_backbone.load_weights_file(weights_file = base_model_cfgs['backbone_weights'], 
                                            input_shape = tuple(base_model_cfgs['input_shape']))

    base_model = builder.YOLOv3(backbone = darknet53_backbone, 
                                detector_cfgs = base_model_cfgs['detector_cfgs'])

    if ema_cfgs['use_ema']:
        ema = ema_model.EMA(base_model = base_model, 
                            decay = ema_cfgs['decay'], 
                            input_shape = tuple(base_model_cfgs['input_shape']))
    else:
        ema = None

    if device.type == 'cuda':
        base_model.compile(dynamic = True)

        if ema_cfgs['use_ema']:
            ema.compile(dynamic = True)

    base_model = base_model.to(device)
    if ema_cfgs['use_ema']:
        ema.to(device)

    scale_anchors, strides, _ = base_model.infer_scale_info(base_model_cfgs['input_shape'])


    # -------------
    # Dataloader
    # -------------
    builders = dataloader_utils.get_dataloaders(
        scale_anchors = scale_anchors,
        strides = strides,
        default_input_size = base_model_cfgs['input_shape'][-1],
        return_builders = True,
        device = device,
        splits = ['train', 'val'],
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
        base_model.parameters(),
        **configs['optimizer']
    )

    scheduler_freq = configs['scheduler']['freq']
    scheduler_timing_args = configs['scheduler']['timing_args']

    # Change scheduler timing arguments depending on the frequency of steps
    if scheduler_freq == 'optim_step':
        effective_batch_size = configs['dataloader']['batch_size'] * configs['train_eval']['accum_steps']
        num_optim_steps = math.ceil(len(builders['train'].dataset) / effective_batch_size)

        for key, value in scheduler_timing_args.items():
            scheduler_timing_args[key] = value * num_optim_steps

    scheduler = schedulers.WarmupCosineAnnealingLR(
        optimizer,
        **configs['scheduler']['static_args'],
        **scheduler_timing_args
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
        base_model = base_model,
        train_builder = builders['train'],
        val_builder = builders['val'],
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        scheduler_freq = scheduler_freq,
        ema = ema,
        te_cfgs = te_cfgs,
        ckpt_cfgs = ckpt_cfgs,
        device = device
    )
    