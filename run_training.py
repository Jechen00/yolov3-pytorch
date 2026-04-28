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
def load_config():
    # Set configuration file as a hyperparameter
    parser = argparse.ArgumentParser(description = 'Train YOLOv3 model')
    parser.add_argument('-cf', '--config-file', 
                        help = 'Path to the configuration YAML file.',
                        type = str, 
                        default = 'config.yaml')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f'Config file not found: {args.config_file}')

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


#####################################
# Training Code
#####################################
if __name__ == '__main__':
    misc.set_seed(0) # Set seed for reproducibility
    config = load_config()
    device = torch.device(config['device']) if config['device'] is not None else constants.DEVICE
    
    
    # ---------------------------
    # Base and EMA Model
    # ---------------------------
    base_model_cfg = config['base_model']
    ema_cfg = config['ema']

    # Using DarkNet53 backbone, as per YOLOv3 paper
    darknet53_backbone = builder.DarkNet53Backbone(cfg_file = base_model_cfg['backbone_cfg'])
    if base_model_cfg['backbone_weights'] is not None:
        darknet53_backbone.load_weights_file(weights_file = base_model_cfg['backbone_weights'], 
                                             input_shape = tuple(base_model_cfg['input_shape']))

    base_model = builder.YOLOv3(backbone = darknet53_backbone, 
                                neck_heads_cfg = base_model_cfg['neck_heads_cfg'])

    if ema_cfg['use_ema']:
        ema = ema_model.EMA(base_model = base_model, 
                            decay = ema_cfg['decay'], 
                            input_shape = tuple(base_model_cfg['input_shape']))
    else:
        ema = None

    if device.type == 'cuda':
        base_model.compile(dynamic = True)

        if ema_cfg['use_ema']:
            ema.compile(dynamic = True)

    base_model = base_model.to(device)
    if ema_cfg['use_ema']:
        ema.to(device)

    scale_anchors, strides, _ = base_model.infer_scale_info(base_model_cfg['input_shape'])


    # -------------
    # Dataloader
    # -------------
    builders = dataloader_utils.get_dataloaders(
        scale_anchors = scale_anchors,
        strides = strides,
        default_input_size = base_model_cfg['input_shape'][-1],
        return_builders = True,
        device = device,
        splits = ['train', 'val'],
        **config['dataloader']
    )


    # ---------------------------
    # Loss, Optimizer, Scheduler
    # ---------------------------
    loss_fn = loss.YOLOv3Loss(
        scale_anchors = scale_anchors,
        strides = strides,
        **config['loss_fn']
    )

    optimizer = optim.SGD(
        base_model.parameters(),
        **config['optimizer']
    )

    # Change scheduler timing arguments depending on the frequency of steps
    scheduler_timing_args = config['scheduler']['timing_args']
    if config['train_eval']['scheduler_freq'] == 'optim_step':
        effective_batch_size = config['dataloader']['batch_size'] * config['train_eval']['accum_steps']
        num_optim_steps = math.ceil(len(builders['train'].dataset) / effective_batch_size)

        for key, value in scheduler_timing_args.items():
            scheduler_timing_args[key] = value * num_optim_steps

    scheduler = schedulers.WarmupCosineAnnealingLR(
        optimizer,
        **config['scheduler']['static_args'],
        **scheduler_timing_args
    )
    
    # ---------------------------
    # Data Class Configs
    # ---------------------------
    # Training/Evaluation Configs
    te_cfg = engine.TrainEvalConfig(
        scale_anchors = scale_anchors,
        strides = strides,
        **config['train_eval']
    )

    # Checkpoint Configs
    ckpt_cfg = engine.CheckpointConfig(
        **config['checkpoint']
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
        ema = ema,
        te_cfg = te_cfg,
        ckpt_cfg = ckpt_cfg,
        device = device
    )
    