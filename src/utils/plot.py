#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from PIL import Image
from typing import Union, List, Tuple, Callable, Optional, Dict, Any
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import patches

from src import evaluate
from src.data_setup import transforms
from src.utils import constants


#####################################
# Functions
#####################################
def draw_preds_yolov3(
    model: nn.Module,
    img: Image.Image,
    img_transforms: Callable,
    scale_anchors: List[torch.Tensor],
    strides: List[Tuple[int, int]],
    class_names = List[str],
    class_clrs = List[Union[str, Tuple[int, int, int]]],
    obj_threshold: float = 0.2,
    nms_threshold: float = 0.5,
    softmax_probs: bool = False,
    rm_lb_pad: bool = True,
    show_scores: bool = True,
    **kwargs
) -> Figure:
    '''
    kwargs is for `matplotlib.pyplot.figure`
    '''
    assert len(class_names) == len(class_clrs), '`class_names` and `class_clrs` must be the same length.'
    
    device = next(model.parameters()).device
    
    # --------------
    # Predictions
    # --------------
    processed_img = img_transforms(img)
    pred_dicts = evaluate.predict_yolov3(
        model = model, 
        X = processed_img.unsqueeze(0).to(device),
        scale_anchors = scale_anchors,
        strides = strides,
        obj_threshold = obj_threshold,
        nms_threshold = nms_threshold,
        softmax_probs = softmax_probs,
    )

    pred_res = pred_dicts[0]
    bboxes, labels = pred_res['boxes'], pred_res['labels']
    
    if rm_lb_pad:
        processed_img, bboxes = transforms.remove_letterbox_pad(
            processed_img, bboxes, (img.size[1], img.size[0])
        )

    # --------------
    # Plotting
    # --------------
    fig = plt.figure(**kwargs)

    plt.imshow(processed_img.cpu().permute(1, 2, 0).numpy())
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        label = label.item()
        xmin, ymin, xmax, ymax = bbox.cpu()

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        bbox_lw = 2
        txt_x = xmin + bbox_lw
        txt_y = ymin + bbox_lw

        # Plot bbox
        name, clr = class_names[label], class_clrs[label]
        rect = patches.Rectangle((xmin, ymin), bbox_w, bbox_h, 
                                 linewidth = bbox_lw, edgecolor = clr, 
                                 facecolor = 'none')
        plt.gca().add_patch(rect)
        
        if show_scores:
            txt_str = f'{name.capitalize()}; {pred_res['scores'][i]:.2f}'
        else:
            txt_str = name.capitalize()

        plt.text(txt_x, txt_y, txt_str,
                 ha = 'left', va = 'top',
                 fontsize = 12, color = 'k', weight = 'bold',
                 bbox = dict(facecolor = clr, alpha = 0.7, pad = 2, edgecolor = 'none'))

    plt.axis(False)
    plt.close()
    
    return fig

def plot_loss_results(train_losses: Dict[str, list], 
                      val_losses: Optional[Dict[str, list]] = None) -> Figure:

    loss_keys = ['total', 'coord', 'class', 'conf']
    assert set(loss_keys) <= set(train_losses.keys()), (
        '`train_losses` must have the keys: total, coord, class, conf.'
    )

    if val_losses is not None:
            assert set(loss_keys) <= set(val_losses.keys()), (
        '`val_losses` must have the same keys: total, coord, class, conf.'
    )

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10), sharex = True)
    flat_axes = axes.flatten()

    for ax, key in zip(flat_axes, loss_keys):
        loss_name = constants.LOSS_NAMES[key]
        train_epochs = range(len(train_losses[key]))
        ax.plot(train_epochs, train_losses[key], label = f'Train: {loss_name}') # Train curve

        if val_losses is not None:
            val_epochs = range(len(val_losses['total']))
            ax.plot(val_epochs, val_losses[key], label = f'Val: {loss_name}') # Validation curve
        ax.grid(alpha = 0.5)
        ax.legend(fontsize = 15)
    
    for i in range(2):
        axes[0, i].tick_params(bottom = False)
        axes[1, i].set_xlabel('Epochs')
        
        axes[i, 1].yaxis.set_label_position('right')
        axes[i, 1].set_ylabel('Loss', rotation = 270, labelpad = 20)
        axes[i, 1].yaxis.tick_right()
        axes[i, 0].set_ylabel('Loss', labelpad = 5)

    plt.subplots_adjust(hspace = 0.04, wspace = 0.04)

    plt.close(fig)
    return fig

def plot_eval_results(eval_history: Dict[int, Dict[str, Any]],
                      eval_keys: List[str]) -> Figure:
    
    # This assumes each evaluation dictionary has the same keys
    assert set(eval_keys) <= set(eval_history[0].keys()), (
        'All keys in `eval_keys` must be present in each dictionary of `eval_history`'
    )

    fig = plt.figure(figsize = (8, 8))
    eval_epochs = list(eval_history.keys())

    for key in eval_keys:
        size_name = constants.EVAL_NAMES.get(key, key)
        size_values = [eval_history[epoch][key] for epoch in eval_epochs]
        plt.plot(eval_epochs, size_values, label = size_name)

    plt.legend(fontsize = 20)
    plt.grid(alpha = 0.5)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel('Evaluation Metric', fontsize = 20)

    plt.close(fig)
    return fig