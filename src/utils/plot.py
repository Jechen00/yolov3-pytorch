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
from matplotlib.axes import Axes

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
    ax: Optional[Axes] = None,
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
    if ax is None:
        if not kwargs:
            kwargs = {'figsize': (10, 10)}
        fig = plt.figure(**kwargs)
        ax = plt.gca()
    else:
        fig = None

    ax.imshow(processed_img.cpu().permute(1, 2, 0).numpy())
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
        ax.add_patch(rect)
        
        if show_scores:
            txt_str = f'{name.capitalize()}; {pred_res["scores"][i]:.2f}'
        else:
            txt_str = name.capitalize()

        ax.text(txt_x, txt_y, txt_str,
                ha = 'left', va = 'top',
                fontsize = 12, color = 'k', weight = 'bold',
                bbox = dict(facecolor = clr, alpha = 0.7, pad = 2, edgecolor = 'none'))

    ax.axis(False)
    if fig is not None:
        plt.close(fig)
        return fig

def plot_loss_results(base_train_losses: Optional[Dict[str, list]] = None, 
                      base_val_losses: Optional[Dict[str, list]] = None,
                      ema_val_losses: Optional[Dict[str, list]] = None) -> Figure:

    loss_keys = ['total', 'coord', 'class', 'conf']
    loss_sets = {
        'base_train_losses': (base_train_losses, 'Train (Base)'),
        'base_val_losses': (base_val_losses, 'Val (Base)'),
        'ema_val_losses': (ema_val_losses, 'Val (EMA)')
    }

    for set_key, loss_set in loss_sets.items():
        if loss_set[0] is not None:
            assert set(loss_keys) <= set(loss_set[0].keys()), (
                f'`{set_key}` must have the keys: total, coord, class, conf.'
            )

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10), sharex = True)
    flat_axes = axes.flatten()

    for ax, loss_key in zip(flat_axes, loss_keys):
        loss_name = constants.LOSS_NAMES[loss_key]

        for loss_set in loss_sets.values():
            loss_log = loss_set[0]
            if loss_log is not None:
                epochs = range(len(loss_log[loss_key]))
                ax.plot(epochs, loss_log[loss_key], label = f'{loss_set[1]}: {loss_name}') # Loss curve

        ax.grid(alpha = 0.5)
        ax.legend(fontsize = 14)
    
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

def plot_eval_results(base_eval_history: Dict[int, Dict[str, Any]],
                      eval_keys: List[str],
                      ema_eval_history: Optional[Dict[int, Dict[str, Any]]] = None) -> Figure:
    
    eval_sets = {
        'base_eval_history': (base_eval_history, 'Base'),
        'ema_eval_history': (ema_eval_history, 'EMA')
    }

    for set_key, eval_set in eval_sets.items():
        eval_log = eval_set[0]
        if eval_log is not None:
            first_eval_epoch = list(eval_log.keys())[0]
            # This assumes each evaluation dictionary has the same keys
            assert set(eval_keys) <= set(eval_log[first_eval_epoch].keys()), (
                f'All keys in `eval_keys` must be present in each dictionary of `{set_key}`'
            )

    fig = plt.figure(figsize = (8, 8))
    ax = plt.gca()

    num_plot_logs = -1
    for eval_set in eval_sets.values():
        eval_log = eval_set[0]
        if eval_log is not None:
            num_plot_logs += 1
            plot_lines = []
            eval_epochs = list(eval_log.keys())
            for key in eval_keys:
                eval_name = constants.EVAL_NAMES.get(key, key)
                eval_values = [eval_log[epoch][key] for epoch in eval_epochs]
                line, = ax.plot(eval_epochs, eval_values, label = f'{eval_set[1]}: {eval_name}')
                plot_lines.append(line)

        legend = ax.legend(handles = plot_lines, 
                           fontsize = 14,             
                           loc = 'lower right',
                           bbox_to_anchor = (1 - num_plot_logs * 0.35, 0))
        ax.add_artist(legend)

    ax.grid(alpha = 0.5)
    ax.set_xlabel('Epochs', fontsize = 20)
    ax.set_ylabel('Evaluation Metric', fontsize = 20)

    plt.close(fig)
    return fig