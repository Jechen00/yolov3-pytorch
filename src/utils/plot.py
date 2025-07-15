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
    rm_lb_pad: bool = False,
    show_probs: bool = True,
    ax: Optional[Axes] = None,
    **kwargs
) -> Optional[Figure]:
    '''
    Constructs a plot overlaying a PIL image with predicted bounding boxes and class labels 
    from a YOLOv3 model.

    Args:
        model (nn.Module): The YOLOv3 model to predict with. Will be set to `.eval()` mode if not done so already.
        img (Image.Image): The PIL image to predict bounding boxes and class labels for.
        img_transforms (Callable): The transformation function to preprocess `img` before using `model` to predict on it.
                                   If this includes a letterbox transform, consider using `rm_lb_pad = True` to remove the padding.
        scale_anchors (List[torch.tensor]): List of anchor tensors for each scale of `model`.
                                            Each element has shape: (num_anchors, 2), where the last dimension gives 
                                            the (width, height) of the anchor in units of the input size (pixels).
        strides (List[Tuple[int, int]]): List of strides (height, width) corresponding to each scale of `model`.
        class_names (List[str]): List of class names for the label predictions.
        class_clrs (List[Union[str, Tuple[int, int, int]]]): List of colors  for each class in `class_names`.
                                                             These should be in the form of strings (e.g. HEX) or tuples (e.g. RGB).
        obj_threshold (float): The probability treshold to filter out low predicted object confidence. Default is 0.2.
        nms_threshold (float): The IoU threshold used when performing non-maximum suppression. Default is 0.5.
        softmax_probs (bool): Whether to use softmax on class scores instead of sigmoid.
        rm_lb_pad (bool): Whether to remove the padding from a letterbox transform on the image before plotting.
                          This should only be used if a letterbox transform was part of the pipeline in `img_transforms`.
                          Default is False.
        show_probs (bool): Whether class probabilities should be plotted along 
                           with the labels and bounding boxes. Default is True.
        ax (optional, Axes): A matplotlib axis to plot the image and predictions.
                             If not provided, a new figure is created and returned.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.figure`.
                  This is only used if `ax` is not provided.

    Returns:
        fig (optional, Figure): A matplotlib figure with the predictions overlaid on the image.
                                This is only returned if `ax` is not provided.
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
        
        if show_probs:
            txt_str = f'{name.capitalize()}: {pred_res["scores"][i]:.2f}'
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
    '''
    Plots the loss-related results from training a YOLOv3 model (the base model).
    This includes the total YOLOv3 loss and its unweighted 
    coodinate, classification, and object confidence components.
    If provided, the EMA model's validation loss values are plotted alongside the base model's.

    Args:
        base_train_losses (optional, Dict[str, list]): Dictionary mapping loss component names to their
            corresponding lists of training loss values from the base model.
            The keys should be as follows:
                - total: The total loss, summing together all weighted components of the YOLOv3 loss.
                - coord: The unweighted coordinate/localization loss from object cells (target P(object) > 0).
                - class: The unweighted classification loss from object cells (target P(object) > 0).
                - conf: The unweighted object confidence loss from all valid cells (target P(object) != -1).
            If not provided, training loss curves are not plotted. Default is None.
        base_val_losses (optional, Dict[str, list]): Same as `base_train_losses`, but for the validation loss values from the base model.
                                                     Default is None.
        ema_val_losses (optional, Dict[str, list]): Same as `base_val_losses`, but for the validation loss values from the EMA model.
                                                    Default is None.

    Returns:
        fig (Figure): A matplotlib figure containing a 2x2 grid of subplots, 
                      one for each key in the loss dictionaries (total, coord, class, conf).
                      The number of curves inside each subplot equals the number of loss dictionaries provided.
    '''
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

def plot_eval_results(base_eval_history: Optional[Dict[int, Dict[str, Any]]] = None,
                      ema_eval_history: Optional[Dict[int, Dict[str, Any]]] = None,
                      eval_keys: Optional[List[str]] = None) -> Figure:
    '''
    Plots the evaluation metrics from training a YOLOv3 model (the base model).
    If provided, the EMA model's evaluation metrics are plotted alongside the base model's.

    Args:
        base_eval_history (optional, Dict[int, Dict[str, Any]]): Dictionary mapping evaluation epoch indices (int)
                                                to dictionaries containing validation metrics (float) for the base model.
                                                Each dictionary must include the keys in `eval_keys`.
                                                Example for `eval_keys = ['map']`: 
                                                    {
                                                        5: {'map': 0.45, ...},
                                                        10: {'map': 0.5, ...}
                                                    }
                                                If not provided, evaluation metric curves for the base model are not plotted.
                                                Default is None.
        ema_eval_history (optional, Dict[int, Dict[str, Any]]): Same as `base_eval_history` but for the EMA model.
                                                                Default is None.
        eval_keys (optional, List[str]): List of evaluation metric key to extract and plot from 
                                         the dictionaries in `base_eval_history` and `ema_eval_history`.
                                         If not provided, defaults to ['map'] for mean Average Precision.
                                         Default is None.

    Returns:
        fig (Figure): A matplotlib figure plotting all curves from `eval_keys`, one per provided eval_history dictionary.
                      The total number of curves is len(eval_keys) * (number of eval_history dictionaries).
    '''
    eval_keys = ['map'] if eval_keys is None else eval_keys
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
        plot_lines = []
        if eval_log is not None:
            num_plot_logs += 1
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