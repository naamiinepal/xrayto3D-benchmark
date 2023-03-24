"""utils to visualize numpy volume"""
import math

import numpy as np
from matplotlib import pyplot as plt

from .np_utils import get_projectionslices_from_3d


def display_projection_slices_from_3d(image: np.ndarray):
    """create matplotlib figure showing projection of 3D volume 
    along the three orthogonal axis"""
    image_list = get_projectionslices_from_3d(image)
    fig, axes = create_figure(*image_list)
    for ax, npa in zip(axes, image_list):
        ax.imshow(npa, cmap="gray")
        ax.set_axis_off()
    return fig, axes


def create_figure(*planes, dpi=96, num_rows=1):
    # https://github.com/anjany/verse/blob/main/utils/data_utilities.py
    """creates a matplotlib figure
    Args:
        planes: numpy arrays to include in the figure
        dpi (int, optional): desired dpi. Defaults to 96.
        num_rows (int, optional):  by default, a single row of subplot. Defaults to 1.
    Returns:
        _type_: _description_
    """
    fig_h = round(2 * planes[0].shape[0] / dpi, 2)
    plane_w = [p.shape[1] for p in planes]
    w = sum(plane_w)
    fig_w = round(2 * w / dpi, 2)
    x_pos = [0]
    for x in plane_w[:-1]:
        x_pos.append(x_pos[-1] + x)
    fig, axs = plt.subplots(
        num_rows, int(math.ceil(len(planes) / num_rows)), figsize=(fig_w, fig_h)
    )
    for a in axs:  # type: ignore
        a.axis("off")
        idx = axs.tolist().index(a)
        a.set_position([x_pos[idx] / w, 0, plane_w[idx] / w, 1])
    return fig, axs
