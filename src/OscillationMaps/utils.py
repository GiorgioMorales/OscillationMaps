import torch
import pynvml
import itertools
import numpy as np
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib.pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent


def plot_osc_maps(input_image, title=None):
    if not isinstance(input_image, np.ndarray):
        input_image = input_image.detach().cpu().numpy()
    titles = [
        [r"$\nu_{e \to e}$", r"$\nu_{e \to \mu}$", r"$\nu_{e \to \tau}$"],
        [r"$\nu_{\mu \to e}$", r"$\nu_{\mu \to \mu}$", r"$\nu_{\mu \to \tau}$"],
        [r"$\nu_{\tau \to e}$", r"$\nu_{\tau \to \mu}$", r"$\nu_{\tau \to \tau}$"]
    ]
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(input_image[:, :, i, j], cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_title(titles[i][j], fontsize=14)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_available_gpus():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def get_least_used_gpus(n=4):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_free_mem = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_free_mem.append((i, mem_info.free))
    # Sort GPUs by most free memory
    sorted_gpus = sorted(gpu_free_mem, key=lambda x: x[1], reverse=True)
    selected_gpu_ids = [gpu_id for gpu_id, _ in sorted_gpus]
    # Repeat GPUs if fewer than n are available
    repeated_gpu_ids = list(itertools.islice(itertools.cycle(selected_gpu_ids), n))
    pynvml.nvmlShutdown()
    return [torch.device(f"cuda:{gpu_id}") for gpu_id in repeated_gpu_ids]


def sinkhorn_normalization(R, num_iters=3, eps=1e-15):
    # R = np.exp(R)
    for _ in range(num_iters):
        # Normalize columns
        R /= R.sum(axis=-2, keepdims=True)  # + eps
        # Normalize rows
        R /= R.sum(axis=-1, keepdims=True)  # + eps
    return R
