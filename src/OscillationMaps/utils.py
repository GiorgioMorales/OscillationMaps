import numpy as np
from pathlib import Path
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
            im = ax.imshow(input_image[:, :, i, j], cmap="viridis", aspect="auto")
            ax.set_title(titles[i][j], fontsize=14)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()