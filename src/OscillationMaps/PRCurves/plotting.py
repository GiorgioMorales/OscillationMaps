import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager
import matplotlib.colors 
import matplotlib.colors as mcolors

SCRIPT_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_SRC_DIR + "/..")

from src.exp_utils import ALL_METHODS_AND_GT
from src.utils import get_angles


METHODS_COLORS = {"ground-truth":"tab:purple",
                    "ipr":"tab:blue",
                    "knn":"tab:orange",
                    "parzen":"tab:green",
                    "coverage":"tab:red",
                    "ppr":"tab:purple"}

def get_method_color(method):
    if not method in METHODS_COLORS.keys():
        raise Exception(f"Method should be in {METHODS_COLORS.keys()} but got {method}")
    else:
        return METHODS_COLORS[method]

def plot_pr_complete(
    precision_recall_pairs,
    labels=None,
    out_path=None,
    legend_loc="best",
    dpi=300,
    title=None,
    display=False,
    scatter_list=None,
    extra_kwargs=None,
    extra_kwargs_scatter=None,
    methods=None
):
    """Plots precision recall curves for distributions.

    Creates the PRD plot for the given data and stores the plot in a given path.

    Args:
      precision_recall_pairs: List of prd_data to plot. Each item in this list is
                              a 2D array of precision and recall values for the
                              same number of ratios. (use get_alphas_betas_list_from_arrays function)
      labels: Optional list of labels of same length as list_of_prd_data. The
              default value is None.
      out_path: Output path for the resulting plot. If None, the plot will be
                opened via plt.show(). The default value is None.
      legend_loc: Location of the legend. The default value is 'lower left'.
      dpi: Dots per inch (DPI) for the figure. The default value is 150.
      title: Description title for the figure
      display: bool to show the figure inline with plt.show()

    Raises:
      ValueError: If labels is a list of different length than list_of_prd_data.
    """
    plt.close()
    
    sanity_check_args_plot(precision_recall_pairs, labels, scatter_list, extra_kwargs, extra_kwargs_scatter, methods)
    
    fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
    plot_handle = fig.add_subplot(111)
    plot_handle.tick_params(axis="both", which="major", labelsize=12)

    default_plot_args = dict(alpha=0.8, linewidth=3)
    colors = []
    for i in range(len(precision_recall_pairs)):
        precision, recall = precision_recall_pairs[i]
        label = labels[i] if labels is not None else None
        kwargs = extra_kwargs[i] if extra_kwargs is not None else {}
        if methods:
            if kwargs.get("color") is None:
                kwargs["color"]=get_method_color(methods[i])
            if methods[i] == "ground-truth":
                kwargs["linestyle"]="--"
        params = {**default_plot_args, **kwargs}
        ax = plt.plot(recall, precision, label=label, **params)
        colors.append(ax[0].get_color())
    if scatter_list is not None:
        for k in range(len(scatter_list)):
            precision, recall = scatter_list[k]
            kwargs = extra_kwargs_scatter[k] if extra_kwargs_scatter is not None else {}
            plt.scatter(recall, precision, **kwargs,c=colors[k])
    if labels is not None:
        plt.legend(loc=legend_loc)

    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlabel(r"Recall ($\beta$)", fontsize=12)
    plt.ylabel(r"Precision ($\alpha$)", fontsize=12)
    plt.tight_layout()

    if title:
        plt.title(title)
    if out_path is None or display:
        plt.show()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()

def sanity_check_args_plot(precision_recall_pairs, labels, scatter_list, extra_kwargs, extra_kwargs_scatter, methods):
    if labels is not None and len(labels) != len(precision_recall_pairs):
        raise ValueError(
            "Length of labels %d must be identical to length of "
            "precision_recall_pairs %d." % (len(labels), len(precision_recall_pairs))
        )

    if extra_kwargs is not None and len(extra_kwargs) != len(precision_recall_pairs):
        raise ValueError(
            "Length of extra_kwargs %d must be identical to length of "
            "precision_recall_pairs %d."
            % (len(extra_kwargs), len(precision_recall_pairs))
        )
    if scatter_list is not None:
        if len(scatter_list) != len(precision_recall_pairs):
            raise ValueError(
                f"There should be as much elements in scatter_list as values of curves to plot."
                f"But got sizes {len(scatter_list)} and {len(precision_recall_pairs)}"
            )
    if extra_kwargs_scatter is not None and len(extra_kwargs_scatter) != len(scatter_list):
        raise ValueError(
            f"Length of extra_kwargs_scatter {len(extra_kwargs)} must be identical to length of "
            "scatter_list =  {len(precision_recall_pairs)}."
        )
    if methods is not None and len(methods)!=len(precision_recall_pairs):
        raise ValueError(
           f"Lenght of methods and precision_recall_pairs should be the same but got {len(methods)} and {len(precision_recall_pairs)}"
        )

def use_latex(use_serif=True, fonts=["Times New Roman"], size=12):
    """
    Use latex as backend to plot using matplotlib
    Change the size and the font of the font
    """

    matplotlib.use("pgf")  # Use the pgf backend for LaTeX rendering
    dict_update = {
        "text.usetex": True,
        "font.family": "serif" if use_serif else "sans-serif",
        "font.size": size,
    }
    if use_serif:
        dict_update["font.serif"] = fonts
    else:
        dict_update["font.sans-serif"] = fonts
    plt.rcParams.update(dict_update)


def change_backend(backend="TkAgg"):
    """
    Revert back to the standard matplotlib
    """
    matplotlib.use(backend)
    plt.rcParams.update({"text.usetex": False})


def get_alphas_betas_list_from_arrays(alphas_betas_list):
    """The current function requires the alphas and betas to be inside a list like so:
    [(alphas1,betas1),(alphas2,betas2),(alphas3,betas3),...]
    This function converts [alphas_betas1,alphas_betas2,alphas_betas3] to the above format
    """
    output_list = []
    if isinstance(alphas_betas_list,np.ndarray):
        alphas_betas_list = [alphas_betas_list]
    for a_b_array in alphas_betas_list:
        output_list.append((a_b_array[:,0],a_b_array[:,1]))
    return output_list
 
    
######################### CODE SPECIFIC TO SHIFTING/SPREADING EXPERIMENT ######################### 
"""
1. Loading arrays of the n runs of the experiments for each metric --> computing the average of $(\alpha_\lambda,\beta_\lambda)$ 
    for each value of $\lambda$
2. Plot the average of the metrics for 2 values of $\lambda$ (in order not to over crowd the plot)
3. Get the mean and average value of IOU for the methods along the n realisations of the experiment
"""

from src.plotting import get_alphas_betas_list_from_arrays
from src.exp_utils import ALL_METHODS_AND_GT


def alterate_color(color,ratio):
    """transform a color to make it more saturated"""
    if isinstance(color, str):
        color = mcolors.to_rgb(color)
    hsv = matplotlib.colors.rgb_to_hsv(color[:3]) 
    hsv[1]=ratio/2+0.5
    return matplotlib.colors.hsv_to_rgb(hsv)

def plot_shifting_exp_from_path(exp_folder, title, setting, save_path=None, show_legend=True, remove_indexes=None):
    arrs_dict = extract_prds_from_path(exp_folder, setting=setting)
    dict_prd_pairs = get_prd_pairs_list(arrs_dict)
    plot_shifting_exp_from_dict(dict_prd_pairs, title, save_path=save_path, show_legend=show_legend, remove_indexes=remove_indexes)

def get_extra_kwargs_plot(method,i, precision_recall_pairs):
    extra_kwargs = [{"color":alterate_color(get_method_color(method),i/len(precision_recall_pairs)),
                        "alpha": 1,
                        "linewidth": 2,
                        "linestyle": "--"
                                }]

def plot_shifting_exp_from_dict(dict_prd_pairs, title, save_path=None, show_legend=True, remove_indexes=None, alpha_opacity=1):
    """
    Plot the results of shifting experiment on Gaussian distributions
    Input a dict of prd pairs in the expected format dict_prd_pairs = {method:{shift_value:(alphas,betas)}}
    remove_indexes = indexes of shifts to remove in order not to cramp the figures
    """
    fig = plt.figure(figsize=(7, 3.5), dpi = 300)
    plot_handle = fig.add_subplot(111)
    plot_handle.tick_params(axis="both", which="major", labelsize=12)
    plot_handle.axis("square")

    for method_index,method in enumerate(dict_prd_pairs):
        print(f"Processing method {method}")
        precision_recall_pairs = dict_prd_pairs[method]
        nb_curves = len(dict_prd_pairs[method])
        first_plotted = True
        for i in range(len(precision_recall_pairs)):
            if remove_indexes:
                if i in remove_indexes:
                    continue
            if method=="ground-truth" :
                extra_kwargs  = [
                        {
                        "color":alterate_color(get_method_color(method),i/len(precision_recall_pairs)),
                        "alpha": 1,
                        "linewidth": 2,
                        "linestyle": "--"
                                } for k in range(len(dict_prd_pairs[method]))
                            ]
            else:
                extra_kwargs=[{
                        "color":alterate_color(get_method_color(method),i/len(precision_recall_pairs)),
                        "alpha": alpha_opacity,
                        "linewidth":2
            } for k in range(len(dict_prd_pairs[method]))]

            precision, recall = dict_prd_pairs[method][i]
            
            label = method if first_plotted else None
            first_plotted = False

            ax = plt.plot(recall, precision, label=label, **extra_kwargs[i])
    if show_legend:
        plot_handle.legend(loc="upper right")
        # Put a legend to the right of the current axis
        plot_handle.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        box = plot_handle.get_position()
        plot_handle.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlabel(r"Recall ($\beta$)", fontsize=12)
    plt.ylabel(r"Precision ($\alpha$)", fontsize=12)
    plt.tight_layout()

    plt.title(title)
    if save_path:
        plt.savefig(save_path,bbox_inches="tight")


def extract_prds_from_path(folder_exp, setting):
    """
    go through a path and extract all the shifting experiments in a dictionnary\
        with keys = method and values = alphas betas for different random seeds
    We need to also extract the values of mean for the shifting exp and of std for spreading exp
    """
    cpt=0
    arrs_dict = {}
    
    for root_path,folders,files in os.walk(folder_exp):
        for file in files:
            method = file.split("_")[3]
            if file.endswith(".npy"):
                file=file.replace(".npy","")
                cpt+=1
                arr = np.load(os.path.join(root_path, file+".npy"))
                if setting=="shift":
                    # select mean
                    var = file.split("_")[-3]
                elif setting=="spreading":
                    # select std
                    var = file.split("_")[-1]
                else:
                    raise NotImplementedError(f"Setting {setting} is not implemented")
                # initialising the method dict
                if arrs_dict.get(method) is None:
                    arrs_dict[method] = {var:[arr]}
                
                # init the var sub dict
                else:
                    if arrs_dict[method].get(var) is None:
                        arrs_dict[method][var]=[arr]
                    else:
                        # in the case where running experiment with varying random seeds
                        arrs_dict[method][var].append(arr)    
    print(f"Found and extracted {cpt} prd curves in {folder_exp}")
    return arrs_dict

def get_average_alphas_betas(dict_exps,method):
    """
    When running experiment with multiple random seeds, we want to plot the average of all the values
    """
    return [np.mean(np.stack(dict_exps[method][shift],axis=0),axis=0) for shift in dict_exps[method]]

def get_prd_pairs_list(arrs_dict):
    """
    Convert dict to format which can then be plotted easily
    """
    dict_average_alphas_betas = {}

    for method in ALL_METHODS_AND_GT:
        if method in arrs_dict.keys():
            dict_average_alphas_betas[method] = get_average_alphas_betas(arrs_dict,method)

    # puting the lists at the necessary format to plot the PRD curves
    dict_prd_pairs = {}
    for method in ALL_METHODS_AND_GT:
        if method in arrs_dict.keys():
            dict_prd_pairs[method] = get_alphas_betas_list_from_arrays(dict_average_alphas_betas[method])
    return dict_prd_pairs

