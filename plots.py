import os
import pickle
from glob import glob

import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
plt.rcParams["figure.figsize"] = (7, 3)

from utils.seed import SEED
from utils.dataset import DATASET_SIZES
from evaluate import evaluate_experiment
from utils.utils import read_sorted_results


bar_plot_params = [
    {'color':'tab:red', 'linestyle':':', 'alpha': 0.75, 'ecolor':to_rgba('black', 0.6)},
    {'color':'tab:blue', 'linestyle':'--', 'alpha': 0.75, 'ecolor':to_rgba('black', 0.6)},
    {'color':'tab:green', 'linestyle':'-.', 'alpha': 0.75, 'ecolor':to_rgba('black', 0.6)},
    {'color':'tab:orange', 'linestyle':'-.', 'alpha': 0.75, 'ecolor':to_rgba('black', 0.6)},
]
error_bar_params = {'elinewidth':0.3, 'capsize':2}
error_plot_params = [
    {'color':'tab:red', 'linestyle':':', 'marker':'o', 'ecolor':to_rgba('tab:red', 0.6)},
    {'color':'tab:blue', 'linestyle':'--', 'marker':'x', 'ecolor':to_rgba('tab:blue', 0.6)},
    {'color':'tab:green', 'linestyle':'-.', 'marker':'^', 'ecolor':to_rgba('tab:green', 0.6)},
    {'color':'tab:orange', 'linestyle':'-.', 'marker':'s', 'ecolor':to_rgba('tab:orange', 0.6)},
]
plot_params = [
    {'color':'tab:red', 'linestyle':':', 'marker':'o',},
    {'color':'tab:blue', 'linestyle':'--', 'marker':'x',},
    {'color':'tab:green', 'linestyle':'-.', 'marker':'^',},
    {'color':'tab:orange', 'linestyle':'-.', 'marker':'s',},
]
plot_params_wo_c = [
    {'linestyle':':', 'marker':'o',},
    {'linestyle':'--', 'marker':'x',},
    {'linestyle':'-.', 'marker':'^',},
    {'linestyle':'-.', 'marker':'s',},
]
line_plot_params = [
    {'color':'tab:red', },
    {'color':'tab:blue', },
    {'color':'tab:green', },
    {'color':'tab:orange', },
]


def create_visualization_dir(exp_path):
    visualization_path = os.path.join(exp_path, "vis")
    os.makedirs(visualization_path, exist_ok=True)
    return visualization_path

def intro_cone():
    logp = [2.15, 2.82, 3.23, ]
    negsas = [-2.048493643, -2.120641301, -3.712117345, ]
    names = ["Propanidid", "Diazepam", "Thiamylal", ]

    molecules = list(zip(names, list(zip(logp, negsas))))

    xlim = [1.75, 3.5]
    ylim = [-4.2, -0.5]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\log$ OPC")
    ax.set_ylabel(r"-SAS")

    # Define the slopes of the lines based on given angles
    m1 = np.tan(-np.pi/6)
    m2 = np.tan(2*np.pi/3)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])

    colors = ["mediumslateblue", "tab:red", "tab:blue", "tab:green"]
    
    for i, (m_name, (m_logp, m_negsas)) in enumerate(molecules):
        # Plot the dot
        ax.scatter(m_logp, m_negsas, c='black')

        # Display the name at the bottom left of the dot
        plt.text(
            m_logp, m_negsas, m_name,
            ha='right', va='top', fontsize=11,
            position=(m_logp-0.03, m_negsas-0.03)
        )

        # For the line moving right below
        x_right = np.linspace(m_logp, xlim[-1], 100)
        y_right = m_negsas + m1 * (x_right - m_logp)
        
        # For the line moving left above
        x_left = np.linspace(xlim[0], m_logp, 100)
        y_left = m_negsas + m2 * (x_left - m_logp)

        verts = np.array([
            [m_logp, m_negsas],
            [x_left[0], y_left[0]],
            [xlim[0], ylim[1]],
            [xlim[1], ylim[1]],
            [x_right[-1], y_right[-1]],
        ])
        polygon = Polygon(
            verts, closed=True, alpha=0.35, color=colors[0]
        )
        ax.add_patch(polygon)
    
    plt.tight_layout()
    plt.savefig("intro_fig.png")
    plt.savefig("intro_fig.pdf")

def paveba_vs_beta(exp_path, paveba_id):
    visualization_path = create_visualization_dir(exp_path)

    paveba_path = glob(os.path.join(exp_path, f"exp{paveba_id}-*"))[0]

    paveba_results_list = read_sorted_results(paveba_path)

    datasets = []
    cone_degrees = []
    epsilons = []

    paveba_sc_beta = np.zeros((len(paveba_results_list), 3))

    def beta_val(x):
        x = (x / 180) * np.pi
        return np.where(x < np.pi/2, 1/np.sin(x), 1)

    # Evaluate each config
    i = 0
    for exp_dict in paveba_results_list:
        cone_degree = exp_dict['cone_degree']
        if cone_degree not in cone_degrees:
            cone_degrees.append(cone_degree)
        if exp_dict['eps'] not in epsilons:
            epsilons.append(exp_dict['eps'])
        if exp_dict['dataset_name'] not in datasets:
            datasets.append(exp_dict['dataset_name'])

        result, result_std = evaluate_experiment(exp_dict)
        beta_sqr = beta_val(cone_degree)**2

        paveba_sc_beta[i] = [result["SC"], result_std["SC Std"], beta_sqr]
        i += 1

    sc_beta_dce3 = paveba_sc_beta.reshape(len(datasets), len(cone_degrees), len(epsilons), 3)
    sc_beta_dec3 = sc_beta_dce3.transpose(0, 2, 1, 3)

    error_scatter_params = {'color':'tab:red', 'fmt':'o', 'ecolor':to_rgba('tab:red', 0.6)}

    fig, ax = plt.subplots(
        len(datasets), len(epsilons), figsize=(5*len(epsilons), 3), squeeze=False
    )
    for (dset_id, beta_sc_ec3) in enumerate(sc_beta_dec3):
        for (eps_id, beta_sc_c3) in enumerate(beta_sc_ec3):
            tmp_ax = ax[dset_id, eps_id]

            A = np.linalg.lstsq(
                (beta_sc_c3[:, 2]).reshape(-1, 1), beta_sc_c3[:, 0].reshape(-1, 1), rcond=None
            )[0].squeeze()

            if dset_id == 0 and eps_id == 0:
                plt_label = r"S.C. regression on $\beta^2$"
                sc_label = "PaVeBa avg. sample complexities(S.C.)"
            else:
                plt_label = None
                sc_label = None

            # SC
            tmp_ax.errorbar(
                cone_degrees, beta_sc_c3[:, 0],
                beta_sc_c3[:, 1]/5, label=sc_label,
                elinewidth=0.3, capsize=2, **error_scatter_params
            )
            plot_points = np.linspace(min(cone_degrees), max(cone_degrees), 100)
            tmp_ax.plot(
                plot_points, A * beta_val(plot_points)**2,
                label=plt_label, **line_plot_params[1]
            )

            if dset_id == 0 and eps_id == 0:
                tmp_ax.legend()
            
            tmp_ax.set_xticks(cone_degrees[::4])
            tmp_ax.set_ylabel("Sample complexity")
            tmp_ax.set_xlabel(r"Cone angles ($^{\circ}$)")

    fig.tight_layout()
    plt.savefig(os.path.join(visualization_path, f'beta.pdf'))

def batched_paveba(exp_path, paveba_d_id, paveba_i_id):
    visualization_path = create_visualization_dir(exp_path)

    paveba_d_path = glob(os.path.join(exp_path, f"exp{paveba_d_id}-*"))[0]
    paveba_i_path = glob(os.path.join(exp_path, f"exp{paveba_i_id}-*"))[0]

    # This assumes batches are simulated in order while running.
    # This method does not sort batch sizes.
    paveba_d_results_list = read_sorted_results(paveba_d_path)
    paveba_i_results_list = read_sorted_results(paveba_i_path)

    assert len(paveba_d_results_list) == len(paveba_i_results_list)

    alg_names = ["PaVeBa-DE", "PaVeBa-IH"]
    datasets = []
    epsilons = []
    batch_sizes = []
    paveba_batched_sc = np.zeros((len(paveba_d_results_list)*2, 2))
    paveba_batched_f1 = np.zeros((len(paveba_d_results_list)*2, 2))
    
    # Evaluate each config
    i = 0
    for exp_dict in paveba_d_results_list:
        if exp_dict['eps'] not in epsilons:
            epsilons.append(exp_dict['eps'])
        if exp_dict['dataset_name'] not in datasets:
            datasets.append(exp_dict['dataset_name'])
        if exp_dict['batch_size'] not in batch_sizes:
            batch_sizes.append(exp_dict['batch_size'])

        result, result_std = evaluate_experiment(exp_dict)
        paveba_batched_sc[i] = [result["SC"], result_std["SC Std"]]
        paveba_batched_f1[i] = [result["F1E"], result_std["F1E Std"]]
        i += 1
    for exp_dict in paveba_i_results_list:
        result, result_std = evaluate_experiment(exp_dict)
        paveba_batched_sc[i] = [result["SC"], result_std["SC Std"]]
        paveba_batched_f1[i] = [result["F1E"], result_std["F1E Std"]]
        i += 1
    
    batch_sizes[batch_sizes.index(-1)] = DATASET_SIZES[datasets[0]]

    sc_ab2 = paveba_batched_sc.reshape(len(alg_names), len(batch_sizes), 2)
    f1_ab2 = paveba_batched_f1.reshape(len(alg_names), len(batch_sizes), 2)

    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    tmp_ax = ax[0]
    result_vals_dset = sc_ab2

    shift_i = 0
    for alg_i, alg in list(enumerate(alg_names))[::-1]:
        tmp_ax.bar(
            np.arange(len(batch_sizes)) + shift_i*0.4,
            result_vals_dset[alg_i, :, 0],
            yerr=result_vals_dset[alg_i, :, 1],
            width=0.4, label=alg, **bar_plot_params[alg_i]
        )
        shift_i += 1

    tmp_ax.set_xticks(np.arange(len(batch_sizes)), batch_sizes)
    tmp_ax.set_ylabel(r"Sample complexity")

    tmp_ax = ax[1]
    result_vals_dset = f1_ab2
    tmp_ax.set_ylim((0, 1.005))
    for alg_i, alg in enumerate(alg_names):
        tmp_ax.errorbar(
            np.arange(len(batch_sizes)), result_vals_dset[alg_i, :, 0],
            result_vals_dset[alg_i, :, 1], label=alg,
            **error_plot_params[alg_i], **error_bar_params
        )

    tmp_ax.set_xticks(np.arange(len(batch_sizes)), batch_sizes)
    tmp_ax.legend()
    tmp_ax.set_ylabel(r"$\epsilon$-F1 Score")

    fig.supxlabel(r"Batch size")
    fig.tight_layout()
    plt.savefig(os.path.join(visualization_path, f'batch_sc.png'))
    plt.savefig(os.path.join(visualization_path, f'batch_sc.pdf'))

if __name__ == "__main__":
    np.random.seed(SEED)

    intro_cone()
    
    # paveba_vs_beta(
    #     os.path.join("outputs", ""), paveba_id=2
    # )
    
    # batched_paveba(
    #     os.path.join("outputs", ""),
    #     paveba_d_id=2, paveba_i_id=1
    # )
