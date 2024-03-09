import os
import pickle
import argparse
from glob import glob
from pathlib import Path

import numpy as np

import utils.dataset
from utils.utils import (
    get_cone_params, get_uncovered_set, read_sorted_results, get_closest_indices_from_points
)


def evaluate_experiment(exp_dict, round_idx=-1):
    dataset_name = exp_dict["dataset_name"]
    cone_degree = exp_dict["cone_degree"]
    cover_eps = exp_dict["eps"]

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)
    if isinstance(dataset, utils.dataset.ContinuousDataset):
        dataset = getattr(utils.dataset, dataset_name+"Wrapper")(cone_degree, dataset)
    delta_cone, true_pareto_indices = dataset.get_params()

    W_CONE, _, _ = get_cone_params(cone_degree, dim=dataset.out_dim)


    result_keys = ['F1E', 'SC',]
    result_sum = np.full((len(exp_dict["results"]), len(result_keys)), np.nan)
    for res_i, iter_result in enumerate(exp_dict["results"]):
        # Calculate for only the round_idx'th round
        if round_idx >= len(iter_result):
            continue
        
        samples, pred_pareto_pts = iter_result[round_idx]
        pred_pareto_pts = np.array(pred_pareto_pts).reshape(-1, dataset.in_dim)
        pred_pareto_indices = get_closest_indices_from_points(pred_pareto_pts, dataset.in_data)

        pred_set = set(pred_pareto_indices)
        gt_set = set(true_pareto_indices)

        indices_of_missed_pareto = list(gt_set - pred_set)

        # Returns non-covered pareto indices that are missed
        uncovered_missed_pareto_indices = get_uncovered_set(
            indices_of_missed_pareto, pred_pareto_indices, dataset.out_data, cover_eps, W_CONE
        )

        true_eps = np.sum(delta_cone[pred_pareto_indices] <= cover_eps, axis=0)[0]

        tp_eps = true_eps
        fp_eps = len(pred_set) - true_eps
        f1_eps = (2 * tp_eps) / (2*tp_eps + fp_eps + len(uncovered_missed_pareto_indices))

        result_sum[res_i] = [
            f1_eps,
            samples,
        ]

    result = np.nanmean(result_sum, axis=0)
    result_std = np.nanstd(result_sum, axis=0)

    result_dict = dict(zip(result_keys, np.around(result, 2).tolist()))
    result_std_dict = dict(zip(
        list(map(lambda x: x+' Std', result_keys)),
        np.around(result_std, 2).tolist()
    ))

    return result_dict, result_std_dict


if __name__ == "__main__":
    exp_path = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=Path, required=False, default=None)
    args = parser.parse_args()

    if args.exp_name:
        exp_path = os.path.join("outputs", args.exp_name)

    # If no path is given, just evaluate the last experiment
    if exp_path is None:
        exp_path = sorted([
            subpath
            for subpath in glob(os.path.join("outputs", "*"))
            if os.path.isdir(subpath)
        ])[-1]

    algorithm_names = sorted(
        [
            subpath
            for subpath in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, subpath))
        ],
        key=lambda x: x.split('-')[0]
    )
    algorithm_names.sort(key=lambda x: len(x.split('-')[0]))
    
    for alg_name in algorithm_names:
        alg_text = alg_name.split('-')[-1]

        # Load results file
        alg_path = os.path.join(exp_path, alg_name)
        results_list = read_sorted_results(alg_path)

        print(
            "---   "
            f"Algorithm: {alg_text}"
            f", Iteration count: {len(results_list[0]['results'])}"
            "   ---"
        )

        # Evaluate each config
        for exp_dict in results_list:
            result, result_std = evaluate_experiment(exp_dict)
            
            for (k, v), std_v in zip(result.items(), result_std.values()):
                if k == "SC":
                    result[k] = f"{v:05.2f} ± {std_v:04.2f}"
                else:
                    result[k] = f"{v:06.2f} ± {std_v:05.2f}"

            print(
                f"D.set: {exp_dict['dataset_name']:<16}"
                f"Cone: {exp_dict['cone_degree']:<4}"
                f"Eps.: {exp_dict['eps']:<6}",
                f"Cont.: {exp_dict['conf_contraction']:<4}",
                f"B.S.: {exp_dict['batch_size']:<4}",
                result
            )
        print()
