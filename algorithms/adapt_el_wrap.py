import logging
from functools import partial
from itertools import product
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor as Pool


from algorithms import adapt_el_algorithm
from algorithms import adapt_el_algorithm_gp
import utils.dataset
from utils.seed import SEED
from utils.utils import set_seed, get_cone_params, save_new_result


def simulate_once(
    i, gp_dict, dataset_name, cone_degree, delt, noise, eps, conf_contraction, batch_size
):
    set_seed(SEED + i + 1)

    use_gp = False if gp_dict is None or not gp_dict["use_gp"] else True
    alg_module = adapt_el_algorithm_gp if use_gp else adapt_el_algorithm

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)

    W_CONE, _, _ = get_cone_params(cone_degree, dataset.out_dim)

    # Run the algorithm and return the pareto set and the sample count
    if not use_gp:
        alg = alg_module.PaVeBa(
            *dataset.in_data.shape, dataset.out_dim, W_CONE, noise, delt, dataset.model_kernel,
            eps, conf_contraction, batch_size
        )
    else:
        alg = alg_module.PaVeBa(
            *dataset.in_data.shape, dataset.out_dim, W_CONE, noise, delt, dataset.model_kernel,
            eps, conf_contraction, batch_size,
            gp_dict["independent"], use_ellipsoid=gp_dict["ellipsoid"]
        )

    alg.prepare(dataset.in_data, dataset.out_data)

    pred_pareto, samples = alg.run()
    logging.info(f"DONE {i}.")

    return [[samples, pred_pareto]]


def adaptive_elimination(
    gp_dict, datasets_and_workers, cone_degrees, noise_var, delta, epsilons, iteration,
    conf_contractions, output_folder_path, alg_config
):
    batch_sizes = gp_dict.get("batch_sizes", [None]) if gp_dict else [None]
    
    # dset, eps, cone, conf, batch
    for dataset_name, dataset_worker in datasets_and_workers:
        for eps in epsilons:
            alg_independent_params = product(cone_degrees, conf_contractions)

            for cone_degree, conf_contraction in alg_independent_params:
                for batch_size in batch_sizes:
                    simulate_part = partial(
                        simulate_once,
                        dataset_name=dataset_name,
                        gp_dict=gp_dict,
                        cone_degree=cone_degree,
                        delt=delta,
                        noise=noise_var,
                        eps=eps,
                        conf_contraction=conf_contraction,
                        batch_size=batch_size
                    )

                    # results = []
                    with Pool(max_workers=dataset_worker) as pool:
                        # future_to_task = {
                        #     pool.submit(simulate_part, it): it for it in range(iteration)
                        # }

                        # while future_to_task:
                        #     for future in concurrent.futures.as_completed(
                        #         future_to_task, timeout=1800
                        #     ):
                        #         task_it = future_to_task.pop(future)

                        #         try:
                        #             result = future.result()
                        #             results.append(result)
                        #         except concurrent.futures.TimeoutError:
                        #             # Resubmit the task that timed out
                        #             logging.info(f"Task {task_it} timed out, resubmitting.")
                        #             future_to_task[pool.submit(simulate_part, task_it)] = task_it

                        results = pool.map(
                            simulate_part,
                            range(iteration),
                        )

                    results = list(results)
                    
                    experiment_res_dict = {
                        "dataset_name": dataset_name,
                        "cone_degree": cone_degree,
                        "alg": "PaVeBa",
                        "delta": delta,
                        "noise_var": noise_var,
                        "eps": eps,
                        "disc": -1,
                        "conf_contraction": conf_contraction,
                        "batch_size": batch_size if batch_size else -1,
                        "results": results
                    }

                    save_new_result(output_folder_path, experiment_res_dict)
