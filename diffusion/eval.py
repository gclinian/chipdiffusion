import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import legalization
import analysis_utils
import common
import os
import sys
import time
import wandb

def cost(output_metrics):
    """
    Returns dict with cost function(s) for hyperparam sweep
    """
    legality_target = 0.995
    macro_legality_target = 0.998
    legality_temp = 0.001
    hpwl = torch.tensor(output_metrics["hpwl_rescaled"]).mean()
    
    legality = torch.tensor(output_metrics["legality_2"]).mean()
    legality_cost_factor = 1 + 10 * torch.nn.functional.relu((legality_target - legality)/legality_temp)
    
    macro_legality = torch.tensor(output_metrics["macro_legality"]).mean()
    macro_legality_cost_factor = 1 + 10 * torch.nn.functional.relu((macro_legality_target - macro_legality)/legality_temp)
    
    full_cost = (legality_cost_factor * hpwl).item()
    macro_cost = (macro_legality_cost_factor * hpwl).item()
    costs = {
        "cost": full_cost,
        "macro_cost": macro_cost,
    }
    return costs


def _ledger_record(log_dir):
    """Append this run to docs/ledger/runs.jsonl and archive its metrics.csv.

    Deliberately unconditional and in-process: a run launched by hand in a plain
    shell still lands in the record, with no hook required. Deliberately fatal to
    nothing -- a bookkeeping failure must never cost an experiment.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts"))
        import ledger
        rid = ledger.record(log_dir)
        if rid:
            print(f"ledger: recorded {rid}")
    except Exception as e:
        print(f"ledger: skipped ({type(e).__name__}: {e})")

def check_checkpoint_collision(cfg, log_dir):
    """
    Guards against two different checkpoints writing into the same run directory.

    log_dir is named f"{task}.{method}.{seed}" and does NOT include the checkpoint
    being evaluated, so two runs that differ only in from_checkpoint silently
    overwrite each other's metrics.csv / samples. If log_dir already holds a
    config.yaml recorded with a different from_checkpoint, abort. Pass
    +allow_overwrite=true to downgrade the abort to a warning.
    """
    def normalize(ckpt):
        # matches the "none" handling used when loading checkpoints below
        return "none" if ckpt in [None, "none", "None", ""] else str(ckpt)

    prev_cfg_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(prev_cfg_path):
        return
    try:
        prev_cfg = utils.load_cfg(prev_cfg_path)
        has_prev = "from_checkpoint" in prev_cfg
        prev_ckpt = normalize(prev_cfg.from_checkpoint) if has_prev else None
    except Exception as e:
        print(f"WARNING: could not read {prev_cfg_path} ({e}). Skipping checkpoint collision check.")
        return
    if not has_prev:
        print(f"WARNING: {prev_cfg_path} records no from_checkpoint. Skipping checkpoint collision check.")
        return
    curr_ckpt = normalize(cfg.from_checkpoint)
    if prev_ckpt == curr_ckpt:
        return  # same checkpoint: legitimate re-run / resume
    msg = (
        f"checkpoint collision in {log_dir}\n"
        f"  existing results were produced by: {prev_ckpt}\n"
        f"  this run would use:                {curr_ckpt}\n"
        f"  (both are relative to log_dir={cfg.log_dir})\n"
        f"The output directory is named '{cfg.task}.{cfg.method}.{cfg.seed}' and does not "
        f"include the checkpoint, so continuing would overwrite the existing run's "
        f"metrics.csv, config.yaml and samples with no way to recover them.\n"
        f"Re-run with a distinct method= (e.g. method={cfg.method}_<checkpoint-name>) so this "
        f"evaluation gets its own directory, or pass +allow_overwrite=true to overwrite on purpose."
    )
    if cfg.get("allow_overwrite", False):
        print(f"WARNING: {msg}")
        print("WARNING: allow_overwrite=true, overwriting anyway.")
    else:
        raise RuntimeError(msg)

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)

    # Prepare legalization function
    if cfg.legalization.mode in [None, "none", "None", ""]:
        legalize_fn = None
    elif cfg.legalization.mode == "scheduled":
        def legalize_fn(x, cond):
            return legalization.legalize(
                x, 
                cond,
                **cfg.legalization,
                )
    elif cfg.legalization.mode == "opt":
        def legalize_fn(x, cond):
            return legalization.legalize_opt(
                x, 
                cond,
                **cfg.legalization,
                )
    # Prepare pre and post processing functions. Note that postprocess fns are applied in reverse order
    preprocess_fns = []
    postprocess_fns = []
    if cfg.cluster.is_cluster:
        def cluster_preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        preprocess_fns.append(cluster_preprocess_fn)
        postprocess_fns.append(cluster_postprocess_fn)
    elif cfg.cluster.cached_clusters:
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        postprocess_fns.append(cluster_postprocess_fn)
    if cfg.sc_halo != 1.0:
        def resize_standard_cells(x, cond):
            _, _, sc_mask = analysis_utils.get_masks(x, cond)
            is_resize = sc_mask.float()
            size_multiplier = (is_resize * cfg.sc_halo) + ((1-is_resize))
            cond.x = cond.x * size_multiplier.unsqueeze(dim=-1)
            return x, cond
        preprocess_fns.append(resize_standard_cells)
    if cfg.edge_dropout > 0.0: # used for debugging
        def edge_dropout(x, cond):
            x, cond = utils.edge_dropout(x, cond, cfg.edge_dropout)
            return x, cond
        preprocess_fns.append(edge_dropout)
    if cfg.macros_only:
        if cfg.cached_macros:
            postprocess_fns.append(utils.add_non_macros)
        else:
            preprocess_fns.append(utils.remove_non_macros)
            postprocess_fns.append(utils.add_non_macros)
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for i, postprocess_step in enumerate(reversed(postprocess_fns)):
            x, cond = postprocess_step(x, cond)    
        return x, cond

    # Preparing dataset
    train_set, val_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = val_set[0][0].shape
    dataloader = utils.GraphDataLoader(
        train_set, 
        val_set, 
        cfg.val_batch_size, 
        cfg.val_batch_size, 
        device,
        preprocess_fn = preprocess_fn,
        val_shuffle = False, # Don't shuffle validation set
        )
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "guided_diffusion", "skip_diffusion", "skip_guided_diffusion", "no_model", "flow_matching"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel,
        "flow_matching": models.FlowMatchingModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
        "no_model": models.NoModel,
    }
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}" if not cfg.param_sweep else None
        wandb_output = common.logger.WandBOutput(wandb_run_name, cfg)
        if cfg.param_sweep:
            with open_dict(cfg):  # for eval/debugging
                cfg.update({
                    "method": f"{cfg.method}.{wandb_output._wandb.run.name}",
                })
        else:
            print("WARNING: param_sweep set to true but wandb disabled. Continuing anyways...")
        outputs.append(wandb_output)
    step = common.Counter()
    logger = common.Logger(step, outputs)

    # Create log and output directories
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    check_checkpoint_collision(cfg, log_dir)
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"saving checkpoints to: {log_dir}")

    # Output config used
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint if exists. Here we only load the model
    checkpointer.register({
        "model": model,
    })
    checkpointer.load(
        None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
        else os.path.join(cfg.log_dir, cfg.from_checkpoint)
    )

    # Start training
    print(f"model has {num_params} params")
    print(f"==== Start Eval on Device: {device} ====")

    if cfg.eval_samples > 0:
        print("generating evaluation report")
        t1 = time.time()
        utils.generate_report(
            cfg.eval_samples, 
            dataloader, 
            model, 
            logger, 
            policy = cfg.eval_policy_algorithm, 
            intermediate_every = cfg.show_intermediate_every,
            )
        logger.write()
        t2 = time.time()
        print(f"generated report in {t2-t1:.3f} sec")

    # output eval samples
    t3 = time.time()
    print("generating output samples")
    output_metrics = {}
    log_metrics = common.Metrics()
    start_sample = cfg.get("start_sample", 0)
    skip_guidance_threshold = cfg.get("skip_guidance_threshold", 0)  # 0 = never skip
    num_candidates = cfg.get("num_candidates", 1)  # 1 = one sample, i.e. current behaviour
    candidate_legality_floor = cfg.get("candidate_legality_floor", 0.97)
    legalize_all_candidates = cfg.get("legalize_all_candidates", False)
    if num_candidates > 1 and cfg.eval_policy_algorithm != "open_loop":
        raise ValueError(f"num_candidates>1 needs eval_policy_algorithm=open_loop, got {cfg.eval_policy_algorithm}")
    for i in range(start_sample, cfg.num_output_samples):
        x, cond = val_set[i]
        # Temporarily disable guidance for large circuits to avoid OOM
        if skip_guidance_threshold > 0 and cfg.macros_only:
            num_macros = cond.is_macros.sum().item() if hasattr(cond, 'is_macros') else x.shape[0]
            if num_macros > skip_guidance_threshold:
                print(f"Sample {i}: {num_macros} macros > threshold {skip_guidance_threshold}, disabling guidance")
                original_guidance_mode = model.guidance_mode
                original_is_guided = model.is_guided_sampling
                model.guidance_mode = "none"
                model.is_guided_sampling = False
            else:
                original_guidance_mode = None
        else:
            original_guidance_mode = None
        metrics, metrics_special, image, image_legalized = utils.save_outputs(
            x,
            cond,
            model,
            save_folder=sample_dir,
            output_number_offset=0,
            policy=cfg.eval_policy_algorithm,
            policy_kwargs=cfg.eval_policy,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            legalization_fn=legalize_fn,
            num_candidates=num_candidates,
            candidate_legality_floor=candidate_legality_floor,
            legalize_all_candidates=legalize_all_candidates,
        )
        print(f"Finished sample {i+1} of {cfg.num_output_samples} \t {metrics}")
        # Restore guidance mode if it was disabled
        if original_guidance_mode is not None:
            model.guidance_mode = original_guidance_mode
            model.is_guided_sampling = original_is_guided
        t5 = time.time()
        # additional metrics
        eig_vals = analysis_utils.get_spectral_info(x, cond, k=1)
        metrics.update({
            "num_vertices": x.shape[0],
            "num_edges": cond.edge_index.shape[1],
            "lambda_2": eig_vals[0],
        })
        logger.add({
            "reverse_samples": {
                **metrics,
                **metrics_special,
                "image": wandb.Image(image_legalized),
                "image_raw": wandb.Image(image),
                "time_elapsed": t5-t3,
            }
        })
        # update metrics
        for k, v in metrics.items():
            if k in output_metrics:
                output_metrics[k].append(v)
            else:
                output_metrics[k] = [v]
        log_metrics.add({k: v for k, v in metrics.items() if not isinstance(v, str)})
    utils.dict_to_csv(output_metrics, os.path.join(log_dir,"metrics.csv"))
    _ledger_record(log_dir)
    for plot_keys in cfg.scatter_plots:
        x_name = plot_keys[0]
        y_name = plot_keys[1]
        if x_name in output_metrics and y_name in output_metrics:
            scatter_plot = utils.plot_scatter(output_metrics[x_name], output_metrics[y_name], x_title=x_name, y_title=y_name)
            logger.add({f"{x_name}_vs_{y_name}": scatter_plot})
    logger.add(log_metrics.result())
    logger.add(cost(output_metrics), prefix = "sweep")
    logger.write()

if __name__=="__main__":
    main()
