import utils
import torch
import hydra
import models
import ddpo
from omegaconf import OmegaConf, open_dict
import common
import os
import sys
import time


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
    being trained from, so two runs that differ only in from_checkpoint silently
    overwrite each other's latest.ckpt / samples. If log_dir already holds a
    config.yaml recorded with a different from_checkpoint, abort. Pass
    +allow_overwrite=true to downgrade the abort to a warning.
    """
    def normalize(ckpt):
        # matches the "none" handling used in load_checkpoint below
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
        f"  existing run started from: {prev_ckpt}\n"
        f"  this run would start from: {curr_ckpt}\n"
        f"  (both are relative to log_dir={cfg.log_dir})\n"
        f"The output directory is named '{cfg.task}.{cfg.method}.{cfg.seed}' and does not "
        f"include the checkpoint, so continuing would overwrite the existing run's "
        f"latest.ckpt, config.yaml and samples with no way to recover them.\n"
        f"Re-run with a distinct method= (e.g. method={cfg.method}_<checkpoint-name>) so this "
        f"run gets its own directory, or pass +allow_overwrite=true to overwrite on purpose."
    )
    if cfg.get("allow_overwrite", False):
        print(f"WARNING: {msg}")
        print("WARNING: allow_overwrite=true, overwriting anyway.")
    else:
        raise RuntimeError(msg)

@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    check_checkpoint_collision(cfg, log_dir)
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Preparing dataset
    train_set, val_set = utils.load_graph_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = train_set[0][0].shape
    dataloader = utils.GraphDataLoader(
        train_set, val_set, cfg.batch_size, cfg.val_batch_size, device,
        augment_dihedral = cfg.augment,
        edge_dropout_p = cfg.get("edge_dropout", 0) or 0,
    )
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "self_cond_diffusion", "skip_diffusion", "guided_diffusion", "skip_guided_diffusion", "flow_matching"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        elif cfg.family in ["mixed_diffusion"]:
            cfg.model.update({
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model, optimizer, and grad scaler (for AMP)
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel, # Use this!
        "flow_matching": models.FlowMatchingModel,
        "self_cond_diffusion": models.SelfCondDiffusionModel,
        "mixed_diffusion": models.ChipDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
    }
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled = (device == "cuda"))
    train_metrics = common.Metrics()
    if cfg.mode == "ddpo":
        ddpo_model = ddpo.DDPO(
            model,
            ddpo.get_reward_fn(cfg.ddpo.legality_weight, cfg.ddpo.hpwl_weight),
            cfg.batch_size,
            cfg.ddpo.ema_factor,
            num_timesteps=cfg.ddpo.get("num_timesteps", -1),
            clip_epsilon=cfg.ddpo.get("clip_epsilon", 0.0),
            local_reward_weight=cfg.ddpo.get("local_reward_weight", 0.0),
            local_reward_every=cfg.ddpo.get("local_reward_every", 5),
            local_reward_last_k=cfg.ddpo.get("local_reward_last_k", 0),
            )

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
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    model.train()

    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, cond = dataloader.get_batch("train")
        # x has (B, N, 2); netlist_data is a single graph in tg.Data format
        t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device)
        optim.zero_grad()
        if cfg.mode != "ddpo":
            addloss = cfg.get("addloss", {}) or {}
            loss, model_metrics = model.loss(
                x, cond, t,
                hpwl_weight=addloss.get("hpwl_weight", 0.0),
                legality_weight=addloss.get("legality_weight", 0.0),
                use_timestep_weighting=addloss.get("use_timestep_weighting", False),
            )
        else:
            ddpo_loss, model_metrics = ddpo_model.loss(x, cond)
            supervised_weight = cfg.ddpo.get("supervised_weight", 0.0)
            if supervised_weight > 0:
                supervised_loss, _ = model.loss(x, cond, t)
                loss = ddpo_loss + supervised_weight * supervised_loss
                model_metrics["supervised_loss"] = supervised_loss.detach().cpu().item()
            else:
                loss = ddpo_loss

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optim)
        grad_scaler.update()

        train_metrics.add({"loss": loss.detach().cpu().item()})
        train_metrics.add(model_metrics)
        step.increment()

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            x_val, cond_val = dataloader.get_batch("val")
            train_logs = utils.validate(x, model, cond)
            val_logs = utils.validate(x_val, model, cond_val)

            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            logger.add(train_metrics.result())
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")

            # display example images
            for split in ["train", "val"]:
                x_disp, cond_disp = dataloader.get_display_batch(cfg.display_examples, split = split)
                utils.display_graph_samples(cfg.display_examples, x_disp, cond_disp, model, logger, prefix = split)
                utils.display_forward_graph_samples(x_disp, cond_disp, model, logger, prefix = split)
            logger.write()
            t_1 = t_2

            checkpointer.save() # save latest checkpoint
            _ledger_record(log_dir)
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                checkpointer.save(os.path.join(log_dir, "best.ckpt"))
                print("saving best model")
            cond_val.to(device="cpu")

        if (cfg.eval_every > 0) and (int(step)) % cfg.eval_every == 0:
            print(f"saving model at step {int(step)}")
            checkpointer.save(os.path.join(log_dir, f"step_{int(step)}.ckpt"))
            print("generating evaluation report")
            t3 = time.time()
            utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
            logger.write()
            t4 = time.time()
            print(f"generated report in {t4-t3:.3f} sec")
        
        cond.to(device="cpu")

def load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler):
    checkpointer.register({
            "step": step,
            "model": model,
            "optim": optim,
            "grad_scaler": grad_scaler,
        })
    if cfg.mode == "train":
        checkpointer.load(
            path_override = None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
            else os.path.join(cfg.log_dir, cfg.from_checkpoint)
        )
    elif cfg.mode in ["finetune", "ddpo"]:
        # Try to resume existing run
        loaded = checkpointer.load()
        if not loaded:
            # No existing run, so load pre-trained model only
            loaded = checkpointer.load(
                path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                filter_keys = ["model"],
            )
            if not loaded:
                print("WARNING Failed to load checkpoint for finetuning. Training from scratch instead.")
    else:
        raise NotImplementedError

if __name__=="__main__":
    main()
