"""Training script for EgoAllo diffusion model using HuggingFace accelerate."""

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import tensorboardX
import torch.optim.lr_scheduler
import torch.utils.data
import tyro
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from loguru import logger

from egoallo import network, training_loss, training_utils
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.data.dataclass import collate_dataclass


@dataclasses.dataclass(frozen=True)
class EgoAlloTrainConfig:
    experiment_name: str
    dataset_hdf5_path: Path
    dataset_files_path: Path

    model: network.EgoDenoiserConfig = network.EgoDenoiserConfig()
    loss: training_loss.TrainingLossConfig = training_loss.TrainingLossConfig()

    # Dataset arguments.
    batch_size: int = 256
    """Effective batch size."""
    num_workers: int = 2
    subseq_len: int = 128
    dataset_slice_strategy: Literal[
        "deterministic", "random_uniform_len", "random_variable_len"
    ] = "random_uniform_len"
    dataset_slice_random_variable_len_proportion: float = 0.3
    """Only used if dataset_slice_strategy == 'random_variable_len'."""
    train_splits: tuple[Literal["train", "val", "test", "just_humaneva"], ...] = (
        "train",
        "val",
    )

    # Optimizer options.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Stop after this many steps (the loop is otherwise infinite). None = run forever.
    num_train_steps: int | None = None
    # SMPL-H model, used by the wrist-position loss (see TrainingLossConfig).
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")


def get_experiment_dir(experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorch lightning."""
    experiment_dir = (
        Path(__file__).absolute().parent
        / "experiments"
        / experiment_name
        / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_name, version + 1)
    else:
        return experiment_dir


def _load_model_weights_column_preserving(
    model: "network.EgoDenoiser", checkpoint_dir: Path
) -> None:
    """Load `model.safetensors` from a released checkpoint into `model`, tolerating
    a widened conditioning input.

    `latent_from_cond.weight` grows from ``[d_latent, d_cond_old]`` to
    ``[d_latent, d_cond_new]`` when a conditioning stream is added (e.g.
    `include_wrist_pose_cond`). We copy the pretrained columns and zero the new
    ones, so at step 0 the model reproduces the released checkpoint exactly and
    then learns to use the new channel. A fresh optimizer is used (we do not call
    `accelerator.load_state`), so pass this instead of `--restore-checkpoint-dir`.
    """
    from safetensors import safe_open

    st_path = checkpoint_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        pretrained = {k: f.get_tensor(k) for k in f.keys()}

    to_load: dict[str, "torch.Tensor"] = {}
    for k, v_own in model.state_dict().items():
        src = k
        if k not in pretrained and ".base." in k:
            # LoRA-wrapped layer: the released checkpoint stored it unwrapped.
            src = k.replace(".base.", ".")
        if src not in pretrained:
            if "lora_A" not in k and "lora_B" not in k:
                logger.warning(f"[restore-model-only] {k}: no pretrained tensor, keeping init")
            continue
        v_pre = pretrained[src]
        if v_pre.shape == v_own.shape:
            to_load[k] = v_pre
        elif (
            k == "latent_from_cond.weight"
            and v_pre.ndim == v_own.ndim == 2
            and v_pre.shape[0] == v_own.shape[0]
            and v_pre.shape[1] < v_own.shape[1]
        ):
            merged = torch.zeros_like(v_own)
            merged[:, : v_pre.shape[1]] = v_pre
            to_load[k] = merged
            logger.info(
                f"[restore-model-only] {k}: copied {v_pre.shape[1]} cols, "
                f"zeroed {v_own.shape[1] - v_pre.shape[1]}"
            )
        else:
            logger.warning(
                f"[restore-model-only] {k}: shape mismatch "
                f"{tuple(v_pre.shape)} vs {tuple(v_own.shape)}, keeping init"
            )
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    logger.info(
        f"[restore-model-only] loaded {len(to_load)} tensors from {st_path}; "
        f"{len(missing)} kept-init, {len(unexpected)} unexpected"
    )


def run_training(
    config: EgoAlloTrainConfig,
    restore_checkpoint_dir: Path | None = None,
    restore_model_only: Path | None = None,
) -> None:
    assert not (restore_checkpoint_dir is not None and restore_model_only is not None), (
        "Pass at most one of --restore-checkpoint-dir (full accelerate state) or "
        "--restore-model-only (weights only, fresh optimizer)."
    )
    # Set up experiment directory + HF accelerate.
    # We're getting to manage logging, checkpoint directories, etc manually,
    # and just use `accelerate` for distibuted training.
    experiment_dir = get_experiment_dir(config.experiment_name)
    assert not experiment_dir.exists()
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )
    writer = (
        tensorboardX.SummaryWriter(logdir=str(experiment_dir), flush_secs=10)
        if accelerator.is_main_process
        else None
    )
    device = accelerator.device

    # Initialize experiment.
    if accelerator.is_main_process:
        training_utils.pdb_safety_net()

        # Save various things that might be useful.
        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        (experiment_dir / "run_config.yaml").write_text(yaml.dump(config))
        (experiment_dir / "model_config.yaml").write_text(yaml.dump(config.model))

        # Add hyperparameters to TensorBoard.
        assert writer is not None
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(config),
            metric_dict={},
            name=".",  # Hack to avoid timestamped subdirectory.
        )

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")

    # Setup.
    model = network.EgoDenoiser(config.model)
    if restore_model_only is not None:
        _load_model_weights_column_preserving(model, restore_model_only)

    # LoRA: freeze the pretrained backbone; train only the LoRA adapters and the
    # hand-conditioning MLP. Forces the wrist signal to be learned through the
    # low-rank adaptation instead of being drowned out by the CPF backbone.
    if config.model.lora_rank > 0:
        for name, p in model.named_parameters():
            p.requires_grad_(
                ("lora_A" in name) or ("lora_B" in name) or ("hand_cond_proj" in name)
            )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"[lora] rank={config.model.lora_rank} targets={config.model.lora_targets}"
            f" -> trainable {n_train:,} / {n_total:,} ({100 * n_train / n_total:.1f}%)"
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=EgoAmassHdf5Dataset(
            config.dataset_hdf5_path,
            config.dataset_files_path,
            splits=config.train_splits,
            subseq_len=config.subseq_len,
            cache_files=True,
            slice_strategy=config.dataset_slice_strategy,
            random_variable_len_proportion=config.dataset_slice_random_variable_len_proportion,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        collate_fn=collate_dataclass,
        drop_last=True,
    )
    optim = torch.optim.AdamW(  # type: ignore
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )

    # HF accelerate setup. We use this for parallelism, etc!
    model, train_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, optim, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    # Restore an existing model checkpoint.
    if restore_checkpoint_dir is not None:
        accelerator.load_state(str(restore_checkpoint_dir))

    # Get the initial step count.
    if restore_checkpoint_dir is not None and restore_checkpoint_dir.name.startswith(
        "checkpoint_"
    ):
        step = int(restore_checkpoint_dir.name.partition("_")[2])
    else:
        step = int(scheduler.state_dict()["last_epoch"])
        assert step == 0 or restore_checkpoint_dir is not None, step

    # Save an initial checkpoint. Not a big deal but currently this has an
    # off-by-one error, in that `step` means something different in this
    # checkpoint vs the others.
    accelerator.save_state(str(experiment_dir / f"checkpoints_{step}"))

    # Run training loop!
    loss_helper = training_loss.TrainingLossComputer(
        config.loss, device=device, smplh_npz_path=config.smplh_npz_path
    )
    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=step)
    prev_checkpoint_path: Path | None = None
    while True:
        for train_batch in train_loader:
            loop_metrics = next(loop_metrics_gen)
            step = loop_metrics.counter

            if (
                config.num_train_steps is not None
                and step >= config.num_train_steps
            ):
                accelerator.save_state(str(experiment_dir / f"checkpoints_{step}"))
                if accelerator.is_main_process:
                    logger.info(f"Reached num_train_steps={step}; saved final checkpoint. Done.")
                return

            loss, log_outputs = loss_helper.compute_denoising_loss(
                model,
                unwrapped_model=accelerator.unwrap_model(model),
                train_batch=train_batch,
            )
            log_outputs["learning_rate"] = scheduler.get_last_lr()[0]
            accelerator.log(log_outputs, step=step)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            # The rest of the loop will only be executed by the main process.
            if not accelerator.is_main_process:
                continue

            # Logging.
            if step % 10 == 0:
                assert writer is not None
                for k, v in log_outputs.items():
                    writer.add_scalar(k, v, step)

            # Print status update to terminal.
            if step % 20 == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()
                logger.info(
                    f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                    f" mem: {(mem_total - mem_free) / 1024**3:.2f}/{mem_total / 1024**3:.2f}G"
                    f" lr: {scheduler.get_last_lr()[0]:.7f}"
                    f" loss: {loss.item():.6f}"
                )

            # Checkpointing.
            if step % 5000 == 0:
                # Save checkpoint.
                checkpoint_path = experiment_dir / f"checkpoints_{step}"
                accelerator.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Keep checkpoints from every 25k steps (denser than upstream's
                # 100k, so the best fine-tune step can be picked qualitatively).
                if prev_checkpoint_path is not None:
                    shutil.rmtree(prev_checkpoint_path)
                prev_checkpoint_path = None if step % 25_000 == 0 else checkpoint_path
                del checkpoint_path


if __name__ == "__main__":
    tyro.cli(run_training)
