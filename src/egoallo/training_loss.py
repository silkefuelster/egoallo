"""Training loss configuration."""

import dataclasses
from pathlib import Path
from typing import Literal

import torch.utils.data
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from . import fncsmpl, network
from .data.amass import EgoTrainingData
from .hand_cond import assemble_hand_cond, fov_visibility_mask
from .sampling import CosineNoiseScheduleConstants
from .transforms import SE3, SO3


@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.1,
            "body_rotmats": 1.0,
            "contacts": 0.1,
            # We don't have many hands in the AMASS dataset...
            "hand_rotmats": 0.01,
        }.copy
    )
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""

    # --- Wrist-pose conditioning: how observed hands are simulated at train time. ---
    # (Only used when the model has `include_wrist_pose_cond=True`.)
    wrist_cond_fov_half_deg: float = 55.0
    """Half-angle of the CPF forward cone a wrist must fall in to count as "observed"."""
    wrist_cond_fov_pitch_deg: float = 0.0
    """Downward pitch of that cone, to approximate the Aria RGB camera mounting."""
    wrist_cond_fov_max_range_m: float = 1.25
    """Max wrist distance from the CPF to count as "observed"."""
    wrist_cond_seq_dropout_prob: float = 0.05
    """Probability an entire sequence gets no wrist conditioning at all (keeps the
    model able to run with no hand tracking at inference)."""
    wrist_cond_frame_dropout_prob: float = 0.02
    """Per-hand, per-frame probability an otherwise-visible wrist is dropped.
    Kept low on purpose: within a tracked sequence we want the model to treat a
    visible wrist as near-non-negotiable."""
    wrist_cond_pos_noise_std: float = 0.02
    """Std (metres) of Gaussian noise added to observed wrist positions."""
    wrist_cond_rot_noise_deg: float = 12.0
    """Std (degrees) of random rotation noise added to observed wrist orientations."""
    wrist_cond_pos_loss_weight: float = 5.0
    """Weight on an explicit loss (via SMPL-H FK on the predicted body) between
    the predicted wrist positions and the *conditioned* wrist positions, on
    observed frames. Forces the model to use the signal. 0 disables it.
    TUNE: too high -> nails wrist position but picks crossed / contorted arm
    configs (body_rotmats MSE gets drowned out); too low -> ignores the signal
    (T-pose). Watch `loss_term/wrist_cond_pos` vs `loss_term/body_rotmats`."""
    wrist_cond_pos_loss_clamp: float = 0.04
    """Per-coordinate squared-error cap (m^2; 0.04 = 20 cm) on the wrist-position
    loss, so one bad-pose sample can't spike the gradient (fp16 stability)."""
    wrist_cond_pos_loss_uniform_t: bool = True
    """Weight the wrist-position loss uniformly over diffusion timesteps instead
    of by `weight_t` (which down-weights high noise). High-t is where x_t is
    near-pure noise and the model must rely on the conditioning, so that's where
    we most want to train 'conditioning -> wrist position'."""

    # --- CPF conditioning corruption ---------------------------------------------
    # On AMASS the released model can already infer arm pose from head/CPF motion
    # (it's overfit + head motion correlates with arm swing for locomotion), so
    # it never has to use `hand_cond`. Corrupting the CPF motion signal at train
    # time removes that crutch: with unreliable head motion, minimizing the loss
    # requires leaning on the wrist conditioning. The signal is clean at inference
    # (real Aria SLAM), so this is train-only regularization.
    cpf_cond_noise_rot_deg: float = 2.0
    """Std (deg) of rotation noise on the per-frame relative CPF transform
    (T_cpf_tm1_cpf_t). Kept mild -- and scaled by (1 - alpha_bar_t), so low-noise
    diffusion steps (where the pose is already in x_t) see ~no corruption and
    only high-noise steps (where the model generates from conditioning) get the
    full amount. 0 disables CPF corruption."""
    cpf_cond_noise_trans_m: float = 0.01
    """Std (m) of translation noise on the relative CPF transform."""
    cpf_cond_noise_heavy_prob: float = 0.05
    """Per-sequence probability of applying `cpf_cond_noise_heavy_scale`x the
    noise (simulates a stretch of bad head tracking -> lean on hand_cond)."""
    cpf_cond_noise_heavy_scale: float = 3.0


class TrainingLossComputer:
    """Helper class for computing the training loss. Contains a single method
    for computing a training loss."""

    def __init__(
        self,
        config: TrainingLossConfig,
        device: torch.device,
        smplh_npz_path: Path | None = None,
    ) -> None:
        self.config = config
        self.noise_constants = (
            CosineNoiseScheduleConstants.compute(timesteps=1000)
            .to(device)
            .map(lambda tensor: tensor.to(torch.float32))
        )

        # SMPL-H body model, only loaded if we need the wrist-position loss.
        self.body_model: fncsmpl.SmplhModel | None = None
        if config.wrist_cond_pos_loss_weight > 0.0 and smplh_npz_path is not None:
            self.body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

        # Emulate loss weight that would be ~equivalent to epsilon prediction.
        #
        # This will penalize later errors (close to the end of sampling) much
        # more than earlier ones (at the start of sampling).
        assert self.config.weight_loss_by_t == "emulate_eps_pred"
        weight_t = self.noise_constants.alpha_bar_t / (
            1 - self.noise_constants.alpha_bar_t
        )
        # Pad for numerical stability, and scale between [padding, 1.0].
        padding = 0.01
        self.weight_t = weight_t / weight_t[1] * (1.0 - padding) + padding

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model.

        Returns:
            A tuple (loss tensor, dictionary of things to log).
        """
        log_outputs: dict[str, Tensor | float] = {}

        batch, time, num_joints, _ = train_batch.body_quats.shape
        assert num_joints == 21
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=None,
            )
        x_0_packed = x_0.pack()
        device = x_0_packed.device
        assert x_0_packed.shape == (batch, time, unwrapped_model.get_d_state())

        # Diffuse.
        t = torch.randint(
            low=1,
            high=unwrapped_model.config.max_t + 1,
            size=(batch,),
            device=device,
        )
        eps = torch.randn(x_0_packed.shape, dtype=x_0_packed.dtype, device=device)
        assert self.noise_constants.alpha_bar_t.shape == (
            unwrapped_model.config.max_t + 1,
        )
        alpha_bar_t = self.noise_constants.alpha_bar_t[t, None, None]
        assert alpha_bar_t.shape == (batch, 1, 1)
        x_t_packed = (
            torch.sqrt(alpha_bar_t) * x_0_packed + torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        hand_positions_wrt_cpf: Tensor | None = None
        if unwrapped_model.config.include_hand_positions_cond:
            # Joints 19 and 20 are the hand positions.
            hand_positions_wrt_cpf = train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(
                (batch, time, 6)
            )

            # Exclude hand positions for some items in the batch. We'll just do
            # this by passing in zeros.
            hand_positions_wrt_cpf = torch.where(
                # Uniformly drop out with some uniformly sampled probability.
                # :)
                (
                    torch.rand((batch, time, 1), device=device)
                    < torch.rand((batch, 1, 1), device=device)
                ),
                hand_positions_wrt_cpf,
                0.0,
            )

        # First-party: assemble the observed wrist-pose conditioning, simulating
        # which hands the ego camera would actually see (FOV gate), detector
        # misses (dropout), and estimator error (noise).
        hand_cond: Tensor | None = None
        wrist_cond_valid: Tensor | None = None  # (b, t, 2) bool, kept for the wrist loss
        if getattr(unwrapped_model.config, "include_wrist_pose_cond", False):
            cfg = self.config
            wrist_pos = train_batch.wrist_pos_wrt_cpf  # (b, t, 2, 3)
            wrist_rot = train_batch.wrist_rot_wrt_cpf  # (b, t, 2, 3, 3)
            palm_pos = train_batch.palm_pos_wrt_cpf  # (b, t, 2, 3)
            palm_normal = train_batch.palm_normal_wrt_cpf  # (b, t, 2, 3)

            visible = fov_visibility_mask(
                wrist_pos,
                half_fov_deg=cfg.wrist_cond_fov_half_deg,
                pitch_deg=cfg.wrist_cond_fov_pitch_deg,
                max_range_m=cfg.wrist_cond_fov_max_range_m,
            )  # (b, t, 2)
            frame_keep = (
                torch.rand((batch, time, 2), device=device)
                > cfg.wrist_cond_frame_dropout_prob
            )
            seq_keep = (
                torch.rand((batch, 1, 1), device=device)
                > cfg.wrist_cond_seq_dropout_prob
            )
            valid = visible & frame_keep & seq_keep  # (b, t, 2)
            wrist_cond_valid = valid

            enc = getattr(
                unwrapped_model.config, "wrist_cond_encoding", "palm_raw"
            )
            wrist_pos = wrist_pos + torch.randn_like(wrist_pos) * (
                cfg.wrist_cond_pos_noise_std
            )
            palm_pos = palm_pos + torch.randn_like(palm_pos) * (
                cfg.wrist_cond_pos_noise_std
            )
            # Small random rotation to mimic Aria estimator error (the measured
            # train/inference disagreement on the palm normal is ~11 deg after the
            # left-hand sign fix -- see verify_palm_normal.py).
            rot_noise = SO3.exp(
                torch.randn((batch, time, 2, 3), device=device)
                * (cfg.wrist_cond_rot_noise_deg * torch.pi / 180.0)
            ).as_matrix()  # (b, t, 2, 3, 3)
            if enc == "rot6d":
                wrist_rot = torch.einsum("...ij,...jk->...ik", wrist_rot, rot_noise)
            else:  # palm_raw: jitter the palm-normal direction
                palm_normal = torch.einsum("...ij,...j->...i", rot_noise, palm_normal)

            hand_cond = assemble_hand_cond(
                valid=valid,
                wrist_pos_cpf=wrist_pos,
                wrist_rot_cpf=wrist_rot,
                palm_pos_cpf=palm_pos,
                palm_normal_cpf=palm_normal,
                encoding=enc,
            )  # (b, t, 20)

        # First-party: corrupt the CPF motion signal so the model can't infer arm
        # pose from head motion alone and must lean on `hand_cond`. Train-only:
        # the FK wrist loss and root recovery still use the clean `train_batch`.
        T_cpf_tm1_cpf_t_in = train_batch.T_cpf_tm1_cpf_t
        _c = self.config
        if (
            getattr(unwrapped_model.config, "include_wrist_pose_cond", False)
            and (_c.cpf_cond_noise_rot_deg > 0.0 or _c.cpf_cond_noise_trans_m > 0.0)
        ):
            heavy = (
                torch.rand((batch, 1, 1), device=device)
                < _c.cpf_cond_noise_heavy_prob
            ).to(T_cpf_tm1_cpf_t_in.dtype)
            # more corruption when the model relies on conditioning (high noise):
            # (1 - alpha_bar_t) is ~0 at low t, ~1 at high t.
            scale = (1.0 + heavy * (_c.cpf_cond_noise_heavy_scale - 1.0)) * (
                1.0 - alpha_bar_t
            )  # (b,1,1)
            rot_n = torch.randn((batch, time, 3), device=device) * (
                _c.cpf_cond_noise_rot_deg * torch.pi / 180.0
            ) * scale
            trans_n = (
                torch.randn((batch, time, 3), device=device)
                * _c.cpf_cond_noise_trans_m
                * scale
            )
            noise = SE3.from_rotation_and_translation(SO3.exp(rot_n), trans_n)
            T_cpf_tm1_cpf_t_in = (SE3(T_cpf_tm1_cpf_t_in) @ noise).parameters()

        # Denoise.
        x_0_packed_pred = model.forward(
            x_t_packed=x_t_packed,
            t=t,
            T_world_cpf=train_batch.T_world_cpf,
            T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t_in,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
            hand_cond=hand_cond,
            project_output_rotmats=False,
            mask=train_batch.mask,
            cond_dropout_keep_mask=torch.rand((batch,), device=device)
            > self.config.cond_dropout_prob
            if self.config.cond_dropout_prob > 0.0
            else None,
        )
        assert isinstance(x_0_packed_pred, torch.Tensor)
        x_0_pred = network.EgoDenoiseTraj.unpack(
            x_0_packed_pred, include_hands=unwrapped_model.config.include_hands
        )

        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            # bt stands for "batch time"
            bt_mask: Bool[Tensor, "b t"] = train_batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(train_batch.mask),
        ) -> Float[Tensor, ""]:
            """Weight and mask per-timestep losses (squared errors)."""
            _, _, d = loss_per_step.shape
            assert loss_per_step.shape == (batch, time, d)
            assert bt_mask.shape == (batch, time)
            assert weight_t.shape == (batch,)
            return (
                # Sum across b axis.
                torch.sum(
                    # Sum across t axis.
                    torch.sum(
                        # Mean across d axis.
                        torch.mean(loss_per_step, dim=-1) * bt_mask,
                        dim=-1,
                    )
                    * weight_t
                )
                / bt_mask_sum
            )

        loss_terms: dict[str, Tensor | float] = {
            "betas": weight_and_mask_loss(
                # (b, t, 16)
                (x_0_pred.betas - x_0.betas) ** 2
                # (16,)
                * x_0.betas.new_tensor(self.config.beta_coeff_weights),
            ),
            "body_rotmats": weight_and_mask_loss(
                # (b, t, 21 * 3 * 3)
                (x_0_pred.body_rotmats - x_0.body_rotmats).reshape(
                    (batch, time, 21 * 3 * 3)
                )
                ** 2,
            ),
            "contacts": weight_and_mask_loss((x_0_pred.contacts - x_0.contacts) ** 2),
        }

        # Include hand objective.
        # We didn't use this in the paper.
        if unwrapped_model.config.include_hands:
            assert x_0_pred.hand_rotmats is not None
            assert x_0.hand_rotmats is not None
            assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

            # Detect whether or not hands move in a sequence.
            # We should only supervise sequences where the hands are actully tracked / move;
            # we mask out hands in AMASS sequences where they are not tracked.
            gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
            hand_motion = (
                torch.sum(  # (b,) from (b, t)
                    torch.sum(  # (b, t) from (b, t, d)
                        torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
                    )
                    # Zero out changes in masked frames.
                    * train_batch.mask,
                    dim=-1,
                )
                > 1e-5
            )
            assert hand_motion.shape == (batch,)

            hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
            loss_terms["hand_rotmats"] = torch.sum(
                weight_and_mask_loss(
                    (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
                        batch, time, 30 * 3 * 3
                    )
                    ** 2,
                    bt_mask=hand_bt_mask,
                    # We want to weight the loss by the number of frames where
                    # the hands actually move, but gradients here can be too
                    # noisy and put NaNs into mixed-precision training when we
                    # inevitably sample too few frames. So we clip the
                    # denominator.
                    bt_mask_sum=torch.maximum(
                        torch.sum(hand_bt_mask), torch.tensor(256, device=device)
                    ),
                )
            )
            # self.log(
            #     "train/hand_motion_proportion",
            #     torch.sum(hand_motion) / batch,
            # )
        else:
            loss_terms["hand_rotmats"] = 0.0

        assert loss_terms.keys() == self.config.loss_weights.keys()

        # Log loss terms.
        for name, term in loss_terms.items():
            log_outputs[f"loss_term/{name}"] = term

        # Return loss.
        loss = sum([loss_terms[k] * self.config.loss_weights[k] for k in loss_terms])
        assert isinstance(loss, Tensor)
        assert loss.shape == ()

        # First-party: explicit wrist-position loss. Run SMPL-H FK on the
        # predicted body and penalise the predicted wrists deviating from the
        # *conditioned* wrist positions, on observed frames -- this is what forces
        # the model to treat a visible wrist as a hard target, not an optional
        # hint. Target is the clean GT (the conditioning it saw was a noisy
        # version, so the best it can do is trust it).
        if (
            self.body_model is not None
            and wrist_cond_valid is not None
            and self.config.wrist_cond_pos_loss_weight > 0.0
        ):
            shaped = self.body_model.with_shape(train_batch.betas)  # batch (b, 1)
            body_quats_pred = SO3.from_matrix(x_0_pred.body_rotmats).wxyz  # (b,t,21,4)
            posed = shaped.with_pose_decomposed(
                T_world_root=train_batch.T_world_root,
                body_quats=body_quats_pred,
            )
            wrist_w = posed.Ts_world_joint[:, :, [19, 20], 4:7]  # (b,t,2,3)
            wrist_pred_cpf = (
                SE3(train_batch.T_world_cpf[:, :, None, :]).inverse() @ wrist_w
            )  # (b,t,2,3)

            w = wrist_cond_valid.to(wrist_pred_cpf.dtype)[..., None]  # (b,t,2,1)
            se = ((wrist_pred_cpf - train_batch.wrist_pos_wrt_cpf) ** 2).clamp(
                max=self.config.wrist_cond_pos_loss_clamp
            ) * w
            per_bt = se.sum(dim=(-1, -2))  # (b,t)
            denom = (
                (w.sum(dim=(-1, -2)) * train_batch.mask).sum().clamp_min(256.0)
            )
            # `weight_t` down-weights high-noise timesteps -- but those are exactly
            # where the model has only the conditioning to go on, so for the wrist
            # objective we weight uniformly across t (see wrist_cond_pos_loss_uniform_t).
            wt = (
                torch.ones_like(weight_t)
                if self.config.wrist_cond_pos_loss_uniform_t
                else weight_t
            )
            wrist_pos_loss = (
                per_bt * train_batch.mask * wt[:, None]
            ).sum() / denom
            log_outputs["loss_term/wrist_cond_pos"] = wrist_pos_loss
            loss = loss + self.config.wrist_cond_pos_loss_weight * wrist_pos_loss

        log_outputs["train_loss"] = loss

        return loss, log_outputs
