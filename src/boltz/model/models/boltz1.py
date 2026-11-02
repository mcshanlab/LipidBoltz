import gc
import random
from typing import Any, Optional

import torch
import torch._dynamo
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor, nn
from torchmetrics import MeanMetric

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.feature.symmetry import (
    minimum_lddt_symmetry_coords,
    minimum_symmetry_coords,
)
from boltz.model.loss.confidence import confidence_loss
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
    weighted_minimum_rmsd,
)
from boltz.model.loss.diffusion import weighted_rigid_align
from boltz.model.modules.confidence import ConfidenceModule
from boltz.model.modules.diffusion import AtomDiffusion
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import ExponentialMovingAverage
from boltz.model.optim.scheduler import AlphaFoldLRScheduler


class Boltz1(LightningModule):
    """Boltz1 model."""

    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        steering_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
        validation_metric_types: Optional[list[str]] = None,
        validation_single_metric_types: Optional[list[str]] = None,
        small_validation_args: Optional[dict[str, Any]] = None,
        trainable_modules: str = "full",
        trainable_module_prefixes: Optional[list[str]] = None,
        print_trainable_parameter_summary: bool = True,
        train_norm_every_n_optimizer_steps: int = 1,
    ) -> None:
        super().__init__()

        if (
            isinstance(train_norm_every_n_optimizer_steps, bool)
            or not isinstance(train_norm_every_n_optimizer_steps, int)
            or train_norm_every_n_optimizer_steps < 0
        ):
            msg = "train_norm_every_n_optimizer_steps must be a non-negative integer"
            raise ValueError(msg)

        self.save_hyperparameters()

        self.all_validation_metric_types = const.out_types + ["pocket_ligand_protein"]
        if validation_metric_types is None:
            self.validation_metric_types = self.all_validation_metric_types
        else:
            self.validation_metric_types = list(validation_metric_types)
            invalid = set(self.validation_metric_types) - set(
                self.all_validation_metric_types
            )
            if invalid:
                msg = f"Unknown validation metric types: {sorted(invalid)}"
                raise ValueError(msg)

        self.all_validation_single_metric_types = const.out_single_types
        if validation_single_metric_types is None:
            self.validation_single_metric_types = self.all_validation_single_metric_types
        else:
            self.validation_single_metric_types = list(validation_single_metric_types)
            invalid = set(self.validation_single_metric_types) - set(
                self.all_validation_single_metric_types
            )
            if invalid:
                msg = f"Unknown validation single metric types: {sorted(invalid)}"
                raise ValueError(msg)
        self.validation_log_prefix = "val"

        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        if confidence_prediction:
            self.top1_lddt = nn.ModuleDict()
            self.iplddt_top1_lddt = nn.ModuleDict()
            self.ipde_top1_lddt = nn.ModuleDict()
            self.pde_top1_lddt = nn.ModuleDict()
            self.ptm_top1_lddt = nn.ModuleDict()
            self.iptm_top1_lddt = nn.ModuleDict()
            self.ligand_iptm_top1_lddt = nn.ModuleDict()
            self.protein_iptm_top1_lddt = nn.ModuleDict()
            self.avg_lddt = nn.ModuleDict()
            self.plddt_mae = nn.ModuleDict()
            self.pde_mae = nn.ModuleDict()
            self.pae_mae = nn.ModuleDict()
        for m in self.all_validation_metric_types:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()
        for m in self.all_validation_single_metric_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()
        self.top1_confidence_ligand_rmsd = MeanMetric()
        self.best_ligand_rmsd = MeanMetric()
        self.top1_confidence_ligand_successrate = MeanMetric()
        self.ligand_successrate_best = MeanMetric()

        self.train_confidence_loss_logger = MeanMetric()
        self.train_confidence_loss_dict_logger = nn.ModuleDict()
        for m in [
            "plddt_loss",
            "resolved_loss",
            "pde_loss",
            "pae_loss",
        ]:
            self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.ema = None
        self.use_ema = ema
        self.ema_decay = ema_decay
        self._ema_eval_active = False

        self.training_args = training_args
        self.validation_args = validation_args
        self.small_validation_args = small_validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.steering_args = steering_args
        self.train_norm_every_n_optimizer_steps = (
            train_norm_every_n_optimizer_steps
        )

        self.use_kernels = use_kernels

        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.is_pairformer_compiled = False

        # Input projections
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "no_atom_encoder": no_atom_encoder,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        self.rel_pos = RelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.no_msa = no_msa
        if not no_msa:
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            # Big models hit the default cache limit (8)
            self.is_pairformer_compiled = True
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        # Output modules
        use_accumulate_token_repr = (
            confidence_prediction
            and "use_s_diffusion" in confidence_model_args
            and confidence_model_args["use_s_diffusion"]
        )
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_z": token_z,
                "token_s": token_s,
                "atom_z": atom_z,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "atom_feature_dim": atom_feature_dim,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr=use_accumulate_token_repr,
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(token_z, num_bins)
        self.confidence_prediction = confidence_prediction
        self.alpha_pae = alpha_pae

        self.structure_prediction_training = structure_prediction_training
        self.confidence_imitate_trunk = confidence_imitate_trunk
        if self.confidence_prediction:
            if self.confidence_imitate_trunk:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    imitate_trunk=True,
                    pairformer_args=pairformer_args,
                    full_embedder_args=full_embedder_args,
                    msa_args=msa_args,
                    **confidence_model_args,
                )
            else:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    **confidence_model_args,
                )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if name.split(".")[0] != "confidence_module":
                    param.requires_grad = False
        self._apply_trainable_module_config(
            trainable_modules=trainable_modules,
            trainable_module_prefixes=trainable_module_prefixes,
            print_summary=print_trainable_parameter_summary,
        )

    @staticmethod
    def _normalize_parameter_name(name: str) -> str:
        """Remove wrappers that can appear in compiled module parameter names."""
        return name.replace("._orig_mod", "")

    @staticmethod
    def _matches_prefix(name: str, prefix: str) -> bool:
        """Return whether a parameter name belongs to a requested module prefix."""
        return name == prefix or name.startswith(f"{prefix}.")

    def _apply_trainable_module_config(
        self,
        trainable_modules: str,
        trainable_module_prefixes: Optional[list[str]],
        print_summary: bool,
    ) -> None:
        """Optionally freeze the model and train only selected module prefixes."""
        if trainable_modules == "full":
            return

        if trainable_modules != "selected":
            msg = (
                "trainable_modules must be 'full' or 'selected', "
                f"got {trainable_modules!r}"
            )
            raise ValueError(msg)

        if not trainable_module_prefixes:
            msg = (
                "trainable_module_prefixes must contain at least one module prefix "
                "when trainable_modules is 'selected'."
            )
            raise ValueError(msg)

        prefixes = [prefix.rstrip(".") for prefix in trainable_module_prefixes]
        matched_parameter_counts = {prefix: 0 for prefix in prefixes}
        matched_tensor_counts = {prefix: 0 for prefix in prefixes}

        for name, param in self.named_parameters():
            normalized_name = self._normalize_parameter_name(name)
            matched_prefixes = [
                prefix
                for prefix in prefixes
                if self._matches_prefix(normalized_name, prefix)
            ]
            param.requires_grad = bool(matched_prefixes)
            for prefix in matched_prefixes:
                matched_parameter_counts[prefix] += param.numel()
                matched_tensor_counts[prefix] += 1

        unmatched_prefixes = [
            prefix for prefix, count in matched_tensor_counts.items() if count == 0
        ]
        if unmatched_prefixes:
            msg = (
                "The following trainable_module_prefixes did not match any "
                f"parameters: {unmatched_prefixes}"
            )
            raise ValueError(msg)

        if print_summary:
            self._print_trainable_parameter_summary(
                matched_parameter_counts=matched_parameter_counts,
                matched_tensor_counts=matched_tensor_counts,
            )

    def _print_trainable_parameter_summary(
        self,
        matched_parameter_counts: dict[str, int],
        matched_tensor_counts: dict[str, int],
    ) -> None:
        """Print a concise summary of which parameters will be optimized."""
        total_params = 0
        trainable_params = 0

        for param in self.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count

        percent = 100 * trainable_params / total_params if total_params else 0
        print("=" * 80)
        print("Trainable parameter check")
        print(
            f"Trainable params: {trainable_params:,} / "
            f"{total_params:,} ({percent:.2f}%)"
        )
        print("Trainable module prefixes:")
        for prefix, param_count in matched_parameter_counts.items():
            tensor_count = matched_tensor_counts[prefix]
            print(f"  {prefix}: {param_count:,} params across {tensor_count:,} tensors")
        print("=" * 80)

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0  # noqa: PLR2004
        ):
            self.use_kernels = False

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        dict_out = {}

        # Compute input embeddings
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Fixes an issue with unused parameters in autocast
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()

                    # Apply recycling
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # Compute pairwise stack
                    if not self.no_msa:
                        z = z + self.msa_module(
                            z, s_inputs, feats, use_kernels=self.use_kernels
                        )

                    # Revert to uncompiled version for validation
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    s, z = pairformer_module(
                        s,
                        z,
                        mask=mask,
                        pair_mask=pair_mask,
                        use_kernels=self.use_kernels,
                    )

            pdistogram = self.distogram_module(z)
            dict_out = {
                "pdistogram": pdistogram,
                "s": s,
                "z": z,
            }

        # Compute structure module
        if self.training and self.structure_prediction_training:
            dict_out.update(
                self.structure_module(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    multiplicity=multiplicity_diffusion_train,
                )
            )

        if (not self.training) or self.confidence_prediction:
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"],
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    train_accumulate_token_repr=self.training,
                    steering_args=self.steering_args,
                )
            )

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    s_diffusion=(
                        dict_out["diff_token_repr"]
                        if self.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )
        if self.confidence_prediction and self.confidence_module.use_s_diffusion:
            dict_out.pop("diff_token_repr", None)
        return dict_out

    def get_true_coordinates(
        self,
        batch,
        out,
        diffusion_samples,
        symmetry_correction,
        lddt_minimization=True,
        compute_ligand_rmsd=False,
    ):
        if symmetry_correction:
            min_coords_routine = (
                minimum_lddt_symmetry_coords
                if lddt_minimization
                else minimum_symmetry_coords
            )
            true_coords = []
            true_coords_resolved_mask = []
            rmsds, best_rmsds = [], []
            ligand_rmsds, best_ligand_rmsds = [], []
            for idx in range(batch["token_index"].shape[0]):
                best_rmsd = float("inf")
                best_ligand_rmsd = None
                for rep in range(diffusion_samples):
                    i = idx * diffusion_samples + rep
                    best_true_coords, rmsd, best_true_coords_resolved_mask = (
                        min_coords_routine(
                            coords=out["sample_atom_coords"][i : i + 1],
                            feats=batch,
                            index_batch=idx,
                            nucleotide_weight=self.nucleotide_rmsd_weight,
                            ligand_weight=self.ligand_rmsd_weight,
                        )
                    )
                    if compute_ligand_rmsd:
                        ligand_rmsd = self.protein_aligned_ligand_rmsd_single(
                            pred_atom_coords=out["sample_atom_coords"][i : i + 1],
                            atom_coords=best_true_coords,
                            atom_mask=best_true_coords_resolved_mask,
                            atom_to_token=batch["atom_to_token"][idx : idx + 1],
                            mol_type=batch["mol_type"][idx : idx + 1],
                        )[0]
                    rmsds.append(rmsd)
                    if compute_ligand_rmsd:
                        ligand_rmsds.append(ligand_rmsd.squeeze(0))
                    true_coords.append(best_true_coords)
                    true_coords_resolved_mask.append(best_true_coords_resolved_mask)
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                    if compute_ligand_rmsd and torch.isfinite(ligand_rmsd).all():
                        ligand_rmsd_value = ligand_rmsd.squeeze(0)
                        if (
                            best_ligand_rmsd is None
                            or ligand_rmsd_value < best_ligand_rmsd
                        ):
                            best_ligand_rmsd = ligand_rmsd_value
                best_rmsds.append(best_rmsd)
                if compute_ligand_rmsd:
                    if best_ligand_rmsd is None:
                        best_ligand_rmsd = out["sample_atom_coords"].new_tensor(
                            float("nan")
                        )
                    best_ligand_rmsds.append(best_ligand_rmsd)
            true_coords = torch.cat(true_coords, dim=0)
            true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
        else:
            true_coords = (
                batch["coords"].squeeze(1).repeat_interleave(diffusion_samples, 0)
            )

            true_coords_resolved_mask = batch["atom_resolved_mask"].repeat_interleave(
                diffusion_samples, 0
            )
            rmsds, best_rmsds = weighted_minimum_rmsd(
                out["sample_atom_coords"],
                batch,
                multiplicity=diffusion_samples,
                nucleotide_weight=self.nucleotide_rmsd_weight,
                ligand_weight=self.ligand_rmsd_weight,
            )
            ligand_rmsds, best_ligand_rmsds = [], []
            if compute_ligand_rmsd:
                ligand_rmsds, best_ligand_rmsds = self.protein_aligned_ligand_rmsd(
                    out["sample_atom_coords"],
                    batch,
                    multiplicity=diffusion_samples,
                )

        return (
            true_coords,
            rmsds,
            best_rmsds,
            ligand_rmsds,
            best_ligand_rmsds,
            true_coords_resolved_mask,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # Sample recycling steps
        recycling_steps = random.randint(0, self.training_args.recycling_steps)
        max_parallel_samples = self.training_args.get(
            "max_parallel_samples", self.training_args.diffusion_samples
        )

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=self.training_args.sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
            diffusion_samples=self.training_args.diffusion_samples,
            max_parallel_samples=max_parallel_samples,
        )

        # Compute losses
        if self.structure_prediction_training:
            disto_loss, _ = distogram_loss(
                out,
                batch,
            )
            try:
                diffusion_loss_dict = self.structure_module.compute_loss(
                    batch,
                    out,
                    multiplicity=self.training_args.diffusion_multiplicity,
                    **self.diffusion_loss_args,
                )
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                return None

        else:
            disto_loss = 0.0
            diffusion_loss_dict = {"loss": 0.0, "loss_breakdown": {}}

        if self.confidence_prediction:
            # confidence model symmetry correction
            (
                true_coords,
                _,
                _,
                _,
                _,
                true_coords_resolved_mask,
            ) = self.get_true_coordinates(
                batch,
                out,
                diffusion_samples=self.training_args.diffusion_samples,
                symmetry_correction=self.training_args.symmetry_correction,
            )

            confidence_loss_dict = confidence_loss(
                out,
                batch,
                true_coords,
                true_coords_resolved_mask,
                alpha_pae=self.alpha_pae,
                multiplicity=self.training_args.diffusion_samples,
            )
        else:
            confidence_loss_dict = {
                "loss": torch.tensor(0.0).to(batch["token_index"].device),
                "loss_breakdown": {},
            }

        # Aggregate losses
        loss = (
            self.training_args.confidence_loss_weight * confidence_loss_dict["loss"]
            + self.training_args.diffusion_loss_weight * diffusion_loss_dict["loss"]
            + self.training_args.distogram_loss_weight * disto_loss
        )
        # Log raw per-batch losses and stable epoch means. Supplying batch_size
        # makes Lightning weight the epoch mean correctly if batch sizes vary.
        batch_size = batch["token_index"].shape[0]
        self._log_training_loss("distogram_loss", disto_loss, batch_size)
        self._log_training_loss(
            "diffusion_loss",
            diffusion_loss_dict["loss"],
            batch_size,
        )
        for k, v in diffusion_loss_dict["loss_breakdown"].items():
            self._log_training_loss(k, v, batch_size)

        if self.confidence_prediction:
            self.train_confidence_loss_logger.update(
                confidence_loss_dict["loss"].detach()
            )

            for k in self.train_confidence_loss_dict_logger.keys():
                self.train_confidence_loss_dict_logger[k].update(
                    confidence_loss_dict["loss_breakdown"][k].detach()
                    if torch.is_tensor(confidence_loss_dict["loss_breakdown"][k])
                    else confidence_loss_dict["loss_breakdown"][k]
                )
        self._log_training_loss("loss", loss, batch_size)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)
        return loss

    @staticmethod
    def _training_log_name(name: str) -> str:
        namespace = "train_norm" if "norm" in name else "train"
        return f"{namespace}/{name}"

    def _log_training_loss(
        self,
        name: str,
        value: Tensor,
        batch_size: int,
    ) -> None:
        """Log a raw training loss and its mean over the complete epoch."""
        self.log(
            self._training_log_name(name),
            value,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        self.log(
            f"train_epoch/{name}",
            value,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def training_log(self):
        self.log(
            self._training_log_name("grad_norm"),
            self.gradient_norm(self),
            prog_bar=False,
        )
        self.log(
            self._training_log_name("param_norm"),
            self.parameter_norm(self),
            prog_bar=False,
        )

        self.log(
            self._training_log_name("grad_norm_msa_module"),
            self.gradient_norm(self.msa_module),
            prog_bar=False,
        )
        self.log(
            self._training_log_name("param_norm_msa_module"),
            self.parameter_norm(self.msa_module),
            prog_bar=False,
        )

        self.log(
            self._training_log_name("grad_norm_pairformer_module"),
            self.gradient_norm(self.pairformer_module),
            prog_bar=False,
        )
        self.log(
            self._training_log_name("param_norm_pairformer_module"),
            self.parameter_norm(self.pairformer_module),
            prog_bar=False,
        )

        self.log(
            self._training_log_name("grad_norm_structure_module"),
            self.gradient_norm(self.structure_module),
            prog_bar=False,
        )
        self.log(
            self._training_log_name("param_norm_structure_module"),
            self.parameter_norm(self.structure_module),
            prog_bar=False,
        )

        if self.confidence_prediction:
            self.log(
                self._training_log_name("grad_norm_confidence_module"),
                self.gradient_norm(self.confidence_module),
                prog_bar=False,
            )
            self.log(
                self._training_log_name("param_norm_confidence_module"),
                self.parameter_norm(self.confidence_module),
                prog_bar=False,
            )

    def on_before_optimizer_step(self, optimizer) -> None:
        """Calculate and log norms at the configured optimizer-step cadence."""
        del optimizer
        frequency = self.train_norm_every_n_optimizer_steps
        optimizer_step = self.global_step + 1
        if frequency > 0 and optimizer_step % frequency == 0:
            self.training_log()

    def on_train_epoch_end(self):
        self.log(
            "train/confidence_loss",
            self.train_confidence_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        for k, v in self.train_confidence_loss_dict_logger.items():
            self.log(
                self._training_log_name(k),
                v,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    def gradient_norm(self, module) -> Tensor:
        # Only compute over parameters that are being trained
        gradients = [
            p.grad
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        return torch.nn.utils.get_total_norm(gradients, norm_type=2.0, foreach=True)

    def parameter_norm(self, module) -> Tensor:
        # Only compute over parameters that are being trained
        parameters = [p for p in module.parameters() if p.requires_grad]
        return torch.nn.utils.get_total_norm(parameters, norm_type=2.0, foreach=True)

    def protein_aligned_ligand_rmsd(
        self,
        pred_atom_coords,
        feats,
        multiplicity=1,
    ):
        """Compute ligand RMSD after protein-only alignment for a batch."""
        atom_coords = feats["coords"].repeat_interleave(multiplicity, 0)[:, 0]
        atom_mask = feats["atom_resolved_mask"].repeat_interleave(multiplicity, 0)

        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["mol_type"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
            .repeat_interleave(multiplicity, 0)
        )

        protein_mask = torch.eq(atom_type, const.chain_type_ids["PROTEIN"]).float()
        ligand_mask = torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()

        align_mask = atom_mask * protein_mask
        score_mask = atom_mask * ligand_mask
        align_weights = atom_coords.new_ones(atom_coords.shape[:2])

        align_count = torch.sum(align_mask, dim=-1)
        score_count = torch.sum(score_mask, dim=-1)

        with torch.no_grad():
            aligned_true_coords = weighted_rigid_align(
                atom_coords,
                pred_atom_coords,
                align_weights,
                mask=align_mask,
            )

        squared_error = ((pred_atom_coords - aligned_true_coords) ** 2).sum(dim=-1)
        ligand_rmsd = torch.sqrt(
            torch.sum(squared_error * score_mask, dim=-1)
            / score_count.clamp_min(1e-7)
        )
        valid = (align_count > 0) & (score_count > 0)
        ligand_rmsd = torch.where(
            valid,
            ligand_rmsd,
            torch.full_like(ligand_rmsd, float("nan")),
        )

        ligand_rmsd_by_sample = ligand_rmsd.reshape(-1, multiplicity)
        finite = torch.isfinite(ligand_rmsd_by_sample)
        inf = torch.full_like(ligand_rmsd_by_sample, float("inf"))
        best_ligand_rmsd = torch.where(finite, ligand_rmsd_by_sample, inf).min(
            dim=1
        ).values
        best_ligand_rmsd = torch.where(
            finite.any(dim=1),
            best_ligand_rmsd,
            torch.full_like(best_ligand_rmsd, float("nan")),
        )

        return ligand_rmsd, best_ligand_rmsd

    def protein_aligned_ligand_rmsd_single(
        self,
        pred_atom_coords,
        atom_coords,
        atom_mask,
        atom_to_token,
        mol_type,
    ):
        """Compute ligand RMSD for one example after protein-only alignment."""
        atom_type = (
            torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )

        protein_mask = torch.eq(atom_type, const.chain_type_ids["PROTEIN"]).float()
        ligand_mask = torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()

        align_mask = atom_mask * protein_mask
        score_mask = atom_mask * ligand_mask
        align_weights = atom_coords.new_ones(atom_coords.shape[:2])

        align_count = torch.sum(align_mask, dim=-1)
        score_count = torch.sum(score_mask, dim=-1)

        with torch.no_grad():
            aligned_true_coords = weighted_rigid_align(
                atom_coords,
                pred_atom_coords,
                align_weights,
                mask=align_mask,
            )

        squared_error = ((pred_atom_coords - aligned_true_coords) ** 2).sum(dim=-1)
        ligand_rmsd = torch.sqrt(
            torch.sum(squared_error * score_mask, dim=-1)
            / score_count.clamp_min(1e-7)
        )
        valid = (align_count > 0) & (score_count > 0)
        ligand_rmsd = torch.where(
            valid,
            ligand_rmsd,
            torch.full_like(ligand_rmsd, float("nan")),
        )

        return ligand_rmsd, aligned_true_coords, align_weights

    @staticmethod
    def _update_metric_ignore_nan(metric, values) -> None:
        if isinstance(values, list):
            if not values:
                return
            values = torch.stack([torch.as_tensor(v) for v in values])
        else:
            values = torch.as_tensor(values)
        values = values.reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() > 0:
            metric.update(finite)

    @staticmethod
    def _update_success_rate_ignore_nan(metric, values, threshold=2.0) -> None:
        if isinstance(values, list):
            if not values:
                return
            values = torch.stack([torch.as_tensor(v) for v in values])
        else:
            values = torch.as_tensor(values)
        values = values.reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() > 0:
            metric.update((finite < threshold).float())

    def _validation_log_name(self, name: str) -> str:
        prefix = getattr(self, "validation_log_prefix", "val").strip("/")
        return f"{prefix}/{name}"

    def _validation_confidence_log_name(self, name: str) -> str:
        prefix = getattr(self, "validation_log_prefix", "val").strip("/")
        return f"{prefix}_confidence/{name}"

    def _update_ligand_rmsd_metrics(
        self,
        out: dict[str, Tensor],
        ligand_rmsds: list,
        best_ligand_rmsds: list,
        n_samples: int,
    ) -> None:
        if self.confidence_prediction:
            ligand_rmsds_by_sample = torch.stack(
                [torch.as_tensor(value) for value in ligand_rmsds]
            ).reshape(-1, n_samples)
            confidence_score = (
                4 * out["complex_plddt"]
                + (
                    out["iptm"]
                    if not torch.allclose(
                        out["iptm"], torch.zeros_like(out["iptm"])
                    )
                    else out["ptm"]
                )
            ) / 5
            top1_confidence_idx = confidence_score.reshape(-1, n_samples).argmax(
                dim=1
            )
            target_idx = torch.arange(
                ligand_rmsds_by_sample.shape[0],
                device=ligand_rmsds_by_sample.device,
            )
            top1_confidence_ligand_rmsds = ligand_rmsds_by_sample[
                target_idx, top1_confidence_idx
            ]
            self._update_metric_ignore_nan(
                self.top1_confidence_ligand_rmsd,
                top1_confidence_ligand_rmsds,
            )
            self._update_success_rate_ignore_nan(
                self.top1_confidence_ligand_successrate,
                top1_confidence_ligand_rmsds,
            )

        self._update_metric_ignore_nan(self.best_ligand_rmsd, best_ligand_rmsds)
        self._update_success_rate_ignore_nan(
            self.ligand_successrate_best,
            best_ligand_rmsds,
        )

    @staticmethod
    def _weighted_available_metric_mean(
        values: dict[str, float],
        available: dict[str, bool],
    ) -> float:
        active_weights = {
            name: weight
            for name, weight in const.out_types_weights.items()
            if available.get(name, False) and weight > 0
        }
        if not active_weights:
            return 0.0
        return sum(
            values[name] * weight for name, weight in active_weights.items()
        ) / sum(active_weights.values())

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        # Compute the forward pass
        validation_args = self._active_validation_args
        n_samples = validation_args.diffusion_samples
        max_parallel_samples = validation_args.get(
            "max_parallel_samples", n_samples
        )
        try:
            out = self(
                batch,
                recycling_steps=validation_args.recycling_steps,
                num_sampling_steps=validation_args.sampling_steps,
                diffusion_samples=n_samples,
                max_parallel_samples=max_parallel_samples,
                run_confidence_sequentially=(
                    validation_args.run_confidence_sequentially
                ),
            )

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        try:
            # Compute distogram LDDT
            boundaries = torch.linspace(2, 22.0, 63)
            lower = torch.tensor([1.0])
            upper = torch.tensor([22.0 + 5.0])
            exp_boundaries = torch.cat((lower, boundaries, upper))
            mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
                out["pdistogram"]
            )

            # Compute predicted dists
            preds = out["pdistogram"]
            pred_softmax = torch.softmax(preds, dim=-1)
            pred_softmax = pred_softmax.argmax(dim=-1)
            pred_softmax = torch.nn.functional.one_hot(
                pred_softmax, num_classes=preds.shape[-1]
            )
            pred_dist = (pred_softmax * mid_points).sum(dim=-1)
            true_center = batch["disto_center"]
            true_dists = torch.cdist(true_center, true_center)

            # Compute lddt's
            batch["token_disto_mask"] = batch["token_disto_mask"]
            disto_lddt_dict, disto_total_dict = factored_token_lddt_dist_loss(
                feats=batch,
                true_d=true_dists,
                pred_d=pred_dist,
            )

            (
                true_coords,
                rmsds,
                best_rmsds,
                ligand_rmsds,
                best_ligand_rmsds,
                true_coords_resolved_mask,
            ) = self.get_true_coordinates(
                batch=batch,
                out=out,
                diffusion_samples=n_samples,
                symmetry_correction=validation_args.symmetry_correction,
                compute_ligand_rmsd=True,
            )

            all_lddt_dict, all_total_dict = factored_lddt_loss(
                feats=batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords,
                pred_atom_coords=out["sample_atom_coords"],
                multiplicity=n_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e
        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        B = true_coords.shape[0] // n_samples
        if n_samples > 1:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict.keys():
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7
            best_complex_idx = complex_lddt.reshape(-1, n_samples).argmax(dim=1)
            for key in all_lddt_dict:
                best_idx = all_lddt_dict[key].reshape(-1, n_samples).argmax(dim=1)
                best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                best_total_dict[key] = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_complex_idx
                ]
                best_complex_total_dict[key] = all_total_dict[key].reshape(
                    -1, n_samples
                )[torch.arange(B), best_complex_idx]
        else:
            best_lddt_dict = all_lddt_dict
            best_total_dict = all_total_dict
            best_complex_lddt_dict = all_lddt_dict
            best_complex_total_dict = all_total_dict

        # Filtering based on confidence
        if self.confidence_prediction and n_samples > 1:
            # note: for now we don't have pae predictions so have to use pLDDT instead of pTM
            # also, while AF3 differentiates the best prediction per confidence type we are currently not doing it
            # consider this in the future as well as weighing the different pLLDT types before aggregation
            mae_plddt_dict, total_mae_plddt_dict = compute_plddt_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_lddt=out["plddt"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pde_dict, total_mae_pde_dict = compute_pde_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pde=out["pde"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pae_dict, total_mae_pae_dict = compute_pae_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pae=out["pae"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )

            plddt = out["complex_plddt"].reshape(-1, n_samples)
            top1_idx = plddt.argmax(dim=1)
            iplddt = out["complex_iplddt"].reshape(-1, n_samples)
            iplddt_top1_idx = iplddt.argmax(dim=1)
            pde = out["complex_pde"].reshape(-1, n_samples)
            pde_top1_idx = pde.argmin(dim=1)
            ipde = out["complex_ipde"].reshape(-1, n_samples)
            ipde_top1_idx = ipde.argmin(dim=1)
            ptm = out["ptm"].reshape(-1, n_samples)
            ptm_top1_idx = ptm.argmax(dim=1)
            iptm = out["iptm"].reshape(-1, n_samples)
            iptm_top1_idx = iptm.argmax(dim=1)
            ligand_iptm = out["ligand_iptm"].reshape(-1, n_samples)
            ligand_iptm_top1_idx = ligand_iptm.argmax(dim=1)
            protein_iptm = out["protein_iptm"].reshape(-1, n_samples)
            protein_iptm_top1_idx = protein_iptm.argmax(dim=1)

            for key in all_lddt_dict:
                top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                iplddt_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                iplddt_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                pde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                pde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                ipde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ipde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                ptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                ligand_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                ligand_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                protein_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]
                protein_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]

                self.top1_lddt[key].update(top1_lddt, top1_total)
                self.iplddt_top1_lddt[key].update(iplddt_top1_lddt, iplddt_top1_total)
                self.pde_top1_lddt[key].update(pde_top1_lddt, pde_top1_total)
                self.ipde_top1_lddt[key].update(ipde_top1_lddt, ipde_top1_total)
                self.ptm_top1_lddt[key].update(ptm_top1_lddt, ptm_top1_total)
                self.iptm_top1_lddt[key].update(iptm_top1_lddt, iptm_top1_total)
                self.ligand_iptm_top1_lddt[key].update(
                    ligand_iptm_top1_lddt, ligand_iptm_top1_total
                )
                self.protein_iptm_top1_lddt[key].update(
                    protein_iptm_top1_lddt, protein_iptm_top1_total
                )

                self.avg_lddt[key].update(all_lddt_dict[key], all_total_dict[key])
                self.pde_mae[key].update(mae_pde_dict[key], total_mae_pde_dict[key])
                self.pae_mae[key].update(mae_pae_dict[key], total_mae_pae_dict[key])

            for key in mae_plddt_dict:
                self.plddt_mae[key].update(
                    mae_plddt_dict[key], total_mae_plddt_dict[key]
                )

        for m in const.out_types:
            if (
                m not in best_lddt_dict
                or m not in best_total_dict
                or m not in disto_lddt_dict
                or m not in disto_total_dict
                or m not in best_complex_lddt_dict
                or m not in best_complex_total_dict
            ):
                continue
            if m == "ligand_protein":
                if torch.any(
                    batch["pocket_feature"][
                        :, :, const.pocket_contact_info["POCKET"]
                    ].bool()
                ):
                    self.lddt["pocket_ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["pocket_ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["pocket_ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
                else:
                    self.lddt["ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
            else:
                self.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
                self.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
                self.complex_lddt[m].update(
                    best_complex_lddt_dict[m], best_complex_total_dict[m]
                )
        self.rmsd.update(rmsds)
        self.best_rmsd.update(best_rmsds)
        self._update_ligand_rmsd_metrics(
            out,
            ligand_rmsds,
            best_ligand_rmsds,
            n_samples,
        )

    def on_validation_epoch_end(self):
        avg_lddt = {}
        avg_disto_lddt = {}
        avg_complex_lddt = {}
        available_lddt = {}
        available_disto_lddt = {}
        available_complex_lddt = {}
        if self.confidence_prediction:
            avg_top1_lddt = {}
            avg_iplddt_top1_lddt = {}
            avg_pde_top1_lddt = {}
            avg_ipde_top1_lddt = {}
            avg_ptm_top1_lddt = {}
            avg_iptm_top1_lddt = {}
            avg_ligand_iptm_top1_lddt = {}
            avg_protein_iptm_top1_lddt = {}
            available_top1_lddt = {}
            available_iplddt_top1_lddt = {}
            available_pde_top1_lddt = {}
            available_ipde_top1_lddt = {}
            available_ptm_top1_lddt = {}
            available_iptm_top1_lddt = {}

            avg_avg_lddt = {}
            available_avg_lddt = {}
            avg_mae_plddt = {}
            avg_mae_pde = {}
            avg_mae_pae = {}

        for m in self.all_validation_metric_types:
            avg_lddt[m] = self.lddt[m].compute()
            available_lddt[m] = bool(torch.isfinite(avg_lddt[m]))
            avg_lddt[m] = avg_lddt[m].item() if available_lddt[m] else 0.0
            self.lddt[m].reset()
            if m in self.validation_metric_types:
                self.log(
                    self._validation_log_name(f"lddt_{m}"),
                    avg_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )

            avg_disto_lddt[m] = self.disto_lddt[m].compute()
            available_disto_lddt[m] = bool(torch.isfinite(avg_disto_lddt[m]))
            avg_disto_lddt[m] = (
                avg_disto_lddt[m].item() if available_disto_lddt[m] else 0.0
            )
            self.disto_lddt[m].reset()
            if m in self.validation_metric_types:
                self.log(
                    self._validation_log_name(f"disto_lddt_{m}"),
                    avg_disto_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
            avg_complex_lddt[m] = self.complex_lddt[m].compute()
            available_complex_lddt[m] = bool(torch.isfinite(avg_complex_lddt[m]))
            avg_complex_lddt[m] = (
                avg_complex_lddt[m].item() if available_complex_lddt[m] else 0.0
            )
            self.complex_lddt[m].reset()
            if m in self.validation_metric_types:
                self.log(
                    self._validation_log_name(f"complex_lddt_{m}"),
                    avg_complex_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
            if self.confidence_prediction:
                avg_top1_lddt[m] = self.top1_lddt[m].compute()
                available_top1_lddt[m] = bool(torch.isfinite(avg_top1_lddt[m]))
                avg_top1_lddt[m] = (
                    avg_top1_lddt[m].item() if available_top1_lddt[m] else 0.0
                )
                self.top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"top1_lddt_{m}"),
                        avg_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_iplddt_top1_lddt[m] = self.iplddt_top1_lddt[m].compute()
                available_iplddt_top1_lddt[m] = bool(
                    torch.isfinite(avg_iplddt_top1_lddt[m])
                )
                avg_iplddt_top1_lddt[m] = (
                    avg_iplddt_top1_lddt[m].item()
                    if available_iplddt_top1_lddt[m]
                    else 0.0
                )
                self.iplddt_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(
                            f"iplddt_top1_lddt_{m}"
                        ),
                        avg_iplddt_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_pde_top1_lddt[m] = self.pde_top1_lddt[m].compute()
                available_pde_top1_lddt[m] = bool(torch.isfinite(avg_pde_top1_lddt[m]))
                avg_pde_top1_lddt[m] = (
                    avg_pde_top1_lddt[m].item() if available_pde_top1_lddt[m] else 0.0
                )
                self.pde_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"pde_top1_lddt_{m}"),
                        avg_pde_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_ipde_top1_lddt[m] = self.ipde_top1_lddt[m].compute()
                available_ipde_top1_lddt[m] = bool(
                    torch.isfinite(avg_ipde_top1_lddt[m])
                )
                avg_ipde_top1_lddt[m] = (
                    avg_ipde_top1_lddt[m].item() if available_ipde_top1_lddt[m] else 0.0
                )
                self.ipde_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"ipde_top1_lddt_{m}"),
                        avg_ipde_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_ptm_top1_lddt[m] = self.ptm_top1_lddt[m].compute()
                available_ptm_top1_lddt[m] = bool(torch.isfinite(avg_ptm_top1_lddt[m]))
                avg_ptm_top1_lddt[m] = (
                    avg_ptm_top1_lddt[m].item() if available_ptm_top1_lddt[m] else 0.0
                )
                self.ptm_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"ptm_top1_lddt_{m}"),
                        avg_ptm_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_iptm_top1_lddt[m] = self.iptm_top1_lddt[m].compute()
                available_iptm_top1_lddt[m] = bool(
                    torch.isfinite(avg_iptm_top1_lddt[m])
                )
                avg_iptm_top1_lddt[m] = (
                    avg_iptm_top1_lddt[m].item() if available_iptm_top1_lddt[m] else 0.0
                )
                self.iptm_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"iptm_top1_lddt_{m}"),
                        avg_iptm_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )

                avg_ligand_iptm_top1_lddt[m] = self.ligand_iptm_top1_lddt[m].compute()
                avg_ligand_iptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_ligand_iptm_top1_lddt[m])
                    else avg_ligand_iptm_top1_lddt[m].item()
                )
                self.ligand_iptm_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(
                            f"ligand_iptm_top1_lddt_{m}"
                        ),
                        avg_ligand_iptm_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )

                avg_protein_iptm_top1_lddt[m] = self.protein_iptm_top1_lddt[m].compute()
                avg_protein_iptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_protein_iptm_top1_lddt[m])
                    else avg_protein_iptm_top1_lddt[m].item()
                )
                self.protein_iptm_top1_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(
                            f"protein_iptm_top1_lddt_{m}"
                        ),
                        avg_protein_iptm_top1_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )

                avg_avg_lddt[m] = self.avg_lddt[m].compute()
                available_avg_lddt[m] = bool(torch.isfinite(avg_avg_lddt[m]))
                avg_avg_lddt[m] = (
                    avg_avg_lddt[m].item() if available_avg_lddt[m] else 0.0
                )
                self.avg_lddt[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"avg_lddt_{m}"),
                        avg_avg_lddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_mae_pde[m] = self.pde_mae[m].compute().item()
                self.pde_mae[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"MAE_pde_{m}"),
                        avg_mae_pde[m],
                        prog_bar=False,
                        sync_dist=True,
                    )
                avg_mae_pae[m] = self.pae_mae[m].compute().item()
                self.pae_mae[m].reset()
                if m in self.validation_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"MAE_pae_{m}"),
                        avg_mae_pae[m],
                        prog_bar=False,
                        sync_dist=True,
                    )

        for m in self.all_validation_single_metric_types:
            if self.confidence_prediction:
                avg_mae_plddt[m] = self.plddt_mae[m].compute().item()
                self.plddt_mae[m].reset()
                if m in self.validation_single_metric_types:
                    self.log(
                        self._validation_confidence_log_name(f"MAE_plddt_{m}"),
                        avg_mae_plddt[m],
                        prog_bar=False,
                        sync_dist=True,
                    )

        overall_disto_lddt = self._weighted_available_metric_mean(
            avg_disto_lddt,
            available_disto_lddt,
        )
        self.log(
            self._validation_log_name("disto_lddt"),
            overall_disto_lddt,
            prog_bar=True,
            sync_dist=True,
        )

        overall_lddt = self._weighted_available_metric_mean(
            avg_lddt,
            available_lddt,
        )
        self.log(
            self._validation_log_name("lddt"),
            overall_lddt,
            prog_bar=True,
            sync_dist=True,
        )

        overall_complex_lddt = self._weighted_available_metric_mean(
            avg_complex_lddt,
            available_complex_lddt,
        )
        self.log(
            self._validation_log_name("complex_lddt"),
            overall_complex_lddt,
            prog_bar=True,
            sync_dist=True,
        )

        if self.confidence_prediction:
            overall_top1_lddt = self._weighted_available_metric_mean(
                avg_top1_lddt,
                available_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("top1_lddt"),
                overall_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_iplddt_top1_lddt = self._weighted_available_metric_mean(
                avg_iplddt_top1_lddt,
                available_iplddt_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("iplddt_top1_lddt"),
                overall_iplddt_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_pde_top1_lddt = self._weighted_available_metric_mean(
                avg_pde_top1_lddt,
                available_pde_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("pde_top1_lddt"),
                overall_pde_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_ipde_top1_lddt = self._weighted_available_metric_mean(
                avg_ipde_top1_lddt,
                available_ipde_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("ipde_top1_lddt"),
                overall_ipde_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_ptm_top1_lddt = self._weighted_available_metric_mean(
                avg_ptm_top1_lddt,
                available_ptm_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("ptm_top1_lddt"),
                overall_ptm_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_iptm_top1_lddt = self._weighted_available_metric_mean(
                avg_iptm_top1_lddt,
                available_iptm_top1_lddt,
            )
            self.log(
                self._validation_confidence_log_name("iptm_top1_lddt"),
                overall_iptm_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_avg_lddt = self._weighted_available_metric_mean(
                avg_avg_lddt,
                available_avg_lddt,
            )
            self.log(
                self._validation_confidence_log_name("avg_lddt"),
                overall_avg_lddt,
                prog_bar=True,
                sync_dist=True,
            )

        self.log(
            self._validation_log_name("rmsd"),
            self.rmsd.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.rmsd.reset()

        self.log(
            self._validation_log_name("best_rmsd"),
            self.best_rmsd.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.best_rmsd.reset()

        if self.confidence_prediction:
            self.log(
                self._validation_log_name("top1_confidence_ligand_rmsd"),
                self.top1_confidence_ligand_rmsd.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.top1_confidence_ligand_rmsd.reset()

        self.log(
            self._validation_log_name("best_ligand_rmsd"),
            self.best_ligand_rmsd.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.best_ligand_rmsd.reset()

        if self.confidence_prediction:
            self.log(
                self._validation_log_name("top1_confidence_ligand_successrate"),
                self.top1_confidence_ligand_successrate.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.top1_confidence_ligand_successrate.reset()

        self.log(
            self._validation_log_name("ligand_successrate_best"),
            self.ligand_successrate_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.ligand_successrate_best.reset()

        # Restore the online parameters before validation-end callbacks save
        # checkpoints. The EMA parameters are stored separately in the checkpoint.
        self._restore_online_parameters()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                max_parallel_samples=self.predict_args["diffusion_samples"],
                run_confidence_sequentially=True,
            )
            pred_dict = {"exception": False}
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["coords"] = out["sample_atom_coords"]
            pred_dict["s"] = out["s"]
            pred_dict["z"] = out["z"]
            if self.predict_args.get("write_confidence_summary", True):
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5
                for key in [
                    "ptm",
                    "iptm",
                    "ligand_iptm",
                    "protein_iptm",
                    "pair_chains_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                    "complex_pde",
                    "complex_ipde",
                    "plddt",
                ]:
                    pred_dict[key] = out[key]
            if self.predict_args.get("write_full_pae", True):
                pred_dict["pae"] = out["pae"]
            if self.predict_args.get("write_full_pde", False):
                pred_dict["pde"] = out["pde"]
            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise

    def configure_optimizers(self):
        """Configure the optimizer."""

        if self.structure_prediction_training:
            parameters = [p for p in self.parameters() if p.requires_grad]
        else:
            parameters = [
                p for p in self.confidence_module.parameters() if p.requires_grad
            ] + [
                p
                for p in self.structure_module.out_token_feat_update.parameters()
                if p.requires_grad
            ]

        optimizer = torch.optim.Adam(
            parameters,
            betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
            eps=self.training_args.adam_eps,
            lr=self.training_args.base_lr,
        )
        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema and "ema" in checkpoint:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
            if self.ema.compatible(checkpoint["ema"]["shadow_params"]):
                self.ema.load_state_dict(checkpoint["ema"], device=torch.device("cpu"))
            else:
                self.ema = None
                print(
                    "Warning: EMA state not loaded due to incompatible model parameters."
                )

    def on_train_start(self):
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        elif self.use_ema:
            self.ema.to(self.device)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Any,
        optimizer_closure: Any,
    ) -> None:
        """Run one optimizer step and update EMA exactly once afterward."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.use_ema:
            self.ema.update(self.parameters())

    def prepare_eval(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        if self.use_ema:
            if self._ema_eval_active:
                return
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())
            self._ema_eval_active = True

    def _restore_online_parameters(self) -> None:
        """Restore parameters saved before an EMA evaluation swap."""
        if self.use_ema and self._ema_eval_active:
            self.ema.restore(self.parameters())
            self._ema_eval_active = False

    def on_validation_start(self):
        data_module = self.trainer.datamodule
        if (
            self.small_validation_args is not None
            and hasattr(data_module, "is_fast_validation")
            and data_module.is_fast_validation()
        ):
            self._active_validation_args = self.small_validation_args
            self.validation_log_prefix = "val_fast"
        else:
            self._active_validation_args = self.validation_args
            if self.trainer.state.fn == TrainerFn.FITTING:
                self.validation_log_prefix = "val"
        self.prepare_eval()

    def on_validation_end(self) -> None:
        """Restore online parameters if validation exited before epoch-end cleanup."""
        self._restore_online_parameters()

    def on_predict_start(self) -> None:
        self.prepare_eval()

    def on_predict_end(self) -> None:
        self._restore_online_parameters()

    def on_test_start(self) -> None:
        self.prepare_eval()

    def on_test_end(self) -> None:
        self._restore_online_parameters()
