# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from einops import einsum
import torch
import torch.nn.functional as F


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)
    weight_sums = weights.sum(dim=1, keepdim=True)
    valid = (
        torch.isfinite(true_coords).all(dim=(1, 2))
        & torch.isfinite(pred_coords).all(dim=(1, 2))
        & torch.isfinite(weight_sums).all(dim=(1, 2))
        & (weight_sums.squeeze(-1).squeeze(-1) > 0)
    )

    if not valid.all():
        print(
            "Warning: Encountered invalid alignment inputs; returning "
            + "unmodified coordinates for those samples."
        )

    if not valid.any():
        aligned_coords = pred_coords.clone()
        aligned_coords.detach_()
        return aligned_coords

    aligned_coords = pred_coords.clone()
    true_coords = true_coords[valid]
    pred_coords = pred_coords[valid]
    weights = weights[valid]
    weight_sums = weight_sums[valid]

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weight_sums
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weight_sums

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)
    rot_valid = torch.isfinite(rot_matrix).all(dim=(1, 2))

    if not rot_valid.all():
        print(
            "Warning: Encountered non-finite rotation matrices; returning "
            + "unmodified coordinates for those samples."
        )

    if not rot_valid.any():
        aligned_coords.detach_()
        return aligned_coords

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(true_coords.shape[0], 1, 1)
    det_values = torch.ones(
        true_coords.shape[0], dtype=cov_matrix_32.dtype, device=cov_matrix.device
    )
    det_values[rot_valid] = torch.det(rot_matrix[rot_valid].cpu()).to(
        device=cov_matrix.device
    )
    F[:, -1, -1] = det_values
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords_valid = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords[valid] = aligned_coords_valid
    aligned_coords.detach_()

    return aligned_coords


def smooth_lddt_loss(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask,
    pair_mask=None,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):
    """Compute weighted alignment.

    Parameters
    ----------
    pred_coords: torch.Tensor
        The predicted atom coordinates
    true_coords: torch.Tensor
        The ground truth atom coordinates
    is_nucleotide: torch.Tensor
        The weights for alignment
    coords_mask: torch.Tensor
        The atoms mask
    pair_mask: torch.Tensor, optional
        Optional pairwise atom mask selecting which atom pairs contribute
    nucleic_acid_cutoff: float
        The nucleic acid cutoff
    other_cutoff: float
        The non nucleic acid cutoff
    multiplicity: int
        The multiplicity
    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    B, N, _ = true_coords.shape
    true_dists = torch.cdist(true_coords, true_coords)
    is_nucleotide = is_nucleotide.repeat_interleave(multiplicity, 0)

    coords_mask = coords_mask.repeat_interleave(multiplicity, 0)
    is_nucleotide_pair = is_nucleotide.unsqueeze(-1).expand(
        -1, -1, is_nucleotide.shape[-1]
    )

    mask = (
        is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        + (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
    )
    mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
    mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))
    if pair_mask is not None:
        pair_mask = pair_mask.repeat_interleave(multiplicity, 0)
        mask = mask * pair_mask

    # Compute distances between all pairs of atoms
    pred_dists = torch.cdist(pred_coords, pred_coords)
    dist_diff = torch.abs(true_dists - pred_dists)

    # Compute epsilon values
    eps = (
        (
            (
                F.sigmoid(0.5 - dist_diff)
                + F.sigmoid(1.0 - dist_diff)
                + F.sigmoid(2.0 - dist_diff)
                + F.sigmoid(4.0 - dist_diff)
            )
            / 4.0
        )
        .view(multiplicity, B // multiplicity, N, N)
        .mean(dim=0)
    )

    # Calculate masked averaging
    eps = eps.repeat_interleave(multiplicity, 0)
    num = (eps * mask).sum(dim=(-1, -2))
    den = mask.sum(dim=(-1, -2))
    lddt = num / den.clamp(min=1)

    if pair_mask is not None:
        valid = den > 0
        if not torch.any(valid):
            return pred_coords.sum() * 0.0
        lddt = lddt[valid]

    return 1.0 - lddt.mean()
