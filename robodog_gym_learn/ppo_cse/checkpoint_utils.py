"""Utilities for deliberately migrating CSE checkpoints between observation contracts."""

from typing import Dict, List, Mapping, Tuple

import torch


_EXPANDABLE_INPUTS = {
    "adaptation_module.0.weight": ("estimator", 0),
    "actor_body.0.weight": ("policy", "privileged"),
    "critic_body.0.weight": ("policy", "privileged"),
}


def _expand_stacked_history_weight(
    source: torch.Tensor,
    target: torch.Tensor,
    history_length: int,
    tail_width: int,
) -> Tuple[torch.Tensor, int, int]:
    """Copy old per-frame inputs and leave newly appended fields at zero.

    Policy observations are flattened as ``[frame_0, ..., frame_n, latent]``.
    New obstacle fields are appended to every frame, so a plain prefix copy
    would move the old latent columns into observation columns.  This helper
    remaps every history frame and copies the latent/privileged tail separately.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Only 2-D linear-layer weights can be expanded")
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"Output width changed from {source.shape[0]} to {target.shape[0]}")
    if history_length <= 0:
        raise ValueError(f"history_length must be positive, got {history_length}")

    old_history_width = source.shape[1] - tail_width
    new_history_width = target.shape[1] - tail_width
    if old_history_width <= 0 or new_history_width <= 0:
        raise ValueError("Invalid history/tail split for checkpoint expansion")
    if old_history_width % history_length or new_history_width % history_length:
        raise ValueError(
            "Observation width is not divisible by the configured history length: "
            f"old={old_history_width}, new={new_history_width}, history={history_length}")

    old_step_width = old_history_width // history_length
    new_step_width = new_history_width // history_length
    if new_step_width <= old_step_width:
        raise ValueError(
            "Observation migration only permits appended inputs: "
            f"old step={old_step_width}, new step={new_step_width}")

    expanded = torch.zeros_like(target)
    source = source.to(device=target.device, dtype=target.dtype)
    for frame in range(history_length):
        old_start = frame * old_step_width
        new_start = frame * new_step_width
        expanded[:, new_start:new_start + old_step_width] = \
            source[:, old_start:old_start + old_step_width]

    if tail_width:
        expanded[:, -tail_width:] = source[:, -tail_width:]
    return expanded, old_step_width, new_step_width


def migrate_observation_expansion(
    model,
    source_state: Mapping[str, torch.Tensor],
    policy_history_length: int,
    estimator_history_length: int,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Return a strict-loadable state dict with zero-initialized new inputs.

    Only the three known CSE input matrices may change shape.  Every other key
    must match exactly, which prevents this opt-in migration from hiding an
    unrelated architecture mismatch.
    """
    target_state = model.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint keys differ; missing={missing}, unexpected={unexpected}")

    migrated: Dict[str, torch.Tensor] = {}
    messages: List[str] = []
    for key, target in target_state.items():
        source = source_state[key]
        if source.shape == target.shape:
            migrated[key] = source.to(device=target.device, dtype=target.dtype)
            continue

        if key not in _EXPANDABLE_INPUTS:
            raise RuntimeError(
                f"Unsupported checkpoint shape change for {key}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}")

        history_kind, tail_kind = _EXPANDABLE_INPUTS[key]
        history_length = (
            policy_history_length if history_kind == "policy"
            else estimator_history_length
        )
        tail_width = model.num_privileged_obs if tail_kind == "privileged" else 0
        migrated[key], old_step, new_step = _expand_stacked_history_weight(
            source, target, history_length, tail_width)
        messages.append(
            f"{key}: per-frame input {old_step} -> {new_step}; "
            f"initialized {new_step - old_step} new fields to zero")

    return migrated, messages
