#!/usr/bin/env python3
"""Validate the user-input H5 contract used before inference graph construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _attr(g, name, default=None):
    return _decode(g.attrs[name]) if name in g.attrs else default


def validate_sim(sim: h5py.Group) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    required_datasets = ["block_features_merged", "filler_blocks"]
    for name in required_datasets:
        if name not in sim:
            errors.append(f"missing dataset: {name}")

    if errors:
        return errors

    merged = np.asarray(sim["block_features_merged"][:])
    fillers = np.asarray(sim["filler_blocks"][:])

    if merged.ndim != 2 or merged.shape[1] < 15:
        errors.append(f"block_features_merged must be [N,15+], got {merged.shape}")
    elif merged.shape[0] == 0:
        errors.append("block_features_merged contains zero real blocks")
    elif not np.isfinite(merged).all():
        errors.append("block_features_merged contains NaN/Inf")

    if fillers.ndim != 2 or fillers.shape[1] != 7:
        errors.append(f"filler_blocks must be exact [F,7] for inference, got {fillers.shape}")
    elif not np.isfinite(fillers).all():
        errors.append("filler_blocks contains NaN/Inf")

    n_layers = int(round(float(_attr(sim, "n_layers", 0) or 0)))
    chip_w = float(_attr(sim, "chip_w_um", 0.0) or 0.0)
    chip_h = float(_attr(sim, "chip_h_um", 0.0) or 0.0)
    if n_layers <= 0:
        errors.append("n_layers must be > 0")
    if chip_w <= 0 or chip_h <= 0:
        errors.append("chip_w_um and chip_h_um must be > 0")

    if merged.ndim == 2 and merged.shape[1] >= 15 and merged.shape[0] > 0:
        w = merged[:, 9]
        h = merged[:, 10]
        thick = merged[:, 14]
        kappa = merged[:, 3]
        if np.any(w <= 0) or np.any(h <= 0):
            errors.append("real block width/height must be positive")
        if np.any(thick <= 0):
            errors.append("real block thickness must be positive")
        if np.any(kappa <= 0):
            errors.append("real block kappa must be positive")

    if fillers.ndim == 2 and fillers.shape[1] == 7 and fillers.shape[0] > 0:
        if np.any(fillers[:, 2] <= 0) or np.any(fillers[:, 3] <= 0):
            errors.append("filler width/height must be positive")
        if np.any(fillers[:, 6] <= 0):
            errors.append("filler thickness must be positive")

    inference_flag = int(round(float(_attr(sim, "is_inference_input_not_solver_output", 0) or 0)))
    if inference_flag != 1:
        errors.append("is_inference_input_not_solver_output must be 1")

    # Training/inference numeric compatibility gates. These do not transform data;
    # they only prevent a silently incompatible H5 from reaching the model.
    feature0_semantics = str(_attr(sim, "ml_feature0_numeric_semantics", ""))
    if feature0_semantics != "historical_v3_raw_power_W_despite_legacy_name":
        errors.append("feature-0 semantics marker does not match the historical V3 fine-tuning contract")

    global2_semantics = str(_attr(sim, "global_feature2_numeric_semantics", ""))
    if global2_semantics != "historical_v3_legacy_sum_feature0_times_area":
        errors.append("global-feature-2 semantics marker does not match the historical V3 fine-tuning contract")

    if merged.ndim == 2 and merged.shape[1] >= 15 and merged.shape[0] > 0:
        expected_legacy_total = float(np.sum(merged[:, 0] * merged[:, 1]))
        stored_legacy_total = float(_attr(sim, "total_power_W", 0.0) or 0.0)
        if not np.isclose(expected_legacy_total, stored_legacy_total, rtol=2e-4, atol=1e-4):
            errors.append(
                f"total_power_W must preserve legacy training semantics: "
                f"sum(feature0*area)={expected_legacy_total}, stored={stored_legacy_total}"
            )

        solver_total = float(_attr(sim, "solver_input_total_power_W", np.sum(merged[:, 0])) or 0.0)
        raw_total = float(np.sum(merged[:, 0]))
        if not np.isclose(raw_total, solver_total, rtol=1e-6, atol=1e-6):
            errors.append(
                f"solver_input_total_power_W mismatch: sum(feature0)={raw_total}, stored={solver_total}"
            )

    # The 27-global-feature builder reads these package attributes directly.
    for attr_name in [
        "rdl_thickness_um", "rdl_kappa_w_mk",
        "interposer_thickness_um", "interposer_kappa_w_mk",
        "has_c4", "c4_thickness_um", "c4_kappa_w_mk",
        "substrate_thickness_um", "substrate_kappa_w_mk",
    ]:
        if attr_name not in sim.attrs:
            errors.append(f"missing package/global attribute required for training parity: {attr_name}")

    if "node_temperature_labels" in sim:
        warnings.append("node_temperature_labels exists but will be ignored by inference graph builder")
    if "point_cloud" in sim:
        warnings.append("point_cloud exists but will be ignored by inference graph builder")

    for w in warnings:
        print(f"WARNING {sim.name}: {w}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    args = parser.parse_args()

    path = Path(args.h5_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"ERROR: H5 not found: {path}")

    all_errors: list[str] = []
    with h5py.File(path, "r") as h5:
        if "simulations" not in h5:
            raise SystemExit("ERROR: missing /simulations group")
        names = sorted(h5["simulations"].keys())
        if not names:
            raise SystemExit("ERROR: /simulations is empty")
        print(f"Validating inference H5: {path}")
        print(f"Simulation groups: {len(names)}")
        for name in names:
            sim = h5["simulations"][name]
            errors = validate_sim(sim)
            if errors:
                all_errors.extend([f"{sim.name}: {e}" for e in errors])
            else:
                merged = sim["block_features_merged"]
                fillers = sim["filler_blocks"]
                print(
                    f"OK {sim.name}: real={merged.shape[0]} filler={fillers.shape[0]} "
                    f"merged_width={merged.shape[1]} filler_width={fillers.shape[1]}"
                )

    if all_errors:
        print("\nVALIDATION FAILED")
        for e in all_errors:
            print(" -", e)
        raise SystemExit(2)

    print("\nVALIDATION PASSED")
    print("This H5 is suitable for the inference graph builder; solver labels are not required.")


if __name__ == "__main__":
    main()
