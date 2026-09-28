#!/usr/bin/env python3
# =============================================================================
# V4.2 FINAL AUDITED DATASET-EXCLUSION REVISION
# =============================================================================
#
# V4.2 DATASET FIX
# ----------------
# Physics, node features, edge features, global features, QA rules, and
# filler-temperature mapping are unchanged from V4.1.
#
# The only dataset-level change is that HDF5 group `sim_001460`
# (stored attr sim_id=`sim_00981`) is explicitly excluded before graph building.
# Dataset audit found that filler index 1 in this single simulation has no
# point_cloud center inside its X/Y footprint. We deliberately SKIP the whole
# simulation rather than invent/interpolate a filler temperature.
#
# The source HDF5 is opened read-only; this builder does NOT delete or modify it.
# =============================================================================
#
# IMPORTANT FINDINGS FROM SERVER + FEA AUDIT
# ------------------------------------------
# 1) For data_source == "3dice_real_fea_v3_self_contained", the historical
#    generator wrote raw block feature column 0 directly to 3D-ICE
#    `power values`.  Therefore, for the EXISTING V3 FEA labels:
#
#        solver-faithful P_real [W] = raw block_features[:, 0]
#        P_filler [W] = 0
#
#    The historical HDF5 attr total_power_W was instead reconstructed as
#    sum(power_density_W_mm2 * area_mm2).  It is preserved as LEGACY metadata
#    but MUST NOT be used as the physics source term for these old FEA labels.
#
# 2) Correcting the source power fixed the dominant global-energy mismatch:
#    median global closure error fell from ~60% to ~0.5% in a 100-graph audit,
#    and graph-equilibrium vs FEA fell from ~114 K median to ~6 K median.
#
# 3) Hard node-by-node KCL is still NOT solver-faithful on the current coarse
#    block/filler graph.  Layer conservation is also poor.  Raw xyaxis solver
#    files containing cell width/height are no longer available, while HDF5
#    point_cloud preserves only X/Y/Z centers + T.
#
# Therefore V4:
#   - preserves the existing ML graph contract (13 node / 11 edge / 27 global)
#   - stores solver-faithful V3 source power separately for physics
#   - NEVER marks the current coarse graph as exact hard-node PINN-ready
#   - provides a conservative EXPERIMENTAL GLOBAL energy constraint only when
#     the ground-truth graph passes a per-graph global closure QA gate
#   - keeps node/layer KCL residuals as DIAGNOSTICS, not training truth
#   - excludes legacy V2 power semantics from physics until independently proven
#
# The legacy global_features[2] is intentionally left unchanged for backward ML
# compatibility.  Use graph.solver_input_total_power_W for physics.
# =============================================================================

# PINN V4 AUDITED GLOBAL-ENERGY REVISION
# Generated from the user's filler-temperature Graph Builder.
# Current H5 inspection confirmed NO explicit boundary face/region/exposed-area
# mapping. However, inspection of generate_thermal_dataset_v3.py confirmed that
# the actual 3D-ICE input always writes `top heat sink:` with the sampled HTC
# and ambient temperature. Therefore the top external surface is recovered from
# authoritative generator behavior, not guessed.
#
"""
Thermal V2 HDF5 -> PyTorch Geometric Graph Builder

FINAL FILLER-AWARE VERSION
WITH FULL CHIP WIDTH + HEIGHT GLOBAL FEATURES
================================================

GLOBAL FEATURE UPDATE
---------------------
The full chip/layer footprint is now explicitly included:

global_features[25] = chip_width_mm
global_features[26] = chip_height_mm

These values are read from HDF5:

chip_w_um
chip_h_um

and converted to mm.

Final feature dimensions:

Node features   : 13
Edge features   : 11
Global features : 27

PINN RAW PHYSICS TENSORS
------------------------
physics_node_power_W
physics_mask
ground_truth_temperature_K
ground_truth_temperature_mask
edge_resistance_K_W
edge_conductance_W_K
top_surface_candidate_area_mm2
bottom_surface_candidate_area_mm2
cooling_exposed_area_mm2
G_amb_W_K

BOUNDARY PHYSICS
----------------
The H5 stores ambient temperature + HTC + cooling type but not the face.
The original generator was inspected and explicitly writes `top heat sink:`.
Therefore, unless a future H5 contains a more explicit node/area mapping,
the builder applies convection to the highest-Z layer (layer n_layers-1).
For each top-surface real/filler node: exposed_area_mm2 = footprint area.
Internal nodes receive exposed_area_mm2 = 0.


NODES
-----
Real floorplan blocks:

block_type:
1 = logic
2 = memory

Reconstructed filler regions:

block_type = 0
power_density = 0
kappa = filler_kappa_w_mk from HDF5

Real nodes are stored first.
Filler nodes are appended after them.


TARGETS
-------
Real nodes use node_temperature_labels.

Filler nodes receive a representative solver temperature from point_cloud:
- select point-cloud samples inside the filler X/Y footprint
- select the Z plane nearest the filler node center
- y = mean temperature on that plane inside the filler footprint

Filler temperatures are kept separate from trusted real-node supervision:
target_mask = False
filler_temperature_mask = True


EDGE TYPES
----------
0 = direct same-layer side contact

1 = recovered same-layer missing-filler path

This is only used when post-hoc filler reconstruction left
a geometric gap that physically represents filler material.

It is NOT:

- KNN
- nearest-neighbor graph construction
- arbitrary fallback
- diagonal bridge

2 = vertical positive XY overlap between adjacent layers


THERMAL RESISTANCE
------------------
Fourier:

R = L / (k A)

Geometry is internally represented in mm:

R[K/W] = 1000 * L_mm / (k * A_mm2)


LATERAL DIRECT:

R =
R_half_A
+
R_half_B


LATERAL THROUGH MISSING RECONSTRUCTED FILLER:

R =
R_half_A
+
R_filler_gap
+
R_half_B


VERTICAL:

R =
R_half_layer_A
+
R_BOND
+
R_half_layer_B


IMPORTANT
---------
Equivalent kappa is NEVER used to calculate resistance.

effective_path_kappa is calculated AFTER resistance
only as a diagnostic edge feature.

Exact TSV positions are not reconstructed or invented.

n_tsv_pairs remains aggregate graph metadata.

RDL / Interposer / C4 / substrate remain graph/global features
because the HDF5 does not provide spatial graph-node geometry
and node-temperature targets for these package layers.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch_geometric.data import Data


# ==============================================================
# CONFIG
# ==============================================================

EPS = 1e-12

DEFAULT_CONTACT_TOL_MM = 0.001

# ------------------------------------------------------------------
# AUDITED DATASET EXCLUSION
# ------------------------------------------------------------------
# Whole-dataset audit:
#   - 268,241 filler regions checked
#   - exactly 1 bad filler
#   - exactly 1 affected simulation
#
# HDF5 group sim_001460 has attr sim_id=sim_00981 and legacy V2 point-cloud
# data. Filler index 1 has footprint approximately:
#   X=[0.000, 0.050] mm
#   Y=[2.050, 5.850] mm
# with zero point-cloud centers inside that footprint.
#
# Do NOT approximate this filler temperature. Exclude the whole simulation.
SKIP_SIMULATION_GROUPS = {
    "sim_001460": (
        "audited bad legacy V2 sample: filler 1 has zero point_cloud "
        "centers inside its X/Y footprint; stored attr sim_id=sim_00981"
    ),
    "sim_003715": (
        "bad sample discovered during full graph build: filler 1 has zero "
        "point_cloud centers inside its X/Y footprint"
    ),
}

# Empirical QA threshold for the historical coarse graph.
# This is NOT a physical constant. It only decides whether the stored
# ground-truth sample is sufficiently consistent with the reconstructed
# global convection balance to permit the optional global-energy regularizer.
GLOBAL_ENERGY_CLOSURE_GATE_PCT = 5.0


# ==============================================================
# FEATURE CONTRACT
# ==============================================================

NODE_FEATURE_NAMES = [
    "power_density_W_mm2",       # 0
    "area_mm2",                  # 1
    "layer_idx",                 # 2
    "kappa_W_mK",                # 3
    "block_type_code",           # 4: 0=filler, 1=logic, 2=memory
    "has_tsv",                   # 5
    "is_hotspot",                # 6
    "x_center_mm",               # 7
    "y_center_mm",               # 8
    "z_center_mm",               # 9
    "width_mm",                  # 10
    "height_mm",                 # 11
    # Cooling-exposed top-face area. For the current dataset the face is
    # recovered from the original generator's explicit `top heat sink:`.
    "exposed_area_mm2",          # 12
]


EDGE_FEATURE_NAMES = [
    "resistance_K_W",                 # 0
    "is_vertical",                    # 1
    "euclidean_distance_mm",          # 2
    "delta_z_mm",                     # 3
    "contact_area_mm2",               # 4
    "effective_path_kappa_W_mK",      # 5 diagnostic only
    "conductance_W_K",                # 6
    "edge_type_code",                 # 7
    "thermal_path_mm",                # 8
    "has_bond_interface",             # 9
    "interface_or_filler_gap_mm",     # 10
]


GLOBAL_FEATURE_NAMES = [
    "ambient_temp_K",                 # 0
    "workload_factor",                # 1
    "total_power_W",                  # 2

    "n_layers",                       # 3
    "n_tsv_pairs",                    # 4

    "has_hotspot",                    # 5
    "hotspot_multiplier",             # 6

    "cooling_type_code",              # 7
    "cooling_htc_w_m2k",              # 8

    "package_recipe_code",            # 9
    "non_uniform_material",           # 10

    "bond_kappa_w_mk",                # 11
    "bond_thickness_mm",              # 12

    "has_rdl",                        # 13
    "rdl_thickness_mm",               # 14
    "rdl_kappa_w_mk",                 # 15

    "has_interposer",                 # 16
    "interposer_thickness_mm",        # 17
    "interposer_kappa_w_mk",          # 18

    "has_c4",                         # 19
    "c4_thickness_mm",                # 20
    "c4_kappa_w_mk",                  # 21

    "has_package_substrate",          # 22
    "substrate_thickness_mm",         # 23
    "substrate_kappa_w_mk",           # 24

    # NEW: full die/layer footprint
    "chip_width_mm",                  # 25
    "chip_height_mm",                 # 26
]


# ==============================================================
# DATA CLASSES
# ==============================================================

@dataclass
class NodeGeometry:
    idx: int
    kind: str
    layer: int

    x: float
    y: float

    w: float
    h: float

    z: float

    thickness: float
    kappa: float

    @property
    def x1(self):
        return self.x

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y

    @property
    def y2(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2.0

    @property
    def cy(self):
        return self.y + self.h / 2.0


@dataclass
class PhysicalEdge:
    a: int
    b: int
    attr: List[float]
    edge_kind: str


# ==============================================================
# ATTRIBUTE HELPERS
# ==============================================================

def decode_value(value):

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.bytes_):
        return value.tobytes().decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.generic):
        return value.item()

    return value


def get_attr(
    group,
    name,
    default=None,
):

    if name not in group.attrs:
        return default

    return decode_value(
        group.attrs[name]
    )


def float_attr(
    group,
    name,
    default=0.0,
):

    value = get_attr(
        group,
        name,
        default,
    )

    try:
        return float(value)

    except (TypeError, ValueError):
        return float(default)


def int_attr(
    group,
    name,
    default=0,
):

    value = get_attr(
        group,
        name,
        default,
    )

    try:
        return int(
            round(
                float(value)
            )
        )

    except (TypeError, ValueError):
        return int(default)


def str_attr(
    group,
    name,
    default="",
):

    return str(
        get_attr(
            group,
            name,
            default,
        )
    )


# ==============================================================
# COLUMN HELPERS
# ==============================================================

def parse_columns(value):
    """
    Parse comma-separated HDF5 column names without splitting
    commas appearing inside parentheses.

    Example:

    block_type(1=logic,2=memory)

    must remain one column.
    """

    if value is None:
        return []

    value = str(
        decode_value(value)
    )

    columns = []
    current = []
    paren_depth = 0

    for char in value:

        if char == "(":
            paren_depth += 1
            current.append(char)

        elif char == ")":
            paren_depth = max(
                0,
                paren_depth - 1,
            )
            current.append(char)

        elif (
            char == ","
            and paren_depth == 0
        ):

            column = "".join(
                current
            ).strip()

            if column:
                columns.append(
                    column
                )

            current = []

        else:
            current.append(char)

    column = "".join(
        current
    ).strip()

    if column:
        columns.append(column)

    return columns


def find_column(
    columns,
    names,
    fallback=None,
):

    for i, col in enumerate(columns):

        low = col.lower()

        for name in names:

            name_low = name.lower()

            if (
                low == name_low
                or low.startswith(
                    name_low
                )
            ):
                return i

    if fallback is not None:
        return fallback

    raise KeyError(
        f"Could not locate {names} "
        f"in columns {columns}"
    )


# ==============================================================
# GEOMETRY
# ==============================================================

def overlap_1d(
    a1,
    a2,
    b1,
    b2,
):

    return max(
        0.0,
        min(a2, b2)
        -
        max(a1, b1),
    )


def rectangle_intersection_area(
    ax1,
    ay1,
    ax2,
    ay2,
    bx1,
    by1,
    bx2,
    by2,
):

    return (
        overlap_1d(
            ax1,
            ax2,
            bx1,
            bx2,
        )
        *
        overlap_1d(
            ay1,
            ay2,
            by1,
            by2,
        )
    )


def center_distance(
    a,
    b,
):

    dx = a.cx - b.cx
    dy = a.cy - b.cy
    dz = a.z - b.z

    return math.sqrt(
        dx * dx
        +
        dy * dy
        +
        dz * dz
    )


# ==============================================================
# THERMAL PHYSICS
# ==============================================================

def resistance_segment(
    length_mm,
    kappa_w_mk,
    area_mm2,
):

    if length_mm <= EPS:
        return 0.0

    if kappa_w_mk <= EPS:

        raise ValueError(
            f"Invalid kappa={kappa_w_mk}"
        )

    if area_mm2 <= EPS:

        raise ValueError(
            f"Invalid area={area_mm2}"
        )

    # ----------------------------------------------------------
    # R = L / (k A)
    #
    # L_mm  -> L_m  = L_mm * 1e-3
    # A_mm2 -> A_m2 = A_mm2 * 1e-6
    #
    # Therefore:
    #
    # R[K/W] = 1000 * L_mm / (k * A_mm2)
    # ----------------------------------------------------------

    return (
        1000.0
        *
        length_mm
        /
        (
            kappa_w_mk
            *
            area_mm2
        )
    )


def diagnostic_effective_kappa(
    path_mm,
    area_mm2,
    resistance,
):
    """
    Diagnostic only.

    This is calculated FROM the already-computed resistance.

    It is NOT used to calculate resistance.
    """

    if (
        path_mm <= EPS
        or area_mm2 <= EPS
        or resistance <= EPS
    ):
        return 0.0

    return (
        1000.0
        *
        path_mm
        /
        (
            resistance
            *
            area_mm2
        )
    )


# ==============================================================
# STACK Z POSITIONS
# ==============================================================

def calculate_layer_z(
    thicknesses_mm,
    bond_thickness_mm,
):

    centers = []

    z_cursor = 0.0

    for i, thickness in enumerate(
        thicknesses_mm
    ):

        centers.append(
            z_cursor
            +
            thickness / 2.0
        )

        z_cursor += thickness

        if (
            i
            <
            len(thicknesses_mm) - 1
        ):
            z_cursor += (
                bond_thickness_mm
            )

    return np.asarray(
        centers,
        dtype=np.float64,
    )


# ==============================================================
# BUILD REAL + FILLER NODES -- ROBUST MIXED-SCHEMA VERSION
# ==============================================================

def build_nodes(
    sim,
    hdf,
):
    """
    Robust loader for the balanced V3/V2-schema HDF5.

    Required core datasets:
        floorplan_blocks          [N,4]
        block_features            [N,7]
        node_temperature_labels   [N]
        filler_blocks             [F,7] preferred, [F,5] legacy fallback

    block_features_merged may be:
        [N,15] -> use exact z_center_mm and thickness_mm as geometry
        [N,7]  -> reconstruct z/thickness from layer stack attrs
        absent -> same reconstruction fallback

    IMPORTANT:
        graph.x remains a compact ML tensor; this PINN-ready version has 13 features.
        The 15-column merged table is NEVER passed directly to the model.
    """

    required = [
        "floorplan_blocks",
        "block_features",
        "node_temperature_labels",
        "filler_blocks",
    ]

    missing = [
        name
        for name in required
        if name not in sim
    ]

    if missing:
        raise ValueError(
            f"{sim.name}: missing required dataset(s): {missing}"
        )

    floorplan = np.asarray(
        sim["floorplan_blocks"][:],
        dtype=np.float64,
    )

    block_features = np.asarray(
        sim["block_features"][:],
        dtype=np.float64,
    )

    real_labels = np.asarray(
        sim["node_temperature_labels"][:],
        dtype=np.float64,
    ).reshape(-1)

    filler_blocks = np.asarray(
        sim["filler_blocks"][:],
        dtype=np.float64,
    )

    # Solver point cloud is required to assign physical temperatures
    # to filler nodes. Columns: X_mm, Y_mm, Z_mm, Temp_K.
    point_cloud = None
    if filler_blocks.shape[0] > 0:
        if "point_cloud" not in sim:
            raise ValueError(
                f"{sim.name}: filler nodes exist but point_cloud is missing; "
                "cannot assign physical filler temperatures"
            )

        point_cloud = np.asarray(
            sim["point_cloud"][:],
            dtype=np.float64,
        )

        if (
            point_cloud.ndim != 2
            or point_cloud.shape[1] < 4
        ):
            raise ValueError(
                f"{sim.name}: point_cloud must be [M,4+] as X,Y,Z,T; "
                f"got {point_cloud.shape}"
            )

        point_cloud = point_cloud[:, :4]

        if not np.all(np.isfinite(point_cloud)):
            raise ValueError(
                f"{sim.name}: point_cloud contains NaN/Inf"
            )

    n_real = int(floorplan.shape[0])
    n_filler = int(filler_blocks.shape[0])

    if (
        floorplan.ndim != 2
        or floorplan.shape != (n_real, 4)
    ):
        raise ValueError(
            f"{sim.name}: floorplan_blocks must be [N,4], "
            f"got {floorplan.shape}"
        )

    if (
        block_features.ndim != 2
        or block_features.shape != (n_real, 7)
    ):
        raise ValueError(
            f"{sim.name}: block_features must be [N,7], "
            f"got {block_features.shape}"
        )

    if len(real_labels) != n_real:
        raise ValueError(
            f"{sim.name}: temperature labels={len(real_labels)} "
            f"but real nodes={n_real}"
        )

    if (
        filler_blocks.ndim != 2
        or filler_blocks.shape[1] not in (5, 7)
    ):
        raise ValueError(
            f"{sim.name}: filler_blocks must be [F,7] "
            f"or legacy [F,5], got {filler_blocks.shape}"
        )

    if not np.all(np.isfinite(floorplan)):
        raise ValueError(
            f"{sim.name}: floorplan_blocks contains NaN/Inf"
        )

    if not np.all(np.isfinite(block_features)):
        raise ValueError(
            f"{sim.name}: block_features contains NaN/Inf"
        )

    if not np.all(np.isfinite(real_labels)):
        raise ValueError(
            f"{sim.name}: node_temperature_labels contains NaN/Inf"
        )

    if not np.all(np.isfinite(filler_blocks)):
        raise ValueError(
            f"{sim.name}: filler_blocks contains NaN/Inf"
        )

    # ----------------------------------------------------------
    # Main block feature columns
    # ----------------------------------------------------------

    columns = parse_columns(
        get_attr(
            sim,
            "block_features_columns",
            "",
        )
    )

    power_col = find_column(
        columns,
        ["power_density_w_mm2"],
        0,
    )

    area_col = find_column(
        columns,
        ["area_mm2"],
        1,
    )

    layer_col = find_column(
        columns,
        ["layer_idx"],
        2,
    )

    kappa_col = find_column(
        columns,
        ["kappa_w_mk"],
        3,
    )

    block_type_col = find_column(
        columns,
        ["block_type"],
        4,
    )

    has_tsv_col = find_column(
        columns,
        ["has_tsv"],
        5,
    )

    hotspot_col = find_column(
        columns,
        ["is_hotspot"],
        6,
    )

    # ----------------------------------------------------------
    # Layer stack
    # ----------------------------------------------------------

    inferred_n_layers = int(
        np.max(
            block_features[:, layer_col]
        )
        + 1
    )

    n_layers = int_attr(
        sim,
        "n_layers",
        inferred_n_layers,
    )

    thicknesses_raw = get_attr(
        sim,
        "layer_thicknesses_um",
        None,
    )

    if thicknesses_raw is None:
        raise ValueError(
            f"{sim.name}: missing layer_thicknesses_um"
        )

    thicknesses_um = np.asarray(
        thicknesses_raw,
        dtype=np.float64,
    ).reshape(-1)

    if len(thicknesses_um) != n_layers:
        raise ValueError(
            f"{sim.name}: n_layers={n_layers} but "
            f"layer_thicknesses_um has {len(thicknesses_um)} values"
        )

    if (
        not np.all(np.isfinite(thicknesses_um))
        or np.any(thicknesses_um <= 0)
    ):
        raise ValueError(
            f"{sim.name}: invalid layer_thicknesses_um"
        )

    thicknesses_mm = (
        thicknesses_um
        / 1000.0
    )

    bond_thickness_mm = (
        float_attr(
            sim,
            "bond_thickness_um",
            0.0,
        )
        / 1000.0
    )

    bond_kappa = float_attr(
        sim,
        "bond_kappa_w_mk",
        0.0,
    )

    layer_z = calculate_layer_z(
        thicknesses_mm,
        bond_thickness_mm,
    )

    # ----------------------------------------------------------
    # Full chip footprint
    # ----------------------------------------------------------

    chip_width_mm = (
        float_attr(
            sim,
            "chip_w_um",
            0.0,
        )
        / 1000.0
    )

    chip_height_mm = (
        float_attr(
            sim,
            "chip_h_um",
            0.0,
        )
        / 1000.0
    )

    if chip_width_mm <= 0:
        raise ValueError(
            f"{sim.name}: invalid or missing chip_w_um"
        )

    if chip_height_mm <= 0:
        raise ValueError(
            f"{sim.name}: invalid or missing chip_h_um"
        )

    # ----------------------------------------------------------
    # Filler material
    # ----------------------------------------------------------

    filler_kappa = float_attr(
        sim,
        "filler_kappa_w_mk",
        float_attr(
            hdf,
            "filler_kappa_w_mk_global",
            130.0,
        ),
    )

    if filler_kappa <= 0:
        raise ValueError(
            f"{sim.name}: invalid filler kappa {filler_kappa}"
        )

    # ----------------------------------------------------------
    # Real-node geometry
    #
    # x/y/w/h always come from floorplan_blocks.
    # z/thickness use merged [N,15] when valid, otherwise
    # reconstruct from the layer stack.
    # ----------------------------------------------------------

    real_layers = np.round(
        block_features[:, layer_col]
    ).astype(int)

    if (
        np.any(real_layers < 0)
        or np.any(real_layers >= n_layers)
    ):
        raise ValueError(
            f"{sim.name}: real block layer outside "
            f"0..{n_layers - 1}"
        )

    real_z = layer_z[
        real_layers
    ].copy()

    real_thickness = thicknesses_mm[
        real_layers
    ].copy()

    merged_width = 0
    real_geometry_source = (
        "reconstructed_from_layer_stack"
    )

    if "block_features_merged" in sim:
        merged = np.asarray(
            sim["block_features_merged"][:],
            dtype=np.float64,
        )

        if (
            merged.ndim == 2
            and merged.shape[0] == n_real
        ):
            merged_width = int(
                merged.shape[1]
            )

            if merged.shape[1] >= 15:
                z_candidate = merged[:, 13]
                t_candidate = merged[:, 14]

                merged_geometry_valid = (
                    np.all(
                        np.isfinite(
                            z_candidate
                        )
                    )
                    and np.all(
                        np.isfinite(
                            t_candidate
                        )
                    )
                    and np.all(
                        z_candidate >= 0
                    )
                    and np.all(
                        t_candidate > 0
                    )
                )

                if merged_geometry_valid:
                    real_z = (
                        z_candidate.copy()
                    )
                    real_thickness = (
                        t_candidate.copy()
                    )
                    real_geometry_source = (
                        "block_features_merged_15"
                    )

    # ----------------------------------------------------------
    # Filler geometry
    #
    # New:
    #   [x,y,w,h,layer,z_center,thickness]
    #
    # Legacy fallback:
    #   [x,y,w,h,layer]
    # ----------------------------------------------------------

    filler_layers = np.round(
        filler_blocks[:, 4]
    ).astype(int)

    if n_filler > 0:
        if (
            np.any(
                filler_layers < 0
            )
            or np.any(
                filler_layers >= n_layers
            )
        ):
            raise ValueError(
                f"{sim.name}: filler layer outside "
                f"0..{n_layers - 1}"
            )

    filler_z = layer_z[
        filler_layers
    ].copy()

    filler_thickness = thicknesses_mm[
        filler_layers
    ].copy()

    filler_geometry_source = (
        "filler_blocks_5_reconstructed"
    )

    if filler_blocks.shape[1] == 7:
        filler_z_candidate = (
            filler_blocks[:, 5]
        )

        filler_t_candidate = (
            filler_blocks[:, 6]
        )

        exact_filler_valid = (
            np.all(
                np.isfinite(
                    filler_z_candidate
                )
            )
            and np.all(
                np.isfinite(
                    filler_t_candidate
                )
            )
            and np.all(
                filler_z_candidate >= 0
            )
            and np.all(
                filler_t_candidate > 0
            )
        )

        if exact_filler_valid:
            filler_z = (
                filler_z_candidate.copy()
            )
            filler_thickness = (
                filler_t_candidate.copy()
            )
            filler_geometry_source = (
                "filler_blocks_7"
            )
        else:
            filler_geometry_source = (
                "filler_blocks_7_invalid_z_fallback"
            )

    # ----------------------------------------------------------
    # Allocate graph tensors
    # ----------------------------------------------------------

    total_nodes = (
        n_real
        + n_filler
    )

    x = np.zeros(
        (
            total_nodes,
            len(
                NODE_FEATURE_NAMES
            ),
        ),
        dtype=np.float32,
    )

    # Keep legacy finite placeholder for compatibility.
    # target_mask=False guarantees filler nodes are not supervised.
    y = np.zeros(
        total_nodes,
        dtype=np.float32,
    )

    pos = np.zeros(
        (
            total_nodes,
            3,
        ),
        dtype=np.float32,
    )

    target_mask = np.zeros(
        total_nodes,
        dtype=bool,
    )

    real_mask = np.zeros(
        total_nodes,
        dtype=bool,
    )

    filler_mask = np.zeros(
        total_nodes,
        dtype=bool,
    )

    # True only where y for a filler node was obtained from the
    # solver point cloud. This is intentionally separate from
    # target_mask, which remains reserved for trusted real-node labels.
    filler_temperature_mask = np.zeros(
        total_nodes,
        dtype=bool,
    )

    geometries = []

    # ==========================================================
    # REAL NODES
    # ==========================================================

    for i in range(n_real):
        (
            px,
            py,
            w,
            h,
        ) = map(
            float,
            floorplan[i],
        )

        layer = int(
            real_layers[i]
        )

        if w <= 0 or h <= 0:
            raise ValueError(
                f"{sim.name}: real node {i} "
                "has non-positive width/height"
            )

        kappa = float(
            block_features[
                i,
                kappa_col,
            ]
        )

        if kappa <= 0:
            raise ValueError(
                f"{sim.name}: real node {i} "
                f"has invalid kappa={kappa}"
            )

        area = float(
            block_features[
                i,
                area_col,
            ]
        )

        if area <= 0:
            area = w * h

        cx = (
            px
            + w / 2.0
        )

        cy = (
            py
            + h / 2.0
        )

        cz = float(
            real_z[i]
        )

        thickness = float(
            real_thickness[i]
        )

        if thickness <= 0:
            raise ValueError(
                f"{sim.name}: real node {i} "
                "has thickness <= 0"
            )

        x[i] = [
            float(
                block_features[
                    i,
                    power_col,
                ]
            ),
            area,
            float(layer),
            kappa,
            float(
                block_features[
                    i,
                    block_type_col,
                ]
            ),
            float(
                block_features[
                    i,
                    has_tsv_col,
                ]
            ),
            float(
                block_features[
                    i,
                    hotspot_col,
                ]
            ),
            cx,
            cy,
            cz,
            w,
            h,
            0.0,  # exposed_area_mm2; filled after solver boundary mapping
        ]

        y[i] = (
            real_labels[i]
        )

        pos[i] = [
            cx,
            cy,
            cz,
        ]

        target_mask[i] = True
        real_mask[i] = True

        geometries.append(
            NodeGeometry(
                idx=i,
                kind="real",
                layer=layer,
                x=px,
                y=py,
                w=w,
                h=h,
                z=cz,
                thickness=thickness,
                kappa=kappa,
            )
        )

    # ==========================================================
    # FILLER NODES
    # ==========================================================

    for filler_idx in range(
        n_filler
    ):
        global_idx = (
            n_real
            + filler_idx
        )

        px = float(
            filler_blocks[
                filler_idx,
                0,
            ]
        )

        py = float(
            filler_blocks[
                filler_idx,
                1,
            ]
        )

        w = float(
            filler_blocks[
                filler_idx,
                2,
            ]
        )

        h = float(
            filler_blocks[
                filler_idx,
                3,
            ]
        )

        layer = int(
            filler_layers[
                filler_idx
            ]
        )

        if w <= 0 or h <= 0:
            raise ValueError(
                f"{sim.name}: filler node "
                f"{filler_idx} has non-positive width/height"
            )

        area = w * h

        cx = (
            px
            + w / 2.0
        )

        cy = (
            py
            + h / 2.0
        )

        cz = float(
            filler_z[
                filler_idx
            ]
        )

        thickness = float(
            filler_thickness[
                filler_idx
            ]
        )

        if thickness <= 0:
            raise ValueError(
                f"{sim.name}: filler node "
                f"{filler_idx} has thickness <= 0"
            )

        x[global_idx] = [
            0.0,
            area,
            float(layer),
            filler_kappa,
            0.0,
            0.0,
            0.0,
            cx,
            cy,
            cz,
            w,
            h,
            0.0,  # exposed_area_mm2; filled after solver boundary mapping
        ]

        # ------------------------------------------------------
        # FILLER TEMPERATURE FROM THE SOLVER POINT CLOUD
        # ------------------------------------------------------
        # The filler node represents a region centered at (cx,cy,cz).
        # We first select solver samples inside its X/Y footprint, then
        # use the available Z plane nearest the filler center. The mean
        # temperature on that plane is the representative filler-node T.
        #
        # IMPORTANT: filler power density remains zero. Temperature does
        # NOT become zero merely because the filler generates no heat.
        tol_xy = 1e-9

        xy_mask = (
            (point_cloud[:, 0] >= px - tol_xy)
            & (point_cloud[:, 0] <= px + w + tol_xy)
            & (point_cloud[:, 1] >= py - tol_xy)
            & (point_cloud[:, 1] <= py + h + tol_xy)
        )

        xy_points = point_cloud[xy_mask]

        if xy_points.shape[0] == 0:
            raise ValueError(
                f"{sim.name}: no point_cloud samples found inside "
                f"filler {filler_idx} X/Y footprint"
            )

        unique_z = np.unique(xy_points[:, 2])
        nearest_z = float(
            unique_z[np.argmin(np.abs(unique_z - cz))]
        )

        # isclose protects against tiny floating-point differences in Z.
        z_mask = np.isclose(
            xy_points[:, 2],
            nearest_z,
            rtol=0.0,
            atol=1e-9,
        )
        plane_points = xy_points[z_mask]

        if plane_points.shape[0] == 0:
            raise ValueError(
                f"{sim.name}: failed to map point_cloud Z plane for "
                f"filler {filler_idx}"
            )

        filler_temperature_K = float(
            np.mean(plane_points[:, 3])
        )

        if not np.isfinite(filler_temperature_K):
            raise ValueError(
                f"{sim.name}: non-finite point-cloud temperature for "
                f"filler {filler_idx}"
            )

        y[global_idx] = filler_temperature_K
        filler_temperature_mask[global_idx] = True

        pos[global_idx] = [
            cx,
            cy,
            cz,
        ]

        filler_mask[
            global_idx
        ] = True

        geometries.append(
            NodeGeometry(
                idx=global_idx,
                kind="filler",
                layer=layer,
                x=px,
                y=py,
                w=w,
                h=h,
                z=cz,
                thickness=thickness,
                kappa=filler_kappa,
            )
        )

    aux = {
        "n_real":
        n_real,

        "n_filler":
        n_filler,

        "n_layers":
        n_layers,

        "layer_thicknesses_mm":
        thicknesses_mm,

        "layer_z_mm":
        layer_z,

        "filler_kappa_w_mk":
        filler_kappa,

        "bond_kappa_w_mk":
        bond_kappa,

        "bond_thickness_mm":
        bond_thickness_mm,

        "chip_width_mm":
        chip_width_mm,

        "chip_height_mm":
        chip_height_mm,

        # Mixed-schema diagnostics
        "block_features_merged_width":
        merged_width,

        "real_geometry_source":
        real_geometry_source,

        "filler_blocks_width":
        int(
            filler_blocks.shape[1]
        ),

        "filler_geometry_source":
        filler_geometry_source,
    }

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(pos),
        torch.from_numpy(
            target_mask
        ),
        torch.from_numpy(
            real_mask
        ),
        torch.from_numpy(
            filler_mask
        ),
        torch.from_numpy(
            filler_temperature_mask
        ),
        geometries,
        aux,
    )


# ==============================================================
# DIRECT SIDE CONTACT
# ==============================================================

def direct_side_contact(
    a,
    b,
    tolerance,
):
    """
    Return:

    orientation,
    shared_length

    orientation = x:
    heat crosses X-facing sides

    orientation = y:
    heat crosses Y-facing sides
    """

    y_overlap = overlap_1d(
        a.y1,
        a.y2,
        b.y1,
        b.y2,
    )

    x_overlap = overlap_1d(
        a.x1,
        a.x2,
        b.x1,
        b.x2,
    )

    x_gap = min(
        abs(
            a.x2
            -
            b.x1
        ),
        abs(
            b.x2
            -
            a.x1
        ),
    )

    if (
        x_gap <= tolerance
        and y_overlap > tolerance
    ):
        return (
            "x",
            y_overlap,
        )

    y_gap = min(
        abs(
            a.y2
            -
            b.y1
        ),
        abs(
            b.y2
            -
            a.y1
        ),
    )

    if (
        y_gap <= tolerance
        and x_overlap > tolerance
    ):
        return (
            "y",
            x_overlap,
        )

    return None


# ==============================================================
# AXIS-ALIGNED MISSING-FILLER GAP
# ==============================================================

def axis_aligned_gap(
    a,
    b,
    tolerance,
):
    """
    Returns:

    orientation,
    overlap_length,
    gap

    Only axis-aligned thermal corridors are allowed.

    Diagonal recovery is forbidden.
    """

    y_overlap = overlap_1d(
        a.y1,
        a.y2,
        b.y1,
        b.y2,
    )

    x_overlap = overlap_1d(
        a.x1,
        a.x2,
        b.x1,
        b.x2,
    )

    # Horizontal A -> B
    if (
        a.x2
        <
        b.x1
        -
        tolerance
    ):

        gap = (
            b.x1
            -
            a.x2
        )

        if y_overlap > tolerance:
            return (
                "x",
                y_overlap,
                gap,
            )

    if (
        b.x2
        <
        a.x1
        -
        tolerance
    ):

        gap = (
            a.x1
            -
            b.x2
        )

        if y_overlap > tolerance:
            return (
                "x",
                y_overlap,
                gap,
            )

    # Vertical in floorplan Y direction
    if (
        a.y2
        <
        b.y1
        -
        tolerance
    ):

        gap = (
            b.y1
            -
            a.y2
        )

        if x_overlap > tolerance:
            return (
                "y",
                x_overlap,
                gap,
            )

    if (
        b.y2
        <
        a.y1
        -
        tolerance
    ):

        gap = (
            a.y1
            -
            b.y2
        )

        if x_overlap > tolerance:
            return (
                "y",
                x_overlap,
                gap,
            )

    return None


# ==============================================================
# RECOVERY CORRIDOR
# ==============================================================

def recovery_corridor(
    a,
    b,
    orientation,
):

    if orientation == "x":

        if a.cx < b.cx:

            x1 = a.x2
            x2 = b.x1

        else:

            x1 = b.x2
            x2 = a.x1

        y1 = max(
            a.y1,
            b.y1,
        )

        y2 = min(
            a.y2,
            b.y2,
        )

    else:

        if a.cy < b.cy:

            y1 = a.y2
            y2 = b.y1

        else:

            y1 = b.y2
            y2 = a.y1

        x1 = max(
            a.x1,
            b.x1,
        )

        x2 = min(
            a.x2,
            b.x2,
        )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def corridor_is_clear(
    a,
    b,
    orientation,
    geometries,
    tolerance,
):

    (
        x1,
        y1,
        x2,
        y2,
    ) = recovery_corridor(
        a,
        b,
        orientation,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return False

    small = min(
        tolerance / 4.0,
        1e-6,
    )

    x1 += small
    y1 += small
    x2 -= small
    y2 -= small

    for other in geometries:

        if other.idx in (
            a.idx,
            b.idx,
        ):
            continue

        if other.layer != a.layer:
            continue

        area = (
            rectangle_intersection_area(
                x1,
                y1,
                x2,
                y2,

                other.x1,
                other.y1,
                other.x2,
                other.y2,
            )
        )

        if area > 1e-10:
            return False

    return True


# ==============================================================
# LATERAL EDGE
# ==============================================================

def make_lateral_edge(
    a,
    b,
    orientation,
    shared_length,
    filler_gap,
    filler_kappa,
    edge_type,
):

    thickness = (
        a.thickness
        +
        b.thickness
    ) / 2.0

    contact_area = (
        shared_length
        *
        thickness
    )

    if contact_area <= EPS:

        raise ValueError(
            "Zero lateral contact area"
        )

    if orientation == "x":

        length_a = (
            a.w
            /
            2.0
        )

        length_b = (
            b.w
            /
            2.0
        )

    else:

        length_a = (
            a.h
            /
            2.0
        )

        length_b = (
            b.h
            /
            2.0
        )

    resistance = resistance_segment(
        length_a,
        a.kappa,
        contact_area,
    )

    if filler_gap > EPS:

        resistance += (
            resistance_segment(
                filler_gap,
                filler_kappa,
                contact_area,
            )
        )

    resistance += (
        resistance_segment(
            length_b,
            b.kappa,
            contact_area,
        )
    )

    thermal_path = (
        length_a
        +
        filler_gap
        +
        length_b
    )

    effective_kappa = (
        diagnostic_effective_kappa(
            thermal_path,
            contact_area,
            resistance,
        )
    )

    conductance = (
        1.0
        /
        resistance
    )

    edge_attr = [
        resistance,               # 0
        0.0,                      # 1 is_vertical

        center_distance(
            a,
            b,
        ),                        # 2

        0.0,                      # 3 delta_z
        contact_area,             # 4
        effective_kappa,          # 5
        conductance,              # 6
        float(edge_type),         # 7
        thermal_path,             # 8
        0.0,                      # 9 has_bond_interface
        filler_gap,               # 10
    ]

    if edge_type == 0:

        edge_kind = (
            "direct_lateral"
        )

    else:

        edge_kind = (
            "recovered_filler_gap"
        )

    return PhysicalEdge(
        a.idx,
        b.idx,
        edge_attr,
        edge_kind,
    )


# ==============================================================
# VERTICAL EDGE
# ==============================================================

def make_vertical_edge(
    a,
    b,
    bond_kappa,
    bond_thickness,
):

    if (
        abs(
            a.layer
            -
            b.layer
        )
        != 1
    ):
        return None

    x_overlap = overlap_1d(
        a.x1,
        a.x2,
        b.x1,
        b.x2,
    )

    y_overlap = overlap_1d(
        a.y1,
        a.y2,
        b.y1,
        b.y2,
    )

    contact_area = (
        x_overlap
        *
        y_overlap
    )

    if contact_area <= EPS:
        return None

    # ----------------------------------------------------------
    # half layer A + BOND_MAT + half layer B
    # ----------------------------------------------------------

    resistance = resistance_segment(
        a.thickness / 2.0,
        a.kappa,
        contact_area,
    )

    if bond_thickness > EPS:

        if bond_kappa <= EPS:

            raise ValueError(
                "Bond thickness exists "
                "but bond kappa <= 0"
            )

        resistance += (
            resistance_segment(
                bond_thickness,
                bond_kappa,
                contact_area,
            )
        )

    resistance += (
        resistance_segment(
            b.thickness / 2.0,
            b.kappa,
            contact_area,
        )
    )

    thermal_path = (
        a.thickness / 2.0
        +
        bond_thickness
        +
        b.thickness / 2.0
    )

    effective_kappa = (
        diagnostic_effective_kappa(
            thermal_path,
            contact_area,
            resistance,
        )
    )

    conductance = (
        1.0
        /
        resistance
    )

    dz = abs(
        a.z
        -
        b.z
    )

    edge_attr = [
        resistance,              # 0
        1.0,                     # 1 vertical

        center_distance(
            a,
            b,
        ),                       # 2

        dz,                      # 3
        contact_area,            # 4
        effective_kappa,         # 5
        conductance,             # 6
        2.0,                     # 7 vertical type
        thermal_path,            # 8

        (
            1.0
            if bond_thickness > EPS
            else 0.0
        ),                       # 9

        bond_thickness,          # 10
    ]

    return PhysicalEdge(
        a.idx,
        b.idx,
        edge_attr,
        "vertical",
    )


# ==============================================================
# CONNECTIVITY
# ==============================================================

def connected_components(
    n_nodes,
    edges,
):

    adjacency = [
        []
        for _ in range(
            n_nodes
        )
    ]

    degree = [
        0
        for _ in range(
            n_nodes
        )
    ]

    for a, b in edges:

        adjacency[a].append(b)
        adjacency[b].append(a)

        degree[a] += 1
        degree[b] += 1

    visited = [
        False
        for _ in range(
            n_nodes
        )
    ]

    components = []

    for start in range(
        n_nodes
    ):

        if visited[start]:
            continue

        stack = [start]

        visited[start] = True

        component = []

        while stack:

            node = stack.pop()

            component.append(
                node
            )

            for neighbour in (
                adjacency[node]
            ):

                if not visited[
                    neighbour
                ]:

                    visited[
                        neighbour
                    ] = True

                    stack.append(
                        neighbour
                    )

        components.append(
            component
        )

    components.sort(
        key=len,
        reverse=True,
    )

    isolated = [
        i
        for i, d
        in enumerate(
            degree
        )
        if d == 0
    ]

    return (
        components,
        isolated,
    )


# ==============================================================
# RECOVER MISSING RECONSTRUCTED FILLER
# ==============================================================

def recover_missing_filler_paths(
    geometries,
    physical_edges,
    filler_kappa,
    tolerance,
    max_gap=None,
):
    """
    Recovery is allowed only if:

    1. nodes are in different graph components
    2. same layer
    3. axis aligned
    4. positive perpendicular overlap
    5. clear corridor
    6. filler material explicitly appears in resistance
    """

    added = 0

    while True:

        components, _ = (
            connected_components(
                len(
                    geometries
                ),
                [
                    (
                        edge.a,
                        edge.b,
                    )
                    for edge
                    in physical_edges
                ],
            )
        )

        if len(
            components
        ) == 1:
            return added

        component_of = {}

        for comp_id, component in enumerate(
            components
        ):

            for node in component:

                component_of[
                    node
                ] = comp_id

        candidates = []

        for i in range(
            len(
                geometries
            )
        ):

            a = geometries[i]

            for j in range(
                i + 1,
                len(
                    geometries
                ),
            ):

                b = geometries[j]

                if (
                    component_of[i]
                    ==
                    component_of[j]
                ):
                    continue

                if (
                    a.layer
                    !=
                    b.layer
                ):
                    continue

                gap_info = (
                    axis_aligned_gap(
                        a,
                        b,
                        tolerance,
                    )
                )

                if gap_info is None:
                    continue

                (
                    orientation,
                    shared_length,
                    gap,
                ) = gap_info

                if (
                    max_gap is not None
                    and
                    gap > max_gap
                ):
                    continue

                if not corridor_is_clear(
                    a,
                    b,
                    orientation,
                    geometries,
                    tolerance,
                ):
                    continue

                candidates.append(
                    (
                        gap,
                        -shared_length,
                        i,
                        j,
                        orientation,
                        shared_length,
                    )
                )

        if not candidates:
            return added

        candidates.sort()

        (
            gap,
            _,
            i,
            j,
            orientation,
            shared_length,
        ) = candidates[0]

        new_edge = (
            make_lateral_edge(
                geometries[i],
                geometries[j],

                orientation,
                shared_length,

                gap,

                filler_kappa,

                edge_type=1,
            )
        )

        physical_edges.append(
            new_edge
        )

        added += 1


# ==============================================================
# BUILD EDGES
# ==============================================================

def build_edges(
    geometries,
    filler_kappa,
    bond_kappa,
    bond_thickness,
    contact_tolerance,
    max_recovery_gap,
):

    physical_edges = []

    n_nodes = len(
        geometries
    )

    # ==========================================================
    # 1. DIRECT SAME-LAYER CONTACT
    # ==========================================================

    for i in range(
        n_nodes
    ):

        a = geometries[i]

        for j in range(
            i + 1,
            n_nodes,
        ):

            b = geometries[j]

            if (
                a.layer
                !=
                b.layer
            ):
                continue

            contact = (
                direct_side_contact(
                    a,
                    b,
                    contact_tolerance,
                )
            )

            if contact is None:
                continue

            (
                orientation,
                shared_length,
            ) = contact

            physical_edges.append(
                make_lateral_edge(
                    a,
                    b,

                    orientation,
                    shared_length,

                    filler_gap=0.0,

                    filler_kappa=
                    filler_kappa,

                    edge_type=0,
                )
            )

    # ==========================================================
    # 2. VERTICAL ADJACENT-LAYER OVERLAP
    # ==========================================================

    for i in range(
        n_nodes
    ):

        a = geometries[i]

        for j in range(
            i + 1,
            n_nodes,
        ):

            b = geometries[j]

            if (
                abs(
                    a.layer
                    -
                    b.layer
                )
                != 1
            ):
                continue

            edge = (
                make_vertical_edge(
                    a,
                    b,
                    bond_kappa,
                    bond_thickness,
                )
            )

            if edge is not None:

                physical_edges.append(
                    edge
                )

    # ==========================================================
    # 3. RECOVER ONLY MISSING FILLER PATHS
    # ==========================================================

    recover_missing_filler_paths(
        geometries,
        physical_edges,
        filler_kappa,
        contact_tolerance,
        max_recovery_gap,
    )

    # ==========================================================
    # FINAL CONNECTIVITY CHECK
    # ==========================================================

    components, isolated = (
        connected_components(
            n_nodes,

            [
                (
                    edge.a,
                    edge.b,
                )
                for edge
                in physical_edges
            ],
        )
    )

    if len(
        components
    ) != 1:

        raise ValueError(
            "physical graph disconnected "
            "after valid filler recovery "
            f"| components="
            f"{len(components)} "
            f"| sizes="
            f"{[len(c) for c in components]} "
            f"| isolated="
            f"{isolated}"
        )

    # ==========================================================
    # COUNTS
    # ==========================================================

    direct_lateral = sum(
        edge.edge_kind
        ==
        "direct_lateral"

        for edge
        in physical_edges
    )

    recovered_lateral = sum(
        edge.edge_kind
        ==
        "recovered_filler_gap"

        for edge
        in physical_edges
    )

    vertical_count = sum(
        edge.edge_kind
        ==
        "vertical"

        for edge
        in physical_edges
    )

    lateral_rr = 0
    lateral_rf = 0
    lateral_ff = 0

    vertical_rr = 0
    vertical_filler = 0

    for edge in physical_edges:

        kind_a = (
            geometries[
                edge.a
            ].kind
        )

        kind_b = (
            geometries[
                edge.b
            ].kind
        )

        if edge.edge_kind in (
            "direct_lateral",
            "recovered_filler_gap",
        ):

            if (
                kind_a == "real"
                and kind_b == "real"
            ):
                lateral_rr += 1

            elif (
                kind_a == "filler"
                and kind_b == "filler"
            ):
                lateral_ff += 1

            else:
                lateral_rf += 1

        elif (
            edge.edge_kind
            ==
            "vertical"
        ):

            if (
                kind_a == "real"
                and kind_b == "real"
            ):
                vertical_rr += 1

            else:
                vertical_filler += 1

    # ==========================================================
    # PYTORCH GEOMETRIC:
    # STORE BOTH DIRECTIONS
    # ==========================================================

    sources = []
    destinations = []
    attributes = []

    for edge in physical_edges:

        sources.extend(
            [
                edge.a,
                edge.b,
            ]
        )

        destinations.extend(
            [
                edge.b,
                edge.a,
            ]
        )

        attributes.extend(
            [
                edge.attr,
                edge.attr,
            ]
        )

    edge_index = torch.tensor(
        [
            sources,
            destinations,
        ],
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        attributes,
        dtype=torch.float32,
    )

    stats = {
        "direct_lateral_edge_count":
        direct_lateral,

        "recovered_filler_gap_edge_count":
        recovered_lateral,

        "lateral_edge_count":
        (
            direct_lateral
            +
            recovered_lateral
        ),

        "vertical_edge_count":
        vertical_count,

        "directed_edge_count":
        edge_index.shape[1],

        "lateral_real_real_count":
        lateral_rr,

        "lateral_real_filler_count":
        lateral_rf,

        "lateral_filler_filler_count":
        lateral_ff,

        "vertical_real_real_count":
        vertical_rr,

        "vertical_involving_filler_count":
        vertical_filler,

        "connected_component_count":
        len(
            components
        ),

        "connected_component_sizes":
        [
            len(c)
            for c
            in components
        ],

        "isolated_node_count":
        len(
            isolated
        ),

        "isolated_nodes":
        isolated,
    }

    return (
        edge_index,
        edge_attr,
        stats,
    )


# ==============================================================
# GLOBAL FEATURES
# ==============================================================

def build_global_features(
    sim,
    aux,
):
    """
    FINAL GLOBAL FEATURE CONTRACT = 27 FEATURES

    Features 25 and 26 are the NEW full-chip dimensions.
    """

    values = [
        # 0
        float_attr(
            sim,
            "ambient_temp_K",
            300.0,
        ),

        # 1
        float_attr(
            sim,
            "workload_factor",
            1.0,
        ),

        # 2
        float_attr(
            sim,
            "total_power_W",
            0.0,
        ),

        # 3
        float(
            aux[
                "n_layers"
            ]
        ),

        # 4
        float(
            int_attr(
                sim,
                "n_tsv_pairs",
                0,
            )
        ),

        # 5
        float(
            int_attr(
                sim,
                "has_hotspot",
                0,
            )
        ),

        # 6
        float_attr(
            sim,
            "hotspot_multiplier",
            1.0,
        ),

        # 7
        float(
            int_attr(
                sim,
                "cooling_type_code",
                0,
            )
        ),

        # 8
        float_attr(
            sim,
            "cooling_htc_w_m2k",
            0.0,
        ),

        # 9
        float(
            int_attr(
                sim,
                "package_recipe_code",
                0,
            )
        ),

        # 10
        float(
            int_attr(
                sim,
                "non_uniform_material",
                0,
            )
        ),

        # 11
        float(
            aux[
                "bond_kappa_w_mk"
            ]
        ),

        # 12
        float(
            aux[
                "bond_thickness_mm"
            ]
        ),

        # 13
        float(
            int_attr(
                sim,
                "has_rdl",
                0,
            )
        ),

        # 14
        float_attr(
            sim,
            "rdl_thickness_um",
            0.0,
        ) / 1000.0,

        # 15
        float_attr(
            sim,
            "rdl_kappa_w_mk",
            0.0,
        ),

        # 16
        float(
            int_attr(
                sim,
                "has_interposer",
                0,
            )
        ),

        # 17
        float_attr(
            sim,
            "interposer_thickness_um",
            0.0,
        ) / 1000.0,

        # 18
        float_attr(
            sim,
            "interposer_kappa_w_mk",
            0.0,
        ),

        # 19
        float(
            int_attr(
                sim,
                "has_c4",
                0,
            )
        ),

        # 20
        float_attr(
            sim,
            "c4_thickness_um",
            0.0,
        ) / 1000.0,

        # 21
        float_attr(
            sim,
            "c4_kappa_w_mk",
            0.0,
        ),

        # 22
        float(
            int_attr(
                sim,
                "has_package_substrate",
                0,
            )
        ),

        # 23
        float_attr(
            sim,
            "substrate_thickness_um",
            0.0,
        ) / 1000.0,

        # 24
        float_attr(
            sim,
            "substrate_kappa_w_mk",
            0.0,
        ),

        # ======================================================
        # NEW FULL CHIP / LAYER FOOTPRINT
        # ======================================================

        # 25
        float(
            aux[
                "chip_width_mm"
            ]
        ),

        # 26
        float(
            aux[
                "chip_height_mm"
            ]
        ),
    ]

    if (
        len(values)
        !=
        len(
            GLOBAL_FEATURE_NAMES
        )
    ):

        raise RuntimeError(
            "Global feature count mismatch: "
            f"values={len(values)}, "
            f"names={len(GLOBAL_FEATURE_NAMES)}"
        )

    return torch.tensor(
        values,
        dtype=torch.float32,
    ).view(
        1,
        -1,
    )


# ==============================================================
# TOP-SURFACE PARTITION QA
# ==============================================================

def rectangle_union_area_clipped(rectangles, chip_width_mm, chip_height_mm):
    W=float(chip_width_mm); H=float(chip_height_mm)
    clipped=[]
    for x0,x1,y0,y1 in rectangles:
        x0=max(0.0,min(W,float(x0))); x1=max(0.0,min(W,float(x1)))
        y0=max(0.0,min(H,float(y0))); y1=max(0.0,min(H,float(y1)))
        if x1>x0 and y1>y0: clipped.append((x0,x1,y0,y1))
    if not clipped: return 0.0,0.0
    xs=sorted(set([r[0] for r in clipped]+[r[1] for r in clipped]))
    union=0.0
    for xa,xb in zip(xs[:-1],xs[1:]):
        if xb<=xa: continue
        ints=[(y0,y1) for x0,x1,y0,y1 in clipped if x0<xb and x1>xa]
        if not ints: continue
        ints.sort(); s,e=ints[0]; yu=0.0
        for a,b in ints[1:]:
            if a<=e: e=max(e,b)
            else: yu += e-s; s,e=a,b
        yu += e-s
        union += (xb-xa)*yu
    summed=sum((x1-x0)*(y1-y0) for x0,x1,y0,y1 in clipped)
    return float(union), float(summed)

def evaluate_top_surface_partition(x, n_layers, chip_width_mm, chip_height_mm):
    top_layer=int(n_layers-1)
    layers=torch.round(x[:,2]).long()
    ids=torch.where(layers==top_layer)[0]
    rects=[]
    for t in ids:
        i=int(t.item()); cx=float(x[i,7]); cy=float(x[i,8]); w=float(x[i,10]); h=float(x[i,11])
        rects.append((cx-w/2,cx+w/2,cy-h/2,cy+h/2))
    union,summed=rectangle_union_area_clipped(rects,chip_width_mm,chip_height_mm)
    chip=float(chip_width_mm)*float(chip_height_mm)
    if chip<=EPS: raise ValueError('Invalid chip area in top-surface QA')
    cov=union/chip; ov=max(0.0,summed-union); ov_ratio=ov/chip; unc=max(0.0,chip-union)
    cov_tol=1e-4; ov_tol=1e-4
    ok=(abs(cov-1.0)<=cov_tol and ov_ratio<=ov_tol)
    return {
        'top_layer':top_layer,'top_node_ids':ids,'chip_top_area_mm2':chip,
        'top_union_area_mm2':union,'top_summed_footprint_area_mm2':summed,
        'top_overlap_area_mm2':ov,'top_uncovered_area_mm2':unc,
        'top_coverage_ratio':cov,'top_overlap_ratio':ov_ratio,
        'top_partition_ok':bool(ok),'coverage_tolerance':cov_tol,'overlap_tolerance':ov_tol,
    }

# ==============================================================
# PINN BOUNDARY / RAW PHYSICS HELPERS
# ==============================================================

def _read_node_vector_dataset(
    sim,
    names,
    n_real,
    total_nodes,
    dtype=np.float32,
):
    """
    Read a future authoritative node-level vector if present.

    Accepted lengths:
      n_real      -> filler values are padded with zero
      total_nodes -> used directly

    Returns:
      (tensor_or_none, source_name_or_none)
    """

    for name in names:
        if name not in sim:
            continue

        values = np.asarray(
            sim[name][:],
            dtype=dtype,
        ).reshape(-1)

        if len(values) == n_real:
            padded = np.zeros(
                total_nodes,
                dtype=dtype,
            )
            padded[:n_real] = values
            return torch.from_numpy(padded), name

        if len(values) == total_nodes:
            return torch.from_numpy(values.copy()), name

        raise ValueError(
            f"{sim.name}: boundary dataset '{name}' has length {len(values)}; "
            f"expected n_real={n_real} or total_nodes={total_nodes}"
        )

    return None, None


def build_boundary_physics(
    sim,
    x,
    n_real,
    n_layers,
):
    """
    Build boundary quantities from the most authoritative source available.

    Priority:
      1) exact node-level exposed-area dataset in H5
      2) exact node-level boundary mask in H5
      3) explicit cooling-face attribute in H5
      4) CURRENT DATASET FALLBACK FROM ORIGINAL GENERATOR:
         generate_thermal_dataset_v3.py writes `top heat sink:`
         using cooling_htc_w_m2k and AMBIENT_TEMP_K.

    Thus, for the current dataset, the highest-Z layer (n_layers-1)
    is the convective external top surface. Each node on that layer
    contributes one horizontal exposed face equal to its footprint area.
    """

    total_nodes = int(x.shape[0])

    area = x[:, 1].to(
        dtype=torch.float32
    )

    layer = torch.round(
        x[:, 2]
    ).to(
        dtype=torch.long
    )

    top_candidate = torch.where(
        layer == int(n_layers - 1),
        area,
        torch.zeros_like(area),
    )

    bottom_candidate = torch.where(
        layer == 0,
        area,
        torch.zeros_like(area),
    )

    cooling_area = torch.zeros_like(
        area
    )

    cooling_mask = torch.zeros(
        total_nodes,
        dtype=torch.bool,
    )

    mapping_available = False
    mapping_source = (
        "pending_authoritative_mapping_resolution"
    )

    # ----------------------------------------------------------
    # Preferred future contract: exact cooling area per node.
    # ----------------------------------------------------------

    area_tensor, area_source = (
        _read_node_vector_dataset(
            sim,
            [
                "cooling_exposed_area_mm2",
                "boundary_exposed_area_mm2",
                "exposed_area_mm2",
            ],
            n_real,
            total_nodes,
            dtype=np.float32,
        )
    )

    if area_tensor is not None:

        if torch.any(
            area_tensor < 0
        ):
            raise ValueError(
                f"{sim.name}: negative exposed boundary area"
            )

        cooling_area = area_tensor
        cooling_mask = (
            cooling_area > 0
        )
        mapping_available = True
        mapping_source = (
            f"dataset:{area_source}"
        )

    else:
        # ------------------------------------------------------
        # Second-best future contract: explicit node boundary mask.
        # If area is not separately stored, node footprint area is
        # used for a z-normal external face ONLY for nodes selected
        # by that authoritative mask.
        # ------------------------------------------------------

        mask_tensor, mask_source = (
            _read_node_vector_dataset(
                sim,
                [
                    "cooling_boundary_mask",
                    "boundary_mask",
                    "surface_mask",
                ],
                n_real,
                total_nodes,
                dtype=np.float32,
            )
        )

        if mask_tensor is not None:

            cooling_mask = (
                mask_tensor > 0.5
            )

            # For an explicit node mask, use one horizontal face area.
            cooling_area = torch.where(
                cooling_mask,
                area,
                torch.zeros_like(area),
            )

            mapping_available = True
            mapping_source = (
                f"dataset:{mask_source}"
            )

        else:
            # --------------------------------------------------
            # Future-compatible explicit face attribute.
            # Current inspected H5 has none of these.
            # --------------------------------------------------

            face_value = None
            face_name = None

            for candidate in (
                "cooling_face",
                "boundary_face",
                "cooling_surface",
                "boundary_surface",
            ):

                if candidate in sim.attrs:
                    face_value = str(
                        decode_value(
                            sim.attrs[candidate]
                        )
                    ).strip().lower()
                    face_name = candidate
                    break

            if face_value is not None:

                normalized = (
                    face_value
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                if normalized in (
                    "top",
                    "top_face",
                    "top_surface",
                ):
                    cooling_area = (
                        top_candidate.clone()
                    )

                elif normalized in (
                    "bottom",
                    "bottom_face",
                    "bottom_surface",
                ):
                    cooling_area = (
                        bottom_candidate.clone()
                    )

                elif normalized in (
                    "top_and_bottom",
                    "both",
                    "both_faces",
                    "top_bottom",
                ):
                    cooling_area = (
                        top_candidate
                        +
                        bottom_candidate
                    )

                else:
                    raise ValueError(
                        f"{sim.name}: unsupported explicit {face_name}="
                        f"{face_value!r}; refusing to guess"
                    )

                cooling_mask = (
                    cooling_area > 0
                )

                mapping_available = True
                mapping_source = (
                    f"attr:{face_name}={face_value}"
                )

            else:
                # --------------------------------------------------
                # Authoritative recovery from the original generator:
                # generate_thermal_dataset_v3.py writes:
                #
                #     top heat sink:
                #       heat transfer coefficient ...
                #       temperature AMBIENT_TEMP_K
                #
                # The H5 lost the face label during dataset assembly,
                # but the solver-generation code establishes the face.
                # Highest layer index is highest Z in this graph builder.
                # --------------------------------------------------

                cooling_area = (
                    top_candidate.clone()
                )

                cooling_mask = (
                    cooling_area > 0
                )

                mapping_available = True

                mapping_source = (
                    "generator:generate_thermal_dataset_v3.py:"
                    "top_heat_sink"
                )

    htc = float_attr(
        sim,
        "cooling_htc_w_m2k",
        0.0,
    )

    if htc < 0:
        raise ValueError(
            f"{sim.name}: cooling_htc_w_m2k < 0"
        )

    # h [W/m^2/K] * A [mm^2] * 1e-6 [m^2/mm^2] = W/K
    G_amb = (
        float(htc)
        *
        cooling_area
        *
        1.0e-6
    )

    return {
        "top_surface_candidate_area_mm2":
        top_candidate,

        "bottom_surface_candidate_area_mm2":
        bottom_candidate,

        "cooling_exposed_area_mm2":
        cooling_area,

        "cooling_boundary_mask":
        cooling_mask,

        "G_amb_W_K":
        G_amb,

        "boundary_mapping_available":
        bool(mapping_available),

        "boundary_mapping_source":
        str(mapping_source),

        "cooling_htc_w_m2k":
        float(htc),
    }


# ==============================================================
# BUILD ONE GRAPH
# ==============================================================

def build_graph_from_group(
    sim,
    hdf,
    contact_tolerance,
    max_recovery_gap,
):

    sim_id = str_attr(
        sim,
        "sim_id",
        sim.name.split("/")[-1],
    )

    (
        x,
        y,
        pos,
        target_mask,
        real_mask,
        filler_mask,
        filler_temperature_mask,
        geometries,
        aux,
    ) = build_nodes(
        sim,
        hdf,
    )

    try:

        (
            edge_index,
            edge_attr,
            stats,
        ) = build_edges(
            geometries,

            filler_kappa=float(
                aux[
                    "filler_kappa_w_mk"
                ]
            ),

            bond_kappa=float(
                aux[
                    "bond_kappa_w_mk"
                ]
            ),

            bond_thickness=float(
                aux[
                    "bond_thickness_mm"
                ]
            ),

            contact_tolerance=
            contact_tolerance,

            max_recovery_gap=
            max_recovery_gap,
        )

    except Exception as exc:

        raise ValueError(
            f"{sim_id}: {exc}"
        ) from exc

    n_real = int(
        aux[
            "n_real"
        ]
    )

    n_filler = int(
        aux[
            "n_filler"
        ]
    )

    global_features = (
        build_global_features(
            sim,
            aux,
        )
    )

    # ==========================================================
    # PINN RAW PHYSICS TENSORS
    # ==========================================================

    # ----------------------------------------------------------
    # AUDITED SOURCE-POWER SEMANTICS
    # ----------------------------------------------------------
    #
    # Historical V3 generator evidence:
    #   block feature column 0 was written DIRECTLY to 3D-ICE
    #   `power values`.
    #
    # Therefore the EXISTING V3 FEA labels were generated with:
    #   P_real = x[:,0] numerically interpreted as W
    # not with:
    #   P_real = x[:,0] * area
    #
    # Legacy V2 power semantics have not been independently proven,
    # so they are deliberately excluded from physics.
    data_source_for_physics = str_attr(
        sim,
        "data_source",
        "",
    )

    physics_power_semantics_verified = bool(
        data_source_for_physics
        ==
        "3dice_real_fea_v3_self_contained"
    )

    physics_node_power_W = torch.zeros(
        x.shape[0],
        dtype=torch.float32,
    )

    if physics_power_semantics_verified:
        physics_node_power_W[:n_real] = (
            x[:n_real, 0]
            .clone()
            .to(dtype=torch.float32)
        )

    # `physics_mask` means the source semantics are verified for this graph.
    # It does NOT imply exact node-level KCL readiness.
    physics_mask = torch.zeros(
        x.shape[0],
        dtype=torch.bool,
    )

    if physics_power_semantics_verified:
        physics_mask[:] = True

    # Preserve the old HDF5 convention for traceability only.
    legacy_reconstructed_power_W = float(
        (
            x[:n_real, 0]
            *
            x[:n_real, 1]
        ).sum()
    )

    solver_input_total_power_W = float(
        physics_node_power_W[
            :n_real
        ].sum()
    )

    ground_truth_temperature_K = (
        y.clone().to(
            dtype=torch.float32
        )
    )

    ground_truth_temperature_mask = (
        target_mask
        |
        filler_temperature_mask
    )

    boundary = build_boundary_physics(
        sim,
        x,
        n_real=n_real,
        n_layers=int(
            aux["n_layers"]
        ),
    )

    top_partition = evaluate_top_surface_partition(
        x=x,
        n_layers=int(aux["n_layers"]),
        chip_width_mm=float(aux["chip_width_mm"]),
        chip_height_mm=float(aux["chip_height_mm"]),
    )

    chip_top_area_mm2 = float(top_partition["chip_top_area_mm2"])
    boundary_geometry_ready = bool(top_partition["top_partition_ok"])
    exact_filler_geometry_cohort = bool(
        int(aux["filler_blocks_width"]) == 7
        and str(aux["filler_geometry_source"]) == "filler_blocks_7"
    )
    full_boundary_eligible = bool(
        boundary["boundary_mapping_available"]
        and boundary_geometry_ready
        and exact_filler_geometry_cohort
    )

    if not full_boundary_eligible:
        boundary["cooling_exposed_area_mm2"] = torch.zeros_like(boundary["cooling_exposed_area_mm2"])
        boundary["cooling_boundary_mask"] = torch.zeros_like(boundary["cooling_boundary_mask"])
        boundary["G_amb_W_K"] = torch.zeros_like(boundary["G_amb_W_K"])

    cooling_exposed_area_sum_mm2 = float(boundary["cooling_exposed_area_mm2"].sum())
    cooling_area_coverage_ratio = float(top_partition["top_coverage_ratio"])

    # Node feature 12 is non-zero only when boundary geometry is trustworthy.
    x[:, 12] = boundary["cooling_exposed_area_mm2"]

    raw_edge_resistance_K_W = (
        edge_attr[:, 0].clone()
    )

    raw_edge_conductance_W_K = (
        edge_attr[:, 6].clone()
    )

    # Edges are stored as (a->b, b->a) consecutively.
    # Pair IDs make it harder to accidentally double count them
    # in global physics diagnostics.
    edge_pair_id = torch.arange(
        edge_index.shape[1] // 2,
        dtype=torch.long,
    ).repeat_interleave(2)

    graph = Data(
        x=x,

        edge_index=
        edge_index,

        edge_attr=
        edge_attr,

        y=y,

        pos=pos,

        target_mask=
        target_mask,

        real_node_mask=
        real_mask,

        filler_node_mask=
        filler_mask,

        filler_temperature_mask=
        filler_temperature_mask,

        filler_y=
        y[
            n_real:
        ].clone(),

        real_y=
        y[
            :n_real
        ].clone(),

        global_features=
        global_features,

        # ------------------------------------------------------
        # Raw physics tensors for PINN / residual QA
        # ------------------------------------------------------
        physics_node_power_W=
        physics_node_power_W,

        physics_mask=
        physics_mask,

        ground_truth_temperature_K=
        ground_truth_temperature_K,

        ground_truth_temperature_mask=
        ground_truth_temperature_mask,

        edge_resistance_K_W=
        raw_edge_resistance_K_W,

        edge_conductance_W_K=
        raw_edge_conductance_W_K,

        edge_pair_id=
        edge_pair_id,

        top_surface_candidate_area_mm2=
        boundary[
            "top_surface_candidate_area_mm2"
        ],

        bottom_surface_candidate_area_mm2=
        boundary[
            "bottom_surface_candidate_area_mm2"
        ],

        cooling_exposed_area_mm2=
        boundary[
            "cooling_exposed_area_mm2"
        ],

        cooling_boundary_mask=
        boundary[
            "cooling_boundary_mask"
        ],

        G_amb_W_K=
        boundary[
            "G_amb_W_K"
        ],
    )

    # ==========================================================
    # GENERAL METADATA
    # ==========================================================

    graph.sim_id = sim_id

    graph.temperature_unit = "K"
    graph.geometry_unit = "mm"
    graph.area_unit = "mm^2"

    graph.kappa_unit = "W/(m*K)"
    graph.resistance_unit = "K/W"
    graph.conductance_unit = "W/K"
    graph.power_unit = "W"
    graph.htc_unit = "W/(m^2*K)"
    graph.boundary_area_unit = "mm^2"

    graph.physics_contract_version = (
        "PINN_V4_1_final_audited_global_energy"
    )

    # Current historical H5 does not preserve the raw xyaxis cell
    # width/height needed to reconstruct the exact 3D-ICE cell operator.
    graph.solver_cell_geometry_available = False

    graph.physics_power_semantics_verified = bool(
        physics_power_semantics_verified
    )

    graph.physics_power_semantics = (
        "V3 historical generator: raw block feature column 0 "
        "was written directly to 3D-ICE power values [W]"
        if graph.physics_power_semantics_verified
        else
        "UNVERIFIED: excluded from physics"
    )

    # Keep the original ML feature name/value for backward compatibility,
    # while making its historical solver semantics explicit.
    graph.node_feature_0_declared_name = (
        "power_density_W_mm2"
    )
    graph.node_feature_0_historical_v3_semantics = (
        "legacy/misnamed numeric field; historical V3 generator passed "
        "this value directly to 3D-ICE power values [W]"
    )
    graph.node_feature_0_ml_contract_preserved = True

    # `conduction_physics_ready` now means exact hard node-KCL readiness.
    # It is intentionally FALSE for this historical coarse graph.
    graph.conduction_physics_ready = False

    graph.boundary_mapping_available = (
        bool(
            boundary[
                "boundary_mapping_available"
            ]
        )
    )

    graph.boundary_mapping_source = (
        str(
            boundary[
                "boundary_mapping_source"
            ]
        )
    )

    graph.boundary_geometry_ready = bool(boundary_geometry_ready)
    graph.exact_filler_geometry_cohort = bool(exact_filler_geometry_cohort)
    graph.boundary_physics_ready = bool(
        full_boundary_eligible
        and
        boundary["cooling_htc_w_m2k"] > 0
    )

    # Hard node-by-node KCL is deliberately disabled for the historical
    # coarse graph, even when the boundary mapping is geometrically valid.
    graph.physics_residual_ready = False
    graph.pinn_full_physics_eligible = False

    # `pinn_global_energy_eligible` is computed later, after the stored
    # ground-truth temperatures are audited against the corrected source.
    graph.pinn_global_energy_eligible = False

    graph.physics_loss_scope = (
        "diagnostic_coarse_node_kcl_plus_optional_global_energy"
    )
    graph.cooling_face = "top"
    graph.chip_top_area_mm2 = float(chip_top_area_mm2)
    graph.top_union_area_mm2 = float(top_partition["top_union_area_mm2"])
    graph.top_summed_footprint_area_mm2 = float(top_partition["top_summed_footprint_area_mm2"])
    graph.top_overlap_area_mm2 = float(top_partition["top_overlap_area_mm2"])
    graph.top_uncovered_area_mm2 = float(top_partition["top_uncovered_area_mm2"])
    graph.cooling_exposed_area_sum_mm2 = float(cooling_exposed_area_sum_mm2)
    graph.cooling_area_coverage_ratio = float(cooling_area_coverage_ratio)
    graph.top_overlap_ratio = float(top_partition["top_overlap_ratio"])
    graph.cooling_area_coverage_warning = (
        ""
        if graph.boundary_physics_ready
        else
        "Boundary tensors disabled: top partition failed exact geometry QA or graph is legacy F,5."
    )

    graph.cooling_htc_w_m2k_raw = (
        float(
            boundary[
                "cooling_htc_w_m2k"
            ]
        )
    )

    graph.ambient_temp_K_raw = (
        float_attr(
            sim,
            "ambient_temp_K",
            300.0,
        )
    )

    graph.physics_residual_definition = (
        "r_i = P_i + sum_j G_ij*(T_j-T_i) "
        "- G_amb_i*(T_i-T_ambient)"
    )

    # ==========================================================
    # V4 GROUND-TRUTH PHYSICS AUDIT
    # ==========================================================
    #
    # Node/layer residuals are DIAGNOSTIC ONLY because the current
    # block/filler graph is a coarse representation of the 3D-ICE
    # cell network and the historical H5 no longer preserves raw
    # solver-cell widths/heights.
    #
    # The only training-oriented physics gate exposed by V4 is an
    # empirical GLOBAL energy-balance gate.
    src_phys = graph.edge_index[0]
    dst_phys = graph.edge_index[1]

    T_gt_phys = graph.ground_truth_temperature_K
    G_edge_phys = graph.edge_conductance_W_K

    q_edge_into_dst = (
        G_edge_phys
        *
        (
            T_gt_phys[src_phys]
            -
            T_gt_phys[dst_phys]
        )
    )

    Q_internal_gt_W = torch.zeros_like(
        T_gt_phys
    )

    Q_internal_gt_W.index_add_(
        0,
        dst_phys,
        q_edge_into_dst,
    )

    Q_ambient_gt_W = (
        graph.G_amb_W_K
        *
        (
            T_gt_phys
            -
            float(graph.ambient_temp_K_raw)
        )
    )

    gt_node_residual_W = (
        graph.physics_node_power_W
        +
        Q_internal_gt_W
        -
        Q_ambient_gt_W
    )

    G_internal_sum = torch.zeros_like(
        T_gt_phys
    )

    G_internal_sum.index_add_(
        0,
        dst_phys,
        G_edge_phys,
    )

    G_reference = (
        G_internal_sum
        +
        graph.G_amb_W_K
    ).clamp(min=1.0e-12)

    gt_node_residual_Kref = (
        gt_node_residual_W
        /
        G_reference
    )

    graph.gt_node_residual_W_rmse = float(
        torch.sqrt(
            torch.mean(
                gt_node_residual_W
                *
                gt_node_residual_W
            )
        )
    )

    graph.gt_node_residual_Kref_rmse = float(
        torch.sqrt(
            torch.mean(
                gt_node_residual_Kref
                *
                gt_node_residual_Kref
            )
        )
    )

    if torch.any(graph.real_node_mask):
        graph.gt_real_node_residual_Kref_rmse = float(
            torch.sqrt(
                torch.mean(
                    gt_node_residual_Kref[
                        graph.real_node_mask
                    ]
                    **
                    2
                )
            )
        )
    else:
        graph.gt_real_node_residual_Kref_rmse = float("nan")

    uncooled_filler_mask = (
        graph.filler_node_mask
        &
        (
            graph.G_amb_W_K
            <= 0
        )
    )

    if torch.any(uncooled_filler_mask):
        graph.gt_uncooled_filler_residual_Kref_rmse = float(
            torch.sqrt(
                torch.mean(
                    gt_node_residual_Kref[
                        uncooled_filler_mask
                    ]
                    **
                    2
                )
            )
        )
    else:
        graph.gt_uncooled_filler_residual_Kref_rmse = float("nan")

    graph.gt_total_source_power_W = float(
        graph.physics_node_power_W.sum()
    )

    graph.gt_total_ambient_removal_W = float(
        Q_ambient_gt_W.sum()
    )

    graph.gt_global_energy_residual_W = float(
        graph.gt_total_source_power_W
        -
        graph.gt_total_ambient_removal_W
    )

    graph.gt_coarse_kcl_diagnostic_evaluated = bool(
        graph.physics_power_semantics_verified
    )

    graph.gt_global_energy_audit_evaluated = bool(
        graph.physics_power_semantics_verified
        and
        graph.boundary_physics_ready
        and
        graph.gt_total_source_power_W > 1.0e-12
    )

    if graph.gt_global_energy_audit_evaluated:
        graph.gt_global_energy_closure_error_pct = float(
            100.0
            *
            abs(
                graph.gt_global_energy_residual_W
            )
            /
            graph.gt_total_source_power_W
        )
    else:
        graph.gt_global_energy_closure_error_pct = float("inf")

    # Layer residuals are stored only to prove/track that layer-wise
    # conservation is NOT a safe training constraint on this coarse graph.
    layer_ids_phys = torch.round(
        graph.x[:, 2]
    ).long()

    layer_abs_residual_pct = []

    # n_layers is already authoritative in aux at this point.
    # graph.n_layers is assigned later in the metadata section.
    n_layers_phys = int(
        aux[
            "n_layers"
        ]
    )

    for layer_id_phys in range(n_layers_phys):
        layer_mask_phys = (
            layer_ids_phys
            ==
            layer_id_phys
        )

        layer_residual_W = float(
            gt_node_residual_W[
                layer_mask_phys
            ].sum()
        )

        if graph.gt_total_source_power_W > 1.0e-12:
            layer_abs_residual_pct.append(
                100.0
                *
                abs(layer_residual_W)
                /
                graph.gt_total_source_power_W
            )

    if layer_abs_residual_pct:
        graph.gt_mean_layer_abs_residual_pct_total_power = float(
            sum(layer_abs_residual_pct)
            /
            len(layer_abs_residual_pct)
        )
        graph.gt_max_layer_abs_residual_pct_total_power = float(
            max(layer_abs_residual_pct)
        )
    else:
        graph.gt_mean_layer_abs_residual_pct_total_power = float("inf")
        graph.gt_max_layer_abs_residual_pct_total_power = float("inf")

    # Candidate status uses INPUT/PROVENANCE information only.
    graph.global_energy_constraint_candidate = bool(
        graph.physics_power_semantics_verified
        and
        graph.boundary_physics_ready
    )

    # Ground-truth audit is a DATASET-QA decision, not a model feature.
    # It must never be fed to the network as an input.
    graph.global_energy_closure_gate_pct = float(
        GLOBAL_ENERGY_CLOSURE_GATE_PCT
    )

    graph.gt_global_energy_audit_pass = bool(
        graph.gt_global_energy_audit_evaluated
        and
        graph.gt_global_energy_closure_error_pct
        <=
        graph.global_energy_closure_gate_pct
    )

    # Backward-friendly convenience alias for Student 3.
    # IMPORTANT: this mask is target-derived QA and must not be an input feature.
    graph.pinn_global_energy_eligible = bool(
        graph.global_energy_constraint_candidate
        and
        graph.gt_global_energy_audit_pass
    )

    graph.pinn_global_energy_eligibility_is_target_derived_qa = True

    graph.global_energy_loss_definition = (
        "r_global = sum_i(P_i) - "
        "sum_i[G_amb_i*(T_i-T_ambient)]"
    )

    # Do not expose numeric coarse-KCL diagnostics as meaningful for cohorts
    # whose historical solver-power semantics have not been proven.
    if not graph.physics_power_semantics_verified:
        graph.gt_node_residual_W_rmse = float("nan")
        graph.gt_node_residual_Kref_rmse = float("nan")
        graph.gt_real_node_residual_Kref_rmse = float("nan")
        graph.gt_uncooled_filler_residual_Kref_rmse = float("nan")
        graph.gt_mean_layer_abs_residual_pct_total_power = float("inf")
        graph.gt_max_layer_abs_residual_pct_total_power = float("inf")

    graph.boundary_warning = (
        "" if graph.boundary_physics_ready else
        "Top heat-sink face is known, but node-level convection is disabled because the top partition is not exact or the graph is legacy F,5."
    )

    # ==========================================================
    # NODE COUNTS
    # ==========================================================

    graph.real_node_count = (
        n_real
    )

    graph.filler_node_count = (
        n_filler
    )

    graph.total_node_count = (
        n_real
        +
        n_filler
    )

    graph.target_node_count = (
        int(
            target_mask.sum()
        )
    )

    # ==========================================================
    # MIXED-SCHEMA DIAGNOSTICS
    # ==========================================================

    graph.block_features_width = 7

    graph.block_features_merged_width = (
        int(
            aux[
                "block_features_merged_width"
            ]
        )
    )

    graph.real_geometry_source = (
        str(
            aux[
                "real_geometry_source"
            ]
        )
    )

    graph.filler_blocks_width = (
        int(
            aux[
                "filler_blocks_width"
            ]
        )
    )

    graph.filler_geometry_source = (
        str(
            aux[
                "filler_geometry_source"
            ]
        )
    )

    # ==========================================================
    # EDGE STATS
    # ==========================================================

    for name, value in (
        stats.items()
    ):

        if isinstance(
            value,
            list,
        ):

            setattr(
                graph,
                name,

                torch.tensor(
                    value,
                    dtype=torch.long,
                ),
            )

        else:

            setattr(
                graph,
                name,
                value,
            )

    # ==========================================================
    # MATERIAL
    # ==========================================================

    graph.filler_kappa_W_mK = (
        float(
            aux[
                "filler_kappa_w_mk"
            ]
        )
    )

    graph.bond_kappa_W_mK = (
        float(
            aux[
                "bond_kappa_w_mk"
            ]
        )
    )

    graph.bond_thickness_mm = (
        float(
            aux[
                "bond_thickness_mm"
            ]
        )
    )

    graph.layer_thicknesses_mm = (
        torch.tensor(
            aux[
                "layer_thicknesses_mm"
            ],
            dtype=torch.float32,
        )
    )

    graph.layer_z_mm = (
        torch.tensor(
            aux[
                "layer_z_mm"
            ],
            dtype=torch.float32,
        )
    )

    graph.n_layers = int(
        aux[
            "n_layers"
        ]
    )

    # ==========================================================
    # NEW FULL CHIP / LAYER DIMENSIONS
    # ==========================================================

    graph.chip_width_mm = (
        float(
            aux[
                "chip_width_mm"
            ]
        )
    )

    graph.chip_height_mm = (
        float(
            aux[
                "chip_height_mm"
            ]
        )
    )

    # Keep original HDF5-unit values as convenient metadata too.
    graph.chip_w_um = (
        graph.chip_width_mm
        *
        1000.0
    )

    graph.chip_h_um = (
        graph.chip_height_mm
        *
        1000.0
    )

    # ==========================================================
    # TSV
    # ==========================================================

    graph.n_tsv_pairs = (
        int_attr(
            sim,
            "n_tsv_pairs",
            0,
        )
    )

    # Preserve exact TSV-pad geometry when the source simulation
    # provides it. TSV pads are metadata only and are NOT added as
    # graph nodes, so the node-feature contract remains unchanged.
    if "tsv_pads" in sim:

        tsv_pads = np.asarray(
            sim["tsv_pads"][:],
            dtype=np.float32,
        )

        if (
            tsv_pads.ndim == 2
            and tsv_pads.shape[1] == 7
        ):

            graph.tsv_pads = (
                torch.from_numpy(
                    tsv_pads.copy()
                )
            )

            graph.tsv_pad_count = (
                int(
                    tsv_pads.shape[0]
                )
            )

            graph.tsv_exact_positions_available = (
                True
            )

        else:

            graph.tsv_pads = (
                torch.empty(
                    (0, 7),
                    dtype=torch.float32,
                )
            )

            graph.tsv_pad_count = 0

            graph.tsv_exact_positions_available = (
                False
            )

    else:

        graph.tsv_pads = (
            torch.empty(
                (0, 7),
                dtype=torch.float32,
            )
        )

        graph.tsv_pad_count = 0

        graph.tsv_exact_positions_available = (
            False
        )

    graph.tsv_topology_reconstructed = (
        False
    )

    graph.vertical_edges_are_geometric_overlap = (
        True
    )

    # ==========================================================
    # FILLER RECONSTRUCTION
    # ==========================================================

    graph.filler_geometry_reconstructed = (
        graph.filler_geometry_source
        !=
        "filler_blocks_7"
    )

    graph.filler_geometry_exact_replay = (
        graph.filler_geometry_source
        ==
        "filler_blocks_7"
    )

    graph.contact_tolerance_mm = (
        float(
            contact_tolerance
        )
    )

    graph.max_recovery_gap_mm = (
        -1.0

        if max_recovery_gap is None

        else float(
            max_recovery_gap
        )
    )

    # ==========================================================
    # ORIGINAL SIMULATION / PACKAGE METADATA
    # ==========================================================

    graph.package_recipe = (
        str_attr(
            sim,
            "package_recipe",
            "",
        )
    )

    graph.package_recipe_code = (
        int_attr(
            sim,
            "package_recipe_code",
            0,
        )
    )

    graph.cooling_type = (
        str_attr(
            sim,
            "cooling_type",
            "",
        )
    )

    graph.cooling_type_code = (
        int_attr(
            sim,
            "cooling_type_code",
            0,
        )
    )

    graph.data_source = (
        str_attr(
            sim,
            "data_source",
            "",
        )
    )

    graph.source_file = (
        str_attr(
            sim,
            "source_file",
            "",
        )
    )

    graph.ambient_temp_K = (
        float_attr(
            sim,
            "ambient_temp_K",
            300.0,
        )
    )

    graph.workload_factor = (
        float_attr(
            sim,
            "workload_factor",
            1.0,
        )
    )

    # Legacy HDF5 field retained for ML/backward compatibility.
    graph.total_power_W = (
        float_attr(
            sim,
            "total_power_W",
            0.0,
        )
    )

    graph.legacy_stored_total_power_W = float(
        graph.total_power_W
    )

    graph.legacy_reconstructed_total_power_W = float(
        legacy_reconstructed_power_W
    )

    graph.solver_input_total_power_W = float(
        solver_input_total_power_W
    )

    graph.global_features_total_power_semantics = (
        "LEGACY HDF5 total_power_W retained at global_features[2] for "
        "backward ML compatibility; historical V3 physics must use "
        "solver_input_total_power_W instead"
    )
    graph.global_feature_2_is_solver_power = False

    graph.max_temp_K = (
        float_attr(
            sim,
            "max_temp_K",
            0.0,
        )
    )

    graph.pass_fail = (
        int_attr(
            sim,
            "pass_fail",
            0,
        )
    )

    graph.non_uniform_material = (
        int_attr(
            sim,
            "non_uniform_material",
            0,
        )
    )

    graph.has_rdl = (
        int_attr(
            sim,
            "has_rdl",
            0,
        )
    )

    graph.rdl_thickness_mm = (
        float_attr(
            sim,
            "rdl_thickness_um",
            0.0,
        )
        /
        1000.0
    )

    graph.rdl_kappa_W_mK = (
        float_attr(
            sim,
            "rdl_kappa_w_mk",
            0.0,
        )
    )

    graph.has_interposer = (
        int_attr(
            sim,
            "has_interposer",
            0,
        )
    )

    graph.interposer_type = (
        str_attr(
            sim,
            "interposer_type",
            "",
        )
    )

    graph.interposer_thickness_mm = (
        float_attr(
            sim,
            "interposer_thickness_um",
            0.0,
        )
        /
        1000.0
    )

    graph.interposer_kappa_W_mK = (
        float_attr(
            sim,
            "interposer_kappa_w_mk",
            0.0,
        )
    )

    graph.has_c4 = (
        int_attr(
            sim,
            "has_c4",
            0,
        )
    )

    graph.c4_thickness_mm = (
        float_attr(
            sim,
            "c4_thickness_um",
            0.0,
        )
        /
        1000.0
    )

    graph.c4_kappa_W_mK = (
        float_attr(
            sim,
            "c4_kappa_w_mk",
            0.0,
        )
    )

    graph.has_package_substrate = (
        int_attr(
            sim,
            "has_package_substrate",
            0,
        )
    )

    graph.substrate_thickness_mm = (
        float_attr(
            sim,
            "substrate_thickness_um",
            0.0,
        )
        /
        1000.0
    )

    graph.substrate_kappa_W_mK = (
        float_attr(
            sim,
            "substrate_kappa_w_mk",
            0.0,
        )
    )

    graph.has_point_cloud_in_source = (
        "point_cloud"
        in sim
    )

    graph.filler_temperature_source = (
        "point_cloud_xy_mean_nearest_z_plane"
        if n_filler > 0
        else "none_no_filler_nodes"
    )

    graph.filler_temperature_count = int(
        filler_temperature_mask.sum()
    )

    return graph


# ==============================================================
# QA
# ==============================================================

def validate_graph(
    graph,
):

    sim_id = graph.sim_id

    # ----------------------------------------------------------
    # Node feature contract
    # ----------------------------------------------------------

    if (
        graph.x.ndim != 2
        or
        graph.x.shape[1]
        !=
        len(
            NODE_FEATURE_NAMES
        )
    ):

        raise ValueError(
            f"{sim_id}: bad x shape "
            f"{graph.x.shape}"
        )

    # ----------------------------------------------------------
    # Edge feature contract
    # ----------------------------------------------------------

    if (
        graph.edge_attr.ndim != 2
        or
        graph.edge_attr.shape[1]
        !=
        len(
            EDGE_FEATURE_NAMES
        )
    ):

        raise ValueError(
            f"{sim_id}: bad edge_attr "
            f"{graph.edge_attr.shape}"
        )

    # ----------------------------------------------------------
    # NEW global feature contract = 27
    # ----------------------------------------------------------

    if (
        graph.global_features.ndim
        != 2
        or
        graph.global_features.shape
        !=
        (
            1,
            len(
                GLOBAL_FEATURE_NAMES
            ),
        )
    ):

        raise ValueError(
            f"{sim_id}: bad global_features shape "
            f"{graph.global_features.shape}; "
            f"expected "
            f"(1,{len(GLOBAL_FEATURE_NAMES)})"
        )

    # ----------------------------------------------------------
    # Full chip dimensions
    # ----------------------------------------------------------

    if graph.chip_width_mm <= 0:

        raise ValueError(
            f"{sim_id}: "
            "chip_width_mm <= 0"
        )

    if graph.chip_height_mm <= 0:

        raise ValueError(
            f"{sim_id}: "
            "chip_height_mm <= 0"
        )

    if not math.isclose(
        float(
            graph.global_features[
                0,
                25,
            ]
        ),
        graph.chip_width_mm,
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):

        raise ValueError(
            f"{sim_id}: "
            "global chip width mismatch"
        )

    if not math.isclose(
        float(
            graph.global_features[
                0,
                26,
            ]
        ),
        graph.chip_height_mm,
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):

        raise ValueError(
            f"{sim_id}: "
            "global chip height mismatch"
        )

    # ----------------------------------------------------------
    # Edge index
    # ----------------------------------------------------------

    if (
        graph.edge_index.shape[0]
        != 2
    ):

        raise ValueError(
            f"{sim_id}: "
            "invalid edge_index"
        )

    if (
        graph.edge_index.shape[1]
        !=
        graph.edge_attr.shape[0]
    ):

        raise ValueError(
            f"{sim_id}: "
            "edge count mismatch"
        )

    # ----------------------------------------------------------
    # Targets
    # ----------------------------------------------------------

    if (
        len(
            graph.y
        )
        !=
        graph.x.shape[0]
    ):

        raise ValueError(
            f"{sim_id}: "
            "y node count mismatch"
        )

    if (
        int(
            graph.target_mask.sum()
        )
        !=
        graph.real_node_count
    ):

        raise ValueError(
            f"{sim_id}: "
            "target mask mismatch"
        )

    # ----------------------------------------------------------
    # Connectivity
    # ----------------------------------------------------------

    if (
        graph.connected_component_count
        != 1
    ):

        raise ValueError(
            f"{sim_id}: "
            "graph disconnected"
        )

    if (
        graph.isolated_node_count
        != 0
    ):

        raise ValueError(
            f"{sim_id}: "
            "isolated nodes"
        )

    # ----------------------------------------------------------
    # NaN / Inf
    # ----------------------------------------------------------

    for name, tensor in [
        (
            "x",
            graph.x,
        ),

        (
            "y",
            graph.y,
        ),

        (
            "pos",
            graph.pos,
        ),

        (
            "edge_attr",
            graph.edge_attr,
        ),

        (
            "global_features",
            graph.global_features,
        ),
    ]:

        if not torch.isfinite(
            tensor
        ).all():

            raise ValueError(
                f"{sim_id}: "
                f"{name} contains NaN/Inf"
            )

    # ----------------------------------------------------------
    # PINN raw physics contract
    # ----------------------------------------------------------

    n_nodes = graph.x.shape[0]

    for name in [
        "physics_node_power_W",
        "physics_mask",
        "ground_truth_temperature_K",
        "ground_truth_temperature_mask",
        "top_surface_candidate_area_mm2",
        "bottom_surface_candidate_area_mm2",
        "cooling_exposed_area_mm2",
        "cooling_boundary_mask",
        "G_amb_W_K",
    ]:
        tensor = getattr(
            graph,
            name,
        )

        if tensor.numel() != n_nodes:
            raise ValueError(
                f"{sim_id}: {name} length mismatch"
            )

    if not torch.isfinite(
        graph.physics_node_power_W
    ).all():
        raise ValueError(
            f"{sim_id}: non-finite physics_node_power_W"
        )

    if torch.any(
        graph.physics_node_power_W < -1e-8
    ):
        raise ValueError(
            f"{sim_id}: negative physics node power"
        )

    # ----------------------------------------------------------
    # V4 POWER CONTRACT VALIDATION
    # ----------------------------------------------------------
    physics_total_power = float(
        graph.physics_node_power_W[
            graph.real_node_mask
        ].sum()
    )

    if graph.physics_power_semantics_verified:
        if not math.isclose(
            physics_total_power,
            graph.solver_input_total_power_W,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"{sim_id}: solver-input physics power mismatch"
            )

        # Historical V3 solver source is the numeric raw col-0 value.
        expected_solver_power = float(
            graph.x[
                graph.real_node_mask,
                0
            ].sum()
        )

        if not math.isclose(
            physics_total_power,
            expected_solver_power,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{sim_id}: audited V3 physics power != sum(raw x[:,0])"
            )

    else:
        if torch.any(
            graph.physics_node_power_W != 0
        ):
            raise ValueError(
                f"{sim_id}: unverified source must not expose physics power"
            )

        if torch.any(
            graph.physics_mask
        ):
            raise ValueError(
                f"{sim_id}: unverified source must have physics_mask=False"
            )

    # Legacy HDF5 total_power_W is expected to match the historical
    # PD*area reconstruction.  This is metadata QA, NOT physics QA.
    if (
        graph.legacy_stored_total_power_W > 1e-8
        and
        not math.isclose(
            graph.legacy_reconstructed_total_power_W,
            graph.legacy_stored_total_power_W,
            rel_tol=2e-4,
            abs_tol=1e-4,
        )
    ):
        raise ValueError(
            f"{sim_id}: legacy total_power_W metadata mismatch "
            f"reconstructed={graph.legacy_reconstructed_total_power_W} W "
            f"stored={graph.legacy_stored_total_power_W} W"
        )

    if not torch.allclose(
        graph.edge_resistance_K_W,
        graph.edge_attr[:, 0],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"{sim_id}: raw edge resistance mismatch"
        )

    if not torch.allclose(
        graph.edge_conductance_W_K,
        graph.edge_attr[:, 6],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"{sim_id}: raw edge conductance mismatch"
        )

    expected_G_amb = (
        graph.cooling_htc_w_m2k_raw
        *
        graph.cooling_exposed_area_mm2
        *
        1.0e-6
    )

    if not torch.allclose(
        graph.G_amb_W_K,
        expected_G_amb,
        rtol=1e-5,
        atol=1e-10,
    ):
        raise ValueError(
            f"{sim_id}: G_amb != h*A"
        )

    if not graph.boundary_mapping_available:
        raise ValueError(
            f"{sim_id}: solver top-heat-sink mapping unavailable"
        )

    # V4 never exposes hard-node PINN eligibility on this historical
    # coarse graph.
    if graph.pinn_full_physics_eligible:
        raise ValueError(
            f"{sim_id}: V4 historical coarse graph must not be hard-PINN eligible"
        )

    if graph.boundary_physics_ready:
        if not math.isclose(
            graph.top_union_area_mm2,
            graph.chip_top_area_mm2,
            rel_tol=1e-4,
            abs_tol=1e-5,
        ):
            raise ValueError(
                f"{sim_id}: boundary-ready graph does not cover top surface"
            )

        if graph.top_overlap_ratio > 1e-4:
            raise ValueError(
                f"{sim_id}: boundary-ready graph has top overlap"
            )

        if not math.isclose(
            float(
                graph.cooling_exposed_area_mm2.sum()
            ),
            graph.chip_top_area_mm2,
            rel_tol=1e-4,
            abs_tol=1e-5,
        ):
            raise ValueError(
                f"{sim_id}: exposed area does not equal chip top area"
            )
    else:
        if (
            torch.any(
                graph.cooling_exposed_area_mm2 != 0
            )
            or
            torch.any(
                graph.G_amb_W_K != 0
            )
        ):
            raise ValueError(
                f"{sim_id}: boundary-ineligible graph must have zero convection tensors"
            )

    if graph.pinn_global_energy_eligible:
        if not graph.physics_power_semantics_verified:
            raise ValueError(
                f"{sim_id}: global-energy eligible graph has unverified power"
            )

        if not graph.boundary_physics_ready:
            raise ValueError(
                f"{sim_id}: global-energy eligible graph has no valid boundary"
            )

        if not graph.global_energy_constraint_candidate:
            raise ValueError(
                f"{sim_id}: global-energy eligible graph is not a valid candidate"
            )

        if not graph.gt_global_energy_audit_pass:
            raise ValueError(
                f"{sim_id}: global-energy eligible graph failed GT QA"
            )

        if (
            graph.gt_global_energy_closure_error_pct
            >
            graph.global_energy_closure_gate_pct
        ):
            raise ValueError(
                f"{sim_id}: global-energy eligibility gate inconsistent"
            )

    # For the recovered top-heat-sink policy, only the highest layer
    # may have non-zero convection area.
    top_layer = int(
        graph.n_layers
        -
        1
    )

    non_top = (
        torch.round(
            graph.x[:, 2]
        ).long()
        !=
        top_layer
    )

    if torch.any(
        graph.cooling_exposed_area_mm2[
            non_top
        ] > 0
    ):
        raise ValueError(
            f"{sim_id}: convection area assigned below top layer"
        )

    # graph.x feature 12 mirrors cooling area.
    if not torch.allclose(
        graph.x[:, 12],
        graph.cooling_exposed_area_mm2,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"{sim_id}: x[:,12] exposed area mismatch"
        )

    # ----------------------------------------------------------
    # Conductance = 1 / resistance
    # ----------------------------------------------------------

    resistance = (
        graph.edge_attr[
            :,
            0,
        ]
    )

    conductance = (
        graph.edge_attr[
            :,
            6,
        ]
    )

    if torch.any(
        resistance <= 0
    ):

        raise ValueError(
            f"{sim_id}: "
            "non-positive resistance"
        )

    if not torch.allclose(
        conductance,

        1.0
        /
        resistance,

        rtol=1e-4,
        atol=1e-7,
    ):

        raise ValueError(
            f"{sim_id}: "
            "conductance != 1/R"
        )

    physical_edges = (
        graph.direct_lateral_edge_count
        +
        graph.recovered_filler_gap_edge_count
        +
        graph.vertical_edge_count
    )

    expected_directed = (
        2
        *
        physical_edges
    )

    if (
        graph.edge_index.shape[1]
        !=
        expected_directed
    ):

        raise ValueError(
            f"{sim_id}: "
            "bidirectional edge mismatch"
        )


# ==============================================================
# BUILD DATASET
# ==============================================================

def build_dataset(
    h5_path,
    limit,
    contact_tolerance,
    max_recovery_gap,
):

    graphs = []

    with h5py.File(
        h5_path,
        "r",
    ) as hdf:

        if "simulations" not in hdf:

            raise KeyError(
                "Missing simulations group"
            )

        all_sim_names = sorted(
            hdf[
                "simulations"
            ].keys()
        )

        # ==========================================================
        # EXPLICIT AUDITED SKIP
        # ==========================================================
        # Skip by HDF5 GROUP NAME, not by the internal sim_id attr.
        # This happens before --limit so --limit N still means
        # "build up to N valid/non-excluded simulation groups".
        present_skips = [
            name
            for name in SKIP_SIMULATION_GROUPS
            if name in hdf["simulations"]
        ]

        sim_names = [
            name
            for name in all_sim_names
            if name not in SKIP_SIMULATION_GROUPS
        ]

        if limit is not None:

            sim_names = (
                sim_names[
                    :limit
                ]
            )

        print(
            "Reading robust mixed-schema thermal dataset:"
        )

        print(
            f" HDF5 simulation groups total = {len(all_sim_names)}"
        )

        if present_skips:
            print(
                f" explicitly excluded groups   = {len(present_skips)}"
            )

            for skipped_name in present_skips:

                skipped_group = (
                    hdf[
                        "simulations"
                    ][
                        skipped_name
                    ]
                )

                skipped_attr_id = str_attr(
                    skipped_group,
                    "sim_id",
                    "",
                )

                print(
                    f"  SKIP {skipped_name}"
                    f" | attr sim_id={skipped_attr_id}"
                    f" | reason="
                    f"{SKIP_SIMULATION_GROUPS[skipped_name]}"
                )

        else:
            print(
                " explicitly excluded groups   = 0 "
                "(configured bad group not present in this HDF5)"
            )

        print(
            f" groups selected for building = {len(sim_names)}"
        )

        print(
            os.path.abspath(
                h5_path
            )
        )

        print()

        print(
            "Feature contract:"
        )

        print(
            f" node features   = "
            f"{len(NODE_FEATURE_NAMES)}"
        )

        print(
            f" edge features   = "
            f"{len(EDGE_FEATURE_NAMES)}"
        )

        print(
            f" global features = "
            f"{len(GLOBAL_FEATURE_NAMES)}"
        )

        print(
            " global[25] = "
            "chip_width_mm"
        )

        print(
            " global[26] = "
            "chip_height_mm"
        )

        print()

        print(
            "Graph topology:"
        )

        print(
            " real blocks = nodes"
        )

        print(
            " filler regions = nodes"
        )

        print(
            " lateral direct = "
            "same-layer side contact"
        )

        print(
            " lateral recovery = "
            "clear missing-filler path"
        )

        print(
            " vertical = "
            "positive XY overlap"
        )

        print(
            " filler kappa = "
            "HDF5 value"
        )

        print(
            " resistance = "
            "direct material-series"
        )

        print(
            " equivalent k = "
            "NOT used for R"
        )

        print(
            " KNN = NONE"
        )

        print(
            " arbitrary bridge = NONE"
        )

        print(
            " contact tolerance =",
            contact_tolerance,
            "mm",
        )

        print()

        total = len(
            sim_names
        )

        for number, sim_name in enumerate(
            sim_names,
            start=1,
        ):

            sim = (
                hdf[
                    "simulations"
                ][
                    sim_name
                ]
            )

            graph = (
                build_graph_from_group(
                    sim,
                    hdf,
                    contact_tolerance,
                    max_recovery_gap,
                )
            )

            validate_graph(
                graph
            )

            graphs.append(
                graph
            )

            if (
                number <= 3
                or
                number % 100 == 0
                or
                number == total
            ):

                print(
                    f"Processed "
                    f"{number:4d}: "
                    f"{graph.sim_id} "

                    f"| real="
                    f"{graph.real_node_count} "

                    f"| filler="
                    f"{graph.filler_node_count} "

                    f"| merged="
                    f"{graph.block_features_merged_width} "

                    f"| real_geo="
                    f"{graph.real_geometry_source} "

                    f"| filler_geo="
                    f"{graph.filler_geometry_source} "

                    f"| nodes="
                    f"{graph.total_node_count} "

                    f"| lateral="
                    f"{graph.lateral_edge_count} "

                    f"| recovered="
                    f"{graph.recovered_filler_gap_edge_count} "

                    f"| vertical="
                    f"{graph.vertical_edge_count} "

                    f"| chip="
                    f"{graph.chip_width_mm:.4f}"
                    f"x"
                    f"{graph.chip_height_mm:.4f} mm "

                    f"| components="
                    f"{graph.connected_component_count} "

                    f"| isolated="
                    f"{graph.isolated_node_count}"
                )

    return graphs


# ==============================================================
# METADATA
# ==============================================================

def metadata_path(
    output,
):

    output = Path(
        output
    )

    return output.with_name(
        output.stem
        +
        "_metadata.json"
    )


def save_metadata(
    output,
    source_h5,
    graphs,
    contact_tolerance,
    max_recovery_gap,
):

    metadata = {
        "source_h5":
        os.path.abspath(
            source_h5
        ),

        "output_pt":
        os.path.abspath(
            output
        ),

        "number_of_graphs":
        len(
            graphs
        ),

        "dataset_exclusion_policy": {
            "mode":
            "explicit_group_skip_before_build",

            "source_h5_modified":
            False,

            "skipped_hdf5_groups":
            list(
                SKIP_SIMULATION_GROUPS.keys()
            ),

            "skip_reasons":
            dict(
                SKIP_SIMULATION_GROUPS
            ),

            "note":
            (
                "sim_001460 is excluded as one audited bad legacy V2 "
                "simulation. No filler temperature is interpolated or invented."
            ),
        },

        "node_feature_count":
        len(
            NODE_FEATURE_NAMES
        ),

        "edge_feature_count":
        len(
            EDGE_FEATURE_NAMES
        ),

        "global_feature_count":
        len(
            GLOBAL_FEATURE_NAMES
        ),

        "node_features":
        NODE_FEATURE_NAMES,

        "pinn_raw_physics": {
            "physics_node_power_W":
            (
                "AUDITED historical V3: raw block_features[:,0] numeric value "
                "as solver input W for real nodes; filler=0. "
                "Legacy V2 excluded until power semantics are proven."
            ),

            "physics_mask":
            (
                "all real+filler nodes only for cohorts with verified historical "
                "solver-power semantics; all False for unverified cohorts"
            ),

            "ground_truth_temperature_K":
            "real labels plus point-cloud-derived filler temperatures",

            "edge_resistance_K_W":
            "raw edge_attr[:,0]",

            "edge_conductance_W_K":
            "raw edge_attr[:,6]",

            "cooling_exposed_area_mm2":
            (
                "one horizontal footprint area for nodes on the solver top heat-sink "
                "surface; zero for internal layers"
            ),

            "G_amb_W_K":
            "cooling_htc_w_m2k * cooling_exposed_area_mm2 * 1e-6",

            "solver_input_total_power_W":
            "sum(physics_node_power_W) for audited V3 solver source",

            "legacy_stored_total_power_W":
            "historical H5 total_power_W = PD*area convention; metadata only",

            "raw_node_feature_0":
            (
                "Name remains power_density_W_mm2 for ML compatibility, but for "
                "audited historical V3 the generator used its numeric value directly "
                "as 3D-ICE power values [W]."
            ),

            "global_feature_2":
            (
                "Legacy total_power_W retained for ML compatibility; NOT the "
                "audited historical V3 solver-input total."
            ),

            "hard_node_kcl":
            "disabled for historical coarse graph; diagnostics only",

            "experimental_global_energy_loss":
            (
                "r_global = sum(P) - sum[G_amb*(T-Tamb)]; "
                "use only when pinn_global_energy_eligible=True"
            ),

            "global_energy_eligibility":
            (
                "pinn_global_energy_eligible is a target-derived dataset-QA mask: "
                "verified V3 source + valid top boundary + stored-FEA global closure "
                "within the configured empirical gate. Never feed this mask as a "
                "model input."
            ),

            "boundary_policy":
            (
                "Prefer explicit H5 mapping. For this dataset, recover top face from "
                "generate_thermal_dataset_v3.py, which explicitly writes `top heat sink:`."
            ),
        },

        "edge_features":
        EDGE_FEATURE_NAMES,

        "global_features":
        GLOBAL_FEATURE_NAMES,

        "chip_dimensions": {
            "source_width":
            "HDF5 chip_w_um",

            "source_height":
            "HDF5 chip_h_um",

            "graph_width":
            "chip_width_mm",

            "graph_height":
            "chip_height_mm",

            "global_width_index":
            25,

            "global_height_index":
            26,

            "conversion":
            "um / 1000 = mm",
        },

        "node_type_codes": {
            "0":
            "reconstructed filler",

            "1":
            "logic",

            "2":
            "memory",
        },

        "edge_type_codes": {
            "0":
            "direct lateral side contact",

            "1":
            "recovered missing-filler thermal path",

            "2":
            "vertical adjacent-layer XY overlap",
        },

        "resistance_model": {
            "lateral_direct":
            (
                "half node A + "
                "half node B"
            ),

            "lateral_recovered":
            (
                "half node A + "
                "missing filler gap + "
                "half node B"
            ),

            "vertical":
            (
                "half layer A + "
                "BOND_MAT + "
                "half layer B"
            ),

            "equivalent_kappa":
            (
                "diagnostic only; "
                "never used to calculate R"
            ),
        },

        "filler": {
            "source":
            "HDF5 filler_blocks",

            "kappa_source":
            "HDF5 filler_kappa_w_mk",

            "geometry_reconstructed":
            "per-graph; exact [F,7] used when available",

            "exact_solver_replay":
            "per-graph; exact [F,7] geometry is preserved",
        },

        "tsv": {
            "exact_positions_available":
            "per-graph; true when tsv_pads [T,7] exists",

            "vertical_edge_rule":
            (
                "positive XY overlap "
                "between adjacent layers"
            ),

            "n_tsv_pairs":
            "aggregate metadata only",
        },

        "mixed_schema_policy": {
            "block_features_merged_Nx15":
            (
                "Use columns 13/14 as exact z_center_mm/"
                "thickness_mm geometry only; graph.x stays compact; this PINN-ready version has 13 features."
            ),

            "block_features_merged_Nx7":
            (
                "Reconstruct real-node z/thickness from "
                "layer_idx + layer_thicknesses_um + bond_thickness_um."
            ),

            "filler_blocks_Fx7":
            (
                "Use exact x,y,w,h,layer,z_center,thickness."
            ),

            "filler_blocks_Fx5":
            (
                "Legacy fallback: reconstruct filler z/thickness "
                "from layer stack."
            ),

            "tsv_pads":
            (
                "Preserve [T,7] as metadata when present; "
                "do not add TSV graph nodes."
            ),
        },

        "contact_tolerance_mm":
        contact_tolerance,

        "max_recovery_gap_mm":
        max_recovery_gap,

        "global_energy_closure_gate_pct":
        GLOBAL_ENERGY_CLOSURE_GATE_PCT,

        "pinn_cohort_policy": {
            "hard_node_kcl":
            (
                "DISABLED for historical coarse block/filler graph because "
                "solver-cell widths/heights were not preserved in H5."
            ),
            "experimental_global_energy":
            (
                "Allowed only for audited V3 power semantics + exact F7 top "
                "boundary geometry + <=5% ground-truth global closure error."
            ),
            "legacy_V2":
            (
                "Kept for ML only; physics source semantics unverified."
            ),
            "global_feature_2":
            (
                "Legacy HDF5 total_power_W retained for backward ML compatibility; "
                "do not use it as the historical V3 solver source."
            )
        },

        "totals": {
            "pinn_full_physics_eligible_graphs":
            sum(
                int(g.pinn_full_physics_eligible)
                for g in graphs
            ),

            "global_energy_constraint_candidate_graphs":
            sum(
                int(g.global_energy_constraint_candidate)
                for g in graphs
            ),

            "gt_global_energy_audit_pass_graphs":
            sum(
                int(g.gt_global_energy_audit_pass)
                for g in graphs
            ),

            "pinn_global_energy_eligible_graphs":
            sum(
                int(g.pinn_global_energy_eligible)
                for g in graphs
            ),

            "power_semantics_verified_graphs":
            sum(
                int(g.physics_power_semantics_verified)
                for g in graphs
            ),

            "boundary_physics_ready_graphs":
            sum(
                int(g.boundary_physics_ready)
                for g in graphs
            ),

            "pinn_boundary_ineligible_graphs":
            sum(
                int(not g.boundary_physics_ready)
                for g in graphs
            ),
            "real_nodes":
            sum(
                g.real_node_count
                for g
                in graphs
            ),

            "filler_nodes":
            sum(
                g.filler_node_count
                for g
                in graphs
            ),

            "total_nodes":
            sum(
                g.total_node_count
                for g
                in graphs
            ),

            "direct_lateral":
            sum(
                g.direct_lateral_edge_count
                for g
                in graphs
            ),

            "recovered_filler_gap":
            sum(
                g.recovered_filler_gap_edge_count
                for g
                in graphs
            ),

            "vertical":
            sum(
                g.vertical_edge_count
                for g
                in graphs
            ),

            "directed_edges":
            sum(
                g.directed_edge_count
                for g
                in graphs
            ),
        },
    }

    path = metadata_path(
        output
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    return str(
        path
    )


# ==============================================================
# PRINT SAMPLE
# ==============================================================

def print_sample(
    graph,
):

    print(
        "\n"
        +
        "=" * 75
    )

    print(
        "SAMPLE GRAPH"
    )

    print(
        "=" * 75
    )

    print(
        graph
    )

    print(
        "\nShapes"
    )

    print(
        "------"
    )

    print(
        "x:",
        tuple(
            graph.x.shape
        ),
    )

    print(
        "edge_index:",
        tuple(
            graph.edge_index.shape
        ),
    )

    print(
        "edge_attr:",
        tuple(
            graph.edge_attr.shape
        ),
    )

    print(
        "y:",
        tuple(
            graph.y.shape
        ),
    )

    print(
        "target_mask:",
        tuple(
            graph.target_mask.shape
        ),
    )

    print(
        "pos:",
        tuple(
            graph.pos.shape
        ),
    )

    print(
        "global_features:",
        tuple(
            graph.global_features.shape
        ),
    )

    print(
        "\nPINN raw physics"
    )

    print(
        "----------------"
    )

    print(
        "physics_node_power_W:",
        tuple(
            graph.physics_node_power_W.shape
        ),
    )

    print(
        "physics_mask nodes:",
        int(
            graph.physics_mask.sum()
        ),
    )

    print(
        "Boundary mapping available:",
        graph.boundary_mapping_available,
    )

    print(
        "Boundary mapping source:",
        graph.boundary_mapping_source,
    )

    print("Boundary geometry ready:", graph.boundary_geometry_ready)
    print("Exact filler geometry cohort:", graph.exact_filler_geometry_cohort)
    print("Boundary physics ready:", graph.boundary_physics_ready)
    print("Physics power semantics verified:", graph.physics_power_semantics_verified)
    print("Solver input total power:", graph.solver_input_total_power_W, "W")
    print("Legacy stored total power:", graph.legacy_stored_total_power_W, "W")
    print("Hard node-KCL ready:", graph.physics_residual_ready)
    print("PINN full-physics eligible:", graph.pinn_full_physics_eligible)
    print("Global-energy candidate:", graph.global_energy_constraint_candidate)
    print("GT global-energy audit evaluated:", graph.gt_global_energy_audit_evaluated)
    print("GT global-energy audit pass:", graph.gt_global_energy_audit_pass)
    print("PINN global-energy eligible:", graph.pinn_global_energy_eligible)
    print("GT global closure error:", graph.gt_global_energy_closure_error_pct, "%")
    print("GT node Kref RMSE:", graph.gt_node_residual_Kref_rmse, "K")
    print("GT mean layer |res|/P:", graph.gt_mean_layer_abs_residual_pct_total_power, "%")
    print("Physics loss scope:", graph.physics_loss_scope)

    print(
        "Cooling face:",
        graph.cooling_face,
    )

    print(
        "Cooling-exposed area sum:",
        float(
            graph.cooling_exposed_area_mm2.sum()
        ),
        "mm^2",
    )

    print(
        "G_amb sum:",
        float(
            graph.G_amb_W_K.sum()
        ),
        "W/K",
    )

    print("Top-area coverage ratio:", graph.cooling_area_coverage_ratio)
    print("Top overlap ratio:", graph.top_overlap_ratio)
    print("Top uncovered area:", graph.top_uncovered_area_mm2, "mm^2")

    if graph.cooling_area_coverage_warning:
        print(
            "Coverage warning:",
            graph.cooling_area_coverage_warning,
        )

    if graph.boundary_warning:
        print(
            "Boundary warning:",
            graph.boundary_warning,
        )

    print(
        "\nFull chip / layer footprint"
    )

    print(
        "---------------------------"
    )

    print(
        "Chip width:",
        graph.chip_width_mm,
        "mm",
    )

    print(
        "Chip height:",
        graph.chip_height_mm,
        "mm",
    )

    print(
        "global[25]:",
        graph.global_features[
            0,
            25,
        ].item(),
    )

    print(
        "global[26]:",
        graph.global_features[
            0,
            26,
        ].item(),
    )

    print(
        "\nNodes"
    )

    print(
        "-----"
    )

    print(
        "Real:",
        graph.real_node_count,
    )

    print(
        "Filler:",
        graph.filler_node_count,
    )

    print(
        "Total:",
        graph.total_node_count,
    )

    print(
        "Target nodes:",
        graph.target_node_count,
    )

    print(
        "\nSchema handling"
    )

    print(
        "---------------"
    )

    print(
        "block_features_merged width:",
        graph.block_features_merged_width,
    )

    print(
        "real geometry source:",
        graph.real_geometry_source,
    )

    print(
        "filler_blocks width:",
        graph.filler_blocks_width,
    )

    print(
        "filler geometry source:",
        graph.filler_geometry_source,
    )

    print(
        "TSV exact positions available:",
        graph.tsv_exact_positions_available,
    )

    print(
        "TSV pad count:",
        graph.tsv_pad_count,
    )

    print(
        "\nEdges"
    )

    print(
        "-----"
    )

    print(
        "Direct lateral:",
        graph.direct_lateral_edge_count,
    )

    print(
        "Recovered filler gap:",
        graph.recovered_filler_gap_edge_count,
    )

    print(
        "Lateral total:",
        graph.lateral_edge_count,
    )

    print(
        "Vertical:",
        graph.vertical_edge_count,
    )

    print(
        "Directed:",
        graph.directed_edge_count,
    )

    print(
        "\nConnectivity"
    )

    print(
        "------------"
    )

    print(
        "Components:",
        graph.connected_component_count,
    )

    print(
        "Component sizes:",
        graph.connected_component_sizes.tolist(),
    )

    print(
        "Isolated:",
        graph.isolated_nodes.tolist(),
    )

    print(
        "\nMaterial"
    )

    print(
        "--------"
    )

    print(
        "Filler kappa:",
        graph.filler_kappa_W_mK,
        "W/(m*K)",
    )

    print(
        "Bond kappa:",
        graph.bond_kappa_W_mK,
        "W/(m*K)",
    )

    print(
        "Bond thickness:",
        graph.bond_thickness_mm,
        "mm",
    )

    print(
        "\nTSV"
    )

    print(
        "---"
    )

    print(
        "n_tsv_pairs:",
        graph.n_tsv_pairs,
    )

    print(
        "Exact TSV positions:",
        graph.tsv_exact_positions_available,
    )

    print(
        "Vertical = geometric overlap:",
        graph.vertical_edges_are_geometric_overlap,
    )

    print(
        "\nGlobal feature names:"
    )

    for i, name in enumerate(
        GLOBAL_FEATURE_NAMES
    ):

        print(
            f"{i:2d}: "
            f"{name:<32} "
            f"= "
            f"{graph.global_features[0, i].item()}"
        )

    print(
        "\nFirst five nodes:"
    )

    print(
        graph.x[:5]
    )

    print(
        "\nFirst five edge attrs:"
    )

    print(
        graph.edge_attr[:5]
    )


# ==============================================================
# COMMAND LINE
# ==============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build V4.2 audited thermal PyG graphs with explicit sim_001460 exclusion: "
            "historical V3 solver-faithful power, hard-node PINN disabled, "
            "optional GT-audited global-energy regularizer."
        )
    )

    parser.add_argument(
        "h5_path",

        help=(
            "thermal_dataset_v3_balanced_v2schema_final.h5"
        ),
    )

    parser.add_argument(
        "--output",

        default=(
            "thermal_graph_PINN_V4_2_balanced_audited.pt"
        ),
    )

    parser.add_argument(
        "--limit",

        type=int,

        default=None,
    )

    parser.add_argument(
        "--contact-tolerance-mm",

        type=float,

        default=
        DEFAULT_CONTACT_TOL_MM,
    )

    parser.add_argument(
        "--max-recovery-gap-mm",

        type=float,

        default=None,

        help=(
            "Optional limit for missing "
            "filler recovery paths. "
            "Default = no hard limit."
        ),
    )

    return parser.parse_args()


# ==============================================================
# MAIN
# ==============================================================

def main():

    args = parse_args()

    h5_path = os.path.abspath(
        os.path.expanduser(
            args.h5_path
        )
    )

    output = os.path.abspath(
        os.path.expanduser(
            args.output
        )
    )

    if not os.path.isfile(
        h5_path
    ):

        raise FileNotFoundError(
            h5_path
        )

    if (
        args.limit is not None
        and
        args.limit <= 0
    ):

        raise ValueError(
            "--limit must be > 0"
        )

    if (
        args.contact_tolerance_mm
        < 0
    ):

        raise ValueError(
            "contact tolerance "
            "must be >= 0"
        )

    if (
        args.max_recovery_gap_mm
        is not None
        and
        args.max_recovery_gap_mm <= 0
    ):

        raise ValueError(
            "max recovery gap "
            "must be > 0"
        )

    graphs = build_dataset(
        h5_path,

        args.limit,

        args.contact_tolerance_mm,

        args.max_recovery_gap_mm,
    )

    if not graphs:

        raise RuntimeError(
            "No graphs created"
        )

    torch.save(
        graphs,
        output,
    )

    meta_path = (
        save_metadata(
            output,
            h5_path,
            graphs,

            args.contact_tolerance_mm,

            args.max_recovery_gap_mm,
        )
    )

    print()

    print(
        f"Created "
        f"{len(graphs)} graphs."
    )

    print(
        "Saved:",
        output,
    )

    print(
        "Metadata:",
        meta_path,
    )

    print_sample(
        graphs[0]
    )


if __name__ == "__main__":
    main()