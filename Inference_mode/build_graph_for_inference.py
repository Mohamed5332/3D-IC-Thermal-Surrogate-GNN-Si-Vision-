#!/usr/bin/env python3
"""
Si-Vision Thermal Surrogate - Inference Graph Builder
=====================================================

Builds PyTorch Geometric inference graphs from the user-input H5 produced by
convert_user_inputs_to_h5.py.

CRITICAL CONTRACT
-----------------
- Preserves the audited ML feature contract:
    13 node features
    11 edge features
    27 global features
- DOES NOT require or fabricate:
    node_temperature_labels
    point_cloud
    y / ground-truth temperatures
- Reuses the audited training graph builder's geometry, edge construction,
  resistance model, global-feature construction, top-surface QA, and boundary
  mapping helpers so inference topology remains aligned with training.

Expected inference H5 datasets per simulation:
    block_features_merged  [N,15]
    filler_blocks          [F,7]
Optional:
    block_names
    tsv_pads               [T,7]

The output is saved as list[torch_geometric.data.Data] to remain compatible
with the existing dataset serialization style. For a normal web inference H5,
the list contains exactly one graph.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, List

import h5py
import numpy as np
import torch
from torch_geometric.data import Data


EXPECTED_NODE_FEATURES = [
    "power_density_W_mm2",
    "area_mm2",
    "layer_idx",
    "kappa_W_mK",
    "block_type_code",
    "has_tsv",
    "is_hotspot",
    "x_center_mm",
    "y_center_mm",
    "z_center_mm",
    "width_mm",
    "height_mm",
    "exposed_area_mm2",
]

EXPECTED_EDGE_FEATURES = [
    "resistance_K_W",
    "is_vertical",
    "euclidean_distance_mm",
    "delta_z_mm",
    "contact_area_mm2",
    "effective_path_kappa_W_mK",
    "conductance_W_K",
    "edge_type_code",
    "thermal_path_mm",
    "has_bond_interface",
    "interface_or_filler_gap_mm",
]

EXPECTED_GLOBAL_FEATURES = [
    "ambient_temp_K",
    "workload_factor",
    "total_power_W",
    "n_layers",
    "n_tsv_pairs",
    "has_hotspot",
    "hotspot_multiplier",
    "cooling_type_code",
    "cooling_htc_w_m2k",
    "package_recipe_code",
    "non_uniform_material",
    "bond_kappa_w_mk",
    "bond_thickness_mm",
    "has_rdl",
    "rdl_thickness_mm",
    "rdl_kappa_w_mk",
    "has_interposer",
    "interposer_thickness_mm",
    "interposer_kappa_w_mk",
    "has_c4",
    "c4_thickness_mm",
    "c4_kappa_w_mk",
    "has_package_substrate",
    "substrate_thickness_mm",
    "substrate_kappa_w_mk",
    "chip_width_mm",
    "chip_height_mm",
]

DEFAULT_REFERENCE_CANDIDATES = [
    Path(__file__).resolve().parent / "build_graph_PINN_ready_final_cohort_gated.py",
    Path("/home/users/svmlint26yalrazek/thermal_dataset_project/v3/build_graph_PINN_ready_final_cohort_gated.py"),
]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr(group: h5py.Group, name: str, default: Any = None) -> Any:
    return _decode(group.attrs[name]) if name in group.attrs else default


def _float_attr(group: h5py.Group, name: str, default: float = 0.0) -> float:
    try:
        return float(_attr(group, name, default))
    except (TypeError, ValueError):
        return float(default)


def _int_attr(group: h5py.Group, name: str, default: int = 0) -> int:
    try:
        return int(round(float(_attr(group, name, default))))
    except (TypeError, ValueError):
        return int(default)


def _str_attr(group: h5py.Group, name: str, default: str = "") -> str:
    return str(_attr(group, name, default))


def _resolve_reference_builder(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Reference audited graph builder not found: {p}")
        return p
    for candidate in DEFAULT_REFERENCE_CANDIDATES:
        p = candidate.expanduser().resolve()
        if p.exists() and p != Path(__file__).resolve():
            return p
    raise FileNotFoundError(
        "Could not locate build_graph_PINN_ready_final_cohort_gated.py. "
        "Pass --reference-builder /path/to/the/audited/training/builder.py"
    )


def _load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("si_vision_audited_graph_contract", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import reference graph builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required_symbols = [
        "NODE_FEATURE_NAMES",
        "EDGE_FEATURE_NAMES",
        "GLOBAL_FEATURE_NAMES",
        "NodeGeometry",
        "build_edges",
        "build_global_features",
        "build_boundary_physics",
        "evaluate_top_surface_partition",
    ]
    missing = [name for name in required_symbols if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Reference builder missing required symbols: {missing}")

    if list(module.NODE_FEATURE_NAMES) != EXPECTED_NODE_FEATURES:
        raise RuntimeError(
            "Node feature contract mismatch between inference builder and audited builder.\n"
            f"Expected: {EXPECTED_NODE_FEATURES}\n"
            f"Audited:  {list(module.NODE_FEATURE_NAMES)}"
        )
    if list(module.EDGE_FEATURE_NAMES) != EXPECTED_EDGE_FEATURES:
        raise RuntimeError("Edge feature contract mismatch with audited builder.")
    if list(module.GLOBAL_FEATURE_NAMES) != EXPECTED_GLOBAL_FEATURES:
        raise RuntimeError("Global feature contract mismatch with audited builder.")
    return module


def _find_col(columns: list[str], names: list[str], fallback: int) -> int:
    lowered = [str(c).strip().lower() for c in columns]
    for target in names:
        target = target.lower()
        for i, col in enumerate(lowered):
            if col == target or col.startswith(target):
                return i
    return fallback


def _merged_columns(sim: h5py.Group, ref) -> list[str]:
    raw = _attr(sim, "block_features_merged_columns", "")
    if raw:
        return ref.parse_columns(raw) if hasattr(ref, "parse_columns") else [x.strip() for x in str(raw).split(",")]
    return [
        "power_density_W_mm2",
        "area_mm2",
        "layer_idx",
        "kappa_W_mK",
        "block_type",
        "has_tsv",
        "is_hotspot",
        "x_mm",
        "y_mm",
        "width_mm",
        "height_mm",
        "x_center_mm",
        "y_center_mm",
        "z_center_mm",
        "thickness_mm",
    ]


def build_inference_nodes(sim: h5py.Group, hdf: h5py.File, ref):
    if "block_features_merged" not in sim:
        raise ValueError(f"{sim.name}: missing block_features_merged")
    if "filler_blocks" not in sim:
        raise ValueError(f"{sim.name}: missing filler_blocks")

    merged = np.asarray(sim["block_features_merged"][:], dtype=np.float64)
    fillers = np.asarray(sim["filler_blocks"][:], dtype=np.float64)

    if merged.ndim != 2 or merged.shape[1] < 15:
        raise ValueError(f"{sim.name}: block_features_merged must be [N,15+], got {merged.shape}")
    if fillers.ndim != 2 or fillers.shape[1] != 7:
        raise ValueError(
            f"{sim.name}: inference filler_blocks must be exact [F,7] "
            f"(x,y,w,h,layer,z_center,thickness), got {fillers.shape}"
        )
    if not np.all(np.isfinite(merged)):
        raise ValueError(f"{sim.name}: block_features_merged contains NaN/Inf")
    if not np.all(np.isfinite(fillers)):
        raise ValueError(f"{sim.name}: filler_blocks contains NaN/Inf")

    cols = _merged_columns(sim, ref)
    idx = {
        "power": _find_col(cols, ["power_density_w_mm2"], 0),
        "area": _find_col(cols, ["area_mm2"], 1),
        "layer": _find_col(cols, ["layer_idx"], 2),
        "kappa": _find_col(cols, ["kappa_w_mk"], 3),
        "type": _find_col(cols, ["block_type"], 4),
        "tsv": _find_col(cols, ["has_tsv"], 5),
        "hotspot": _find_col(cols, ["is_hotspot"], 6),
        "x": _find_col(cols, ["x_mm"], 7),
        "y": _find_col(cols, ["y_mm"], 8),
        "w": _find_col(cols, ["width_mm"], 9),
        "h": _find_col(cols, ["height_mm"], 10),
        "cx": _find_col(cols, ["x_center_mm"], 11),
        "cy": _find_col(cols, ["y_center_mm"], 12),
        "cz": _find_col(cols, ["z_center_mm"], 13),
        "thickness": _find_col(cols, ["thickness_mm"], 14),
    }

    n_real = int(merged.shape[0])
    n_filler = int(fillers.shape[0])
    total_nodes = n_real + n_filler

    if n_real <= 0:
        raise ValueError(f"{sim.name}: no real blocks found")

    real_layers = np.round(merged[:, idx["layer"]]).astype(int)
    filler_layers = np.round(fillers[:, 4]).astype(int) if n_filler else np.zeros((0,), dtype=int)
    max_real_layer = int(np.max(real_layers)) if real_layers.size else 0
    max_filler_layer = int(np.max(filler_layers)) if filler_layers.size else 0
    inferred_layers = int(max(max_real_layer, max_filler_layer) + 1)
    n_layers = _int_attr(sim, "n_layers", inferred_layers)
    if n_layers <= 0:
        raise ValueError(f"{sim.name}: invalid n_layers={n_layers}")
    if np.any(real_layers < 0) or np.any(real_layers >= n_layers):
        raise ValueError(f"{sim.name}: real node layer outside 0..{n_layers - 1}")
    if n_filler and (np.any(filler_layers < 0) or np.any(filler_layers >= n_layers)):
        raise ValueError(f"{sim.name}: filler node layer outside 0..{n_layers - 1}")

    chip_width_mm = _float_attr(sim, "chip_w_um", 0.0) / 1000.0
    chip_height_mm = _float_attr(sim, "chip_h_um", 0.0) / 1000.0
    if chip_width_mm <= 0 or chip_height_mm <= 0:
        raise ValueError(f"{sim.name}: chip_w_um/chip_h_um must be positive")

    filler_kappa = _float_attr(sim, "filler_kappa_w_mk", 130.0)
    if filler_kappa <= 0:
        raise ValueError(f"{sim.name}: invalid filler_kappa_w_mk={filler_kappa}")

    bond_kappa = _float_attr(sim, "bond_kappa_w_mk", 0.0)
    bond_thickness_mm = _float_attr(sim, "bond_thickness_um", 0.0) / 1000.0

    x = np.zeros((total_nodes, len(EXPECTED_NODE_FEATURES)), dtype=np.float32)
    pos = np.zeros((total_nodes, 3), dtype=np.float32)
    real_mask = np.zeros(total_nodes, dtype=bool)
    filler_mask = np.zeros(total_nodes, dtype=bool)
    geometries = []

    for i in range(n_real):
        px = float(merged[i, idx["x"]])
        py = float(merged[i, idx["y"]])
        w = float(merged[i, idx["w"]])
        h = float(merged[i, idx["h"]])
        cx = float(merged[i, idx["cx"]])
        cy = float(merged[i, idx["cy"]])
        cz = float(merged[i, idx["cz"]])
        thickness = float(merged[i, idx["thickness"]])
        area = float(merged[i, idx["area"]])
        layer = int(real_layers[i])
        kappa = float(merged[i, idx["kappa"]])

        if w <= 0 or h <= 0 or thickness <= 0 or kappa <= 0:
            raise ValueError(f"{sim.name}: invalid geometry/material for real node {i}")
        if area <= 0:
            area = w * h

        x[i] = [
            float(merged[i, idx["power"]]),
            area,
            float(layer),
            kappa,
            float(merged[i, idx["type"]]),
            float(merged[i, idx["tsv"]]),
            float(merged[i, idx["hotspot"]]),
            cx,
            cy,
            cz,
            w,
            h,
            0.0,
        ]
        pos[i] = [cx, cy, cz]
        real_mask[i] = True
        geometries.append(
            ref.NodeGeometry(
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

    for fi in range(n_filler):
        gi = n_real + fi
        px, py, w, h, layer_raw, cz, thickness = map(float, fillers[fi, :7])
        layer = int(round(layer_raw))
        if w <= 0 or h <= 0 or thickness <= 0:
            raise ValueError(f"{sim.name}: invalid geometry for filler node {fi}")
        area = w * h
        cx = px + w / 2.0
        cy = py + h / 2.0
        x[gi] = [
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
            0.0,
        ]
        pos[gi] = [cx, cy, cz]
        filler_mask[gi] = True
        geometries.append(
            ref.NodeGeometry(
                idx=gi,
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

    thicknesses_um = np.asarray(_attr(sim, "layer_thicknesses_um", []), dtype=np.float64).reshape(-1)
    if thicknesses_um.size not in (0, n_layers):
        raise ValueError(
            f"{sim.name}: layer_thicknesses_um has {thicknesses_um.size} values but n_layers={n_layers}"
        )

    aux = {
        "n_real": n_real,
        "n_filler": n_filler,
        "n_layers": n_layers,
        "filler_kappa_w_mk": filler_kappa,
        "bond_kappa_w_mk": bond_kappa,
        "bond_thickness_mm": bond_thickness_mm,
        "chip_width_mm": chip_width_mm,
        "chip_height_mm": chip_height_mm,
        "filler_blocks_width": 7,
        "filler_geometry_source": "filler_blocks_7",
        "block_features_merged_width": int(merged.shape[1]),
        "real_geometry_source": "block_features_merged_15",
        "layer_thicknesses_mm": thicknesses_um / 1000.0 if thicknesses_um.size else np.zeros((n_layers,)),
    }

    return (
        torch.from_numpy(x),
        torch.from_numpy(pos),
        torch.from_numpy(real_mask),
        torch.from_numpy(filler_mask),
        geometries,
        aux,
    )


def build_graph(sim: h5py.Group, hdf: h5py.File, ref, contact_tolerance: float, max_recovery_gap: Optional[float]) -> Data:
    sim_id = _str_attr(sim, "sim_id", sim.name.split("/")[-1])

    x, pos, real_mask, filler_mask, geometries, aux = build_inference_nodes(sim, hdf, ref)

    edge_index, edge_attr, stats = ref.build_edges(
        geometries,
        filler_kappa=float(aux["filler_kappa_w_mk"]),
        bond_kappa=float(aux["bond_kappa_w_mk"]),
        bond_thickness=float(aux["bond_thickness_mm"]),
        contact_tolerance=contact_tolerance,
        max_recovery_gap=max_recovery_gap,
    )

    global_features = ref.build_global_features(sim, aux)

    # Reproduce the audited builder's inference-safe node feature 12 policy.
    boundary = ref.build_boundary_physics(
        sim,
        x,
        n_real=int(aux["n_real"]),
        n_layers=int(aux["n_layers"]),
    )
    top_partition = ref.evaluate_top_surface_partition(
        x=x,
        n_layers=int(aux["n_layers"]),
        chip_width_mm=float(aux["chip_width_mm"]),
        chip_height_mm=float(aux["chip_height_mm"]),
    )
    exact_filler_geometry = True  # inference contract requires F,7
    boundary_geometry_ready = bool(top_partition["top_partition_ok"])
    full_boundary_eligible = bool(
        boundary["boundary_mapping_available"]
        and boundary_geometry_ready
        and exact_filler_geometry
    )
    if not full_boundary_eligible:
        boundary["cooling_exposed_area_mm2"] = torch.zeros_like(boundary["cooling_exposed_area_mm2"])
        boundary["cooling_boundary_mask"] = torch.zeros_like(boundary["cooling_boundary_mask"])
        boundary["G_amb_W_K"] = torch.zeros_like(boundary["G_amb_W_K"])
    x[:, 12] = boundary["cooling_exposed_area_mm2"]

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        global_features=global_features,
        real_node_mask=real_mask,
        filler_node_mask=filler_mask,
        cooling_exposed_area_mm2=boundary["cooling_exposed_area_mm2"],
        cooling_boundary_mask=boundary["cooling_boundary_mask"],
        G_amb_W_K=boundary["G_amb_W_K"],
    )

    # Intentionally NO graph.y / labels / target mask / point-cloud targets.
    graph.sim_id = sim_id
    graph.is_inference_graph = True
    graph.inference_contract_version = "SiVision_inference_graph_v1"
    graph.reference_training_contract = "13_node__11_edge__27_global"
    graph.temperature_targets_present = False
    graph.point_cloud_required = False
    graph.real_node_count = int(aux["n_real"])
    graph.filler_node_count = int(aux["n_filler"])
    graph.total_node_count = graph.real_node_count + graph.filler_node_count
    graph.n_layers = int(aux["n_layers"])
    graph.chip_width_mm = float(aux["chip_width_mm"])
    graph.chip_height_mm = float(aux["chip_height_mm"])
    graph.filler_kappa_W_mK = float(aux["filler_kappa_w_mk"])
    graph.bond_kappa_W_mK = float(aux["bond_kappa_w_mk"])
    graph.bond_thickness_mm = float(aux["bond_thickness_mm"])
    graph.boundary_mapping_available = bool(boundary["boundary_mapping_available"])
    graph.boundary_mapping_source = str(boundary["boundary_mapping_source"])
    graph.boundary_geometry_ready = boundary_geometry_ready
    graph.boundary_physics_ready = bool(full_boundary_eligible and boundary["cooling_htc_w_m2k"] > 0)
    graph.cooling_face = "top"
    graph.data_source = _str_attr(sim, "data_source", "")
    graph.package_recipe = _str_attr(sim, "package_recipe", "")
    graph.cooling_type = _str_attr(sim, "cooling_type", "")
    graph.n_tsv_pairs = _int_attr(sim, "n_tsv_pairs", 0)
    graph.contact_tolerance_mm = float(contact_tolerance)
    graph.max_recovery_gap_mm = -1.0 if max_recovery_gap is None else float(max_recovery_gap)

    for name, value in stats.items():
        if isinstance(value, list):
            setattr(graph, name, torch.tensor(value, dtype=torch.long))
        else:
            setattr(graph, name, value)

    if "tsv_pads" in sim:
        tsv = np.asarray(sim["tsv_pads"][:], dtype=np.float32)
        if tsv.ndim == 2 and tsv.shape[1] == 7:
            graph.tsv_pads = torch.from_numpy(tsv.copy())
            graph.tsv_pad_count = int(tsv.shape[0])
            graph.tsv_exact_positions_available = True

    return graph


def validate_inference_graph(graph: Data, ref) -> None:
    if tuple(graph.x.shape)[1:] != (13,):
        raise ValueError(f"{graph.sim_id}: x must be [N,13], got {tuple(graph.x.shape)}")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(f"{graph.sim_id}: edge_index must be [2,E]")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 11:
        raise ValueError(f"{graph.sim_id}: edge_attr must be [E,11]")
    if tuple(graph.global_features.shape) != (1, 27):
        raise ValueError(f"{graph.sim_id}: global_features must be [1,27]")
    if tuple(graph.pos.shape) != (graph.x.shape[0], 3):
        raise ValueError(f"{graph.sim_id}: pos must be [N,3]")
    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError(f"{graph.sim_id}: edge count mismatch")
    if int(graph.connected_component_count) != 1 or int(graph.isolated_node_count) != 0:
        raise ValueError(f"{graph.sim_id}: inference graph is disconnected")
    for name in ["x", "edge_attr", "pos", "global_features"]:
        if not torch.isfinite(getattr(graph, name)).all():
            raise ValueError(f"{graph.sim_id}: {name} contains NaN/Inf")
    if "y" in graph.keys():
        raise ValueError(f"{graph.sim_id}: inference graph must not contain y")
    if list(ref.NODE_FEATURE_NAMES) != EXPECTED_NODE_FEATURES:
        raise RuntimeError("Reference node feature contract changed during build")


def metadata_path(output: Path) -> Path:
    return output.with_name(output.stem + "_metadata.json")


def build_dataset(
    h5_path: Path,
    ref,
    contact_tolerance: float,
    max_recovery_gap: Optional[float],
    limit: Optional[int],
) -> List[Data]:
    graphs = []
    with h5py.File(h5_path, "r") as hdf:
        if "simulations" not in hdf:
            raise KeyError("Missing /simulations group")
        names = sorted(hdf["simulations"].keys())
        if limit is not None:
            names = names[:limit]
        if not names:
            raise ValueError("No simulation groups in H5")

        print("Si-Vision inference graph builder")
        print("---------------------------------")
        print("H5:", os.path.abspath(h5_path))
        print("Simulations:", len(names))
        print("Contract: 13 node / 11 edge / 27 global")
        print("Ground-truth y: NOT CREATED")
        print("node_temperature_labels: NOT REQUIRED")
        print("point_cloud: NOT REQUIRED")
        print()

        for i, name in enumerate(names, start=1):
            sim = hdf["simulations"][name]
            graph = build_graph(sim, hdf, ref, contact_tolerance, max_recovery_gap)
            validate_inference_graph(graph, ref)
            graphs.append(graph)
            print(
                f"Built {i:4d}/{len(names)}: {graph.sim_id} "
                f"| real={graph.real_node_count} filler={graph.filler_node_count} "
                f"| nodes={graph.total_node_count} "
                f"| lateral={graph.lateral_edge_count} vertical={graph.vertical_edge_count} "
                f"| components={graph.connected_component_count} isolated={graph.isolated_node_count}"
            )
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inference-only PyG graph(s) with no temperature targets")
    parser.add_argument("h5_path", help="Inference H5 created from user .stk/.flp files")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--reference-builder", default=None, help="Audited training graph builder used as topology/feature contract reference")
    parser.add_argument("--contact-tolerance", type=float, default=0.001, help="Same-layer contact tolerance in mm")
    parser.add_argument("--max-recovery-gap", type=float, default=None, help="Optional maximum recovery gap in mm")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    h5_path = Path(args.h5_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    reference_path = _resolve_reference_builder(args.reference_builder)
    ref = _load_reference(reference_path)

    graphs = build_dataset(
        h5_path=h5_path,
        ref=ref,
        contact_tolerance=float(args.contact_tolerance),
        max_recovery_gap=args.max_recovery_gap,
        limit=args.limit,
    )

    torch.save(graphs, output)

    meta = {
        "source_h5": str(h5_path),
        "output_pt": str(output),
        "reference_audited_builder": str(reference_path),
        "serialization": "list[torch_geometric.data.Data]",
        "number_of_graphs": len(graphs),
        "ground_truth_y_present": False,
        "node_temperature_labels_required": False,
        "point_cloud_required": False,
        "node_feature_count": len(EXPECTED_NODE_FEATURES),
        "edge_feature_count": len(EXPECTED_EDGE_FEATURES),
        "global_feature_count": len(EXPECTED_GLOBAL_FEATURES),
        "node_features": EXPECTED_NODE_FEATURES,
        "edge_features": EXPECTED_EDGE_FEATURES,
        "global_features": EXPECTED_GLOBAL_FEATURES,
        "graphs": [
            {
                "sim_id": g.sim_id,
                "real_nodes": int(g.real_node_count),
                "filler_nodes": int(g.filler_node_count),
                "total_nodes": int(g.total_node_count),
                "directed_edges": int(g.edge_index.shape[1]),
                "n_layers": int(g.n_layers),
                "chip_width_mm": float(g.chip_width_mm),
                "chip_height_mm": float(g.chip_height_mm),
                "boundary_physics_ready": bool(g.boundary_physics_ready),
            }
            for g in graphs
        ],
    }
    meta_path = metadata_path(output)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print("SUCCESS")
    print("Graph file:", output)
    print("Metadata:  ", meta_path)
    print("Graphs:    ", len(graphs))
    print("No y / no ground-truth temperature targets were stored.")


if __name__ == "__main__":
    main()
