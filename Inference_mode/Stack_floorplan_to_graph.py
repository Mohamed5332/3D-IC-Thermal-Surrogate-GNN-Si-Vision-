#!/usr/bin/env python3
"""
stack_floorplan_to_graph.py
============================
One-shot wrapper: give it a folder with one .stk + its .flp files,
it returns ONE ready torch_geometric.data.Data graph.

It does NOT reimplement any logic. It simply imports and calls, in order,
the three already-audited pipeline scripts so the ML contract stays
byte-identical to the multi-step version:

    1. convert_user_inputs_to_h5.build_v3_h5   (.stk/.flp -> H5, in a temp file)
    2. validate_inference_h5.validate_sim      (H5 contract check, raises on failure)
    3. build_graph_for_inference.build_graph   (H5 -> PyG Data)

Requires the following files to sit next to this script (or pass explicit paths):
    convert_user_inputs_to_h5.py
    validate_inference_h5.py
    build_graph_for_inference.py
    build_graph_PINN_ready_final_cohort_gated.py   (reference contract)

Usage (CLI):
    python stack_floorplan_to_graph.py <input_dir> [--output graph.pt]

Usage (import):
    from stack_floorplan_to_graph import build_one_graph
    graph = build_one_graph("inputs/")
    graph.x, graph.edge_index, graph.edge_attr, graph.global_features
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import h5py
import torch
from torch_geometric.data import Data

THIS_DIR = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_one_graph(
    input_dir: str | Path,
    sim_name: Optional[str] = None,
    contact_tolerance: float = 0.001,
    max_recovery_gap: Optional[float] = None,
    converter_path: str | Path = THIS_DIR / "convert_user_inputs_to_h5.py",
    validator_path: str | Path = THIS_DIR / "validate_inference_h5.py",
    graph_builder_path: str | Path = THIS_DIR / "build_graph_for_inference.py",
    reference_builder_path: str | Path = THIS_DIR / "build_graph_PINN_ready_final_cohort_gated.py",
) -> Data:
    """
    Input:  a folder containing exactly one .stk file and the .flp files it references.
    Output: one torch_geometric.data.Data graph
                x               [N, 13]
                edge_index      [2, E]
                edge_attr       [E, 11]
                global_features [1, 27]
    Raises ValueError if H5 validation fails; raises on any contract mismatch.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    sim_name = sim_name or f"sim_{uuid.uuid4().hex[:8]}"

    converter = _load_module(Path(converter_path), "sv_converter")
    validator = _load_module(Path(validator_path), "sv_validator")
    graph_builder = _load_module(Path(graph_builder_path), "sv_graph_builder")
    reference_path = Path(reference_builder_path).expanduser().resolve()

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "inference_input.h5"

        # Step 1: .stk + .flp -> H5
        converter.build_v3_h5(str(input_dir), str(h5_path), sim_name)

        # Step 2: validate the H5 contract before it's allowed to become a graph
        with h5py.File(h5_path, "r") as h5f:
            sim = h5f["simulations"][sim_name]
            errors = validator.validate_sim(sim)
            if errors:
                raise ValueError(
                    "H5 validation failed:\n" + "\n".join(f" - {e}" for e in errors)
                )

        # Step 3: H5 -> graph
        ref = graph_builder._load_reference(reference_path)
        with h5py.File(h5_path, "r") as h5f:
            sim = h5f["simulations"][sim_name]
            graph = graph_builder.build_graph(
                sim, h5f, ref,
                contact_tolerance=contact_tolerance,
                max_recovery_gap=max_recovery_gap,
            )
            graph_builder.validate_inference_graph(graph, ref)

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one inference graph from .stk/.flp files")
    parser.add_argument("input_dir", help="Folder containing one .stk file and its .flp files")
    parser.add_argument("--output", default=None, help="Optional path to save the graph as .pt")
    parser.add_argument("--sim-name", default=None)
    parser.add_argument(
        "--reference-builder",
        default=str(THIS_DIR / "build_graph_PINN_ready_final_cohort_gated.py"),
    )
    args = parser.parse_args()

    graph = build_one_graph(
        input_dir=args.input_dir,
        sim_name=args.sim_name,
        reference_builder_path=args.reference_builder,
    )

    print(f"Graph built: {graph.sim_id}")
    print(f"  nodes: {graph.total_node_count} (real={graph.real_node_count}, filler={graph.filler_node_count})")
    print(f"  edges: {graph.edge_index.shape[1]}")
    print(f"  x: {tuple(graph.x.shape)}  edge_attr: {tuple(graph.edge_attr.shape)}  "
          f"global_features: {tuple(graph.global_features.shape)}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, out_path)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()