#!/usr/bin/env python3
"""
extract_ground_truth_temperatures.py
=====================================
يطلع الـ ground truth الحقيقي (متوسط حرارة كل بلوك) من ملفات tflp_*.txt
اللي طلعها 3D-ICE فعليًا (output: Tflp (..., average, final)).

مش محتاج tmap ولا xyaxis: أسماء الأعمدة في tflp مطابقة حرفيًا لأسماء
البلوكات (real + filler + TSV) في ملفات .flp بتاعة نفس الـ layer.

Usage:
    from extract_ground_truth_temperatures import build_ground_truth
    gt = build_ground_truth("path/to/sim_folder")
    # gt = {"L0_Die0": 468.447, "FILL_L0_d0": 461.545, ...}

    # أو عشان تجيب ترتيب مطابق لـ block_names في الـ H5 (نفس ترتيب real_blocks
    # اللي بيطلعها convert_user_inputs_to_h5.parse_flp_file):
    from convert_user_inputs_to_h5 import parse_stk_file, parse_flp_file
    import numpy as np, os

    stk_data = parse_stk_file(os.path.join(input_dir, "sim.stk"))
    gt = build_ground_truth(input_dir)

    # === مهم جداً: لازم تعمل PASS للـ real لوحدهم على الاستاك كله، وبعدين PASS
    # تاني للـ filler لوحدهم على الاستاك كله -- بالظبط زي convert_user_inputs_to_h5.py.
    # لو عملتهم مع بعض جوه نفس الـ loop لكل layer (real+filler لكل layer)، الترتيب
    # هيبقى غلط وهيقارن node غلط بـ node تاني (real layer 1 مع filler layer 0 مثلاً).

    y_all_in_graph_order = []

    # Pass 1: كل الـ real blocks على الاستاك كله (بنفس ترتيب block_features_merged)
    for layer_entry in stk_data["layers_ordered"]:
        flp_path = os.path.join(input_dir, layer_entry["flp_filename"])
        real_blocks, filler_blocks, tsv_pads = parse_flp_file(flp_path)
        for b in real_blocks:
            y_all_in_graph_order.append(gt[b["name"]])

    # Pass 2: كل الـ filler blocks على الاستاك كله (بنفس ترتيب filler_blocks dataset)
    for layer_entry in stk_data["layers_ordered"]:
        flp_path = os.path.join(input_dir, layer_entry["flp_filename"])
        real_blocks, filler_blocks, tsv_pads = parse_flp_file(flp_path)
        for fb in filler_blocks:
            y_all_in_graph_order.append(gt[fb["name"]])

    y_all_in_graph_order = np.array(y_all_in_graph_order, dtype=np.float32)
    # y_all_in_graph_order[i] دلوقتي متطابق تمامًا مع graph.x[i] (graph.real_node_mask/filler_node_mask)
"""
from __future__ import annotations
import glob
import os
import re
from typing import Dict


def parse_tflp_file(tflp_path: str) -> Dict[str, float]:
    """
    يرجع dict: block_name -> temperature_K من ملف tflp واحد.
    بيتوقع "average, final" يعني سطر بيانات واحد بس بعد الهيدر (steady state).
    لو فيه أكتر من سطر بيانات (transient)، بياخد آخر سطر (الأحدث/الأقرب لـ final).
    """
    with open(tflp_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    header_line = None
    data_lines = []
    for ln in lines:
        if ln.startswith("%") and "Time" in ln:
            header_line = ln
        elif ln.startswith("%"):
            continue
        else:
            data_lines.append(ln)

    if header_line is None:
        raise ValueError(f"{tflp_path}: couldn't find header line with column names")
    if not data_lines:
        raise ValueError(f"{tflp_path}: no data rows found")

    # الأعمدة مفصولة بـ tab، وكل اسم عمود منتهي بـ "(K)"
    columns = [c.strip() for c in header_line.lstrip("%").split("\t") if c.strip()]
    # أول عمود هو "Time(s)" -- نتجاهله
    block_names = [re.sub(r"\(K\)$", "", c).strip() for c in columns[1:]]

    last_row = data_lines[-1]
    values = [v.strip() for v in last_row.split("\t") if v.strip()]
    temps = [float(v) for v in values[1:]]  # نتجاهل عمود الوقت

    if len(temps) != len(block_names):
        raise ValueError(
            f"{tflp_path}: column/value count mismatch "
            f"({len(block_names)} names vs {len(temps)} values)"
        )

    return dict(zip(block_names, temps))


def build_ground_truth(input_dir: str, pattern: str = "tflp_*.txt") -> Dict[str, float]:
    """
    يقرا كل ملفات tflp_*.txt في المجلد ويرجع dict واحد شامل:
    block_name -> temperature_K (لكل الـ real + filler + TSV blocks في كل الطبقات)
    """
    tflp_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not tflp_files:
        raise FileNotFoundError(f"No tflp files found in: {input_dir}")

    ground_truth: Dict[str, float] = {}
    for path in tflp_files:
        layer_gt = parse_tflp_file(path)
        overlap = set(layer_gt) & set(ground_truth)
        if overlap:
            raise ValueError(f"Duplicate block names across tflp files: {overlap}")
        ground_truth.update(layer_gt)

    return ground_truth


def build_ground_truth_in_graph_order(input_dir: str) -> "list[float]":
    """
    يرجع ground truth كـ list واحدة بنفس ترتيب graph.x بالظبط:
    [كل الـ real nodes على الاستاك كله بالترتيب] + [كل الـ filler nodes على الاستاك كله بالترتيب]

    هام: ده الترتيب اللي بيبنيه convert_user_inputs_to_h5.build_v3_h5 فعليًا
    (pass كامل للـ real الأول عبر كل الطبقات، بعدين pass كامل للـ filler) --
    مش real+filler لكل layer على حدة، لأن ده هيدّي ترتيب غلط.
    """
    from convert_user_inputs_to_h5 import parse_stk_file, parse_flp_file

    stk_path = os.path.join(input_dir, "sim.stk")
    stk_data = parse_stk_file(stk_path)
    gt = build_ground_truth(input_dir)

    ordered = []
    # Pass 1: real blocks على الاستاك كله
    for layer_entry in stk_data["layers_ordered"]:
        flp_path = os.path.join(input_dir, layer_entry["flp_filename"])
        real_blocks, filler_blocks, tsv_pads = parse_flp_file(flp_path)
        for b in real_blocks:
            ordered.append(gt[b["name"]])
    # Pass 2: filler blocks على الاستاك كله
    for layer_entry in stk_data["layers_ordered"]:
        flp_path = os.path.join(input_dir, layer_entry["flp_filename"])
        real_blocks, filler_blocks, tsv_pads = parse_flp_file(flp_path)
        for fb in filler_blocks:
            ordered.append(gt[fb["name"]])

    return ordered


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract ground-truth block temperatures from tflp files")
    parser.add_argument("input_dir", help="Folder containing tflp_*.txt files")
    parser.add_argument("--output", default=None, help="Optional path to save as JSON")
    args = parser.parse_args()

    gt = build_ground_truth(args.input_dir)
    print(f"Loaded {len(gt)} block temperatures from {args.input_dir}")
    for name, temp in list(gt.items())[:10]:
        print(f"  {name}: {temp:.3f} K")
    if len(gt) > 10:
        print(f"  ... and {len(gt) - 10} more")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()