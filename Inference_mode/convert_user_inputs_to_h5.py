#!/usr/bin/env python3
"""
convert_user_inputs_to_h5_FIXED.py
يحول ملفات .flp + .stk (floor plan + stack) الحقيقية إلى H5 بنفس الـ v3 schema
تماماً زي الملفات اللي اتولدت بواسطة generate_thermal_dataset_v3.py، بس من غير
حقول نتائج الـ solver (اللي محتاجة الموديل يطلعها وقت الـ inference).

الفروقات عن النسخة القديمة:
1. فصل real blocks عن filler blocks (FILL_*) عن TSV pads (TSV*)
2. block_type و kappa بيتحددوا من مادة الـ die في .stk (source line) مش قيمة ثابتة
3. z-stacking بياخد بالحسبان سمك الـ bond مع سمك السيليكون
4. تحويل الوحدات x/y/w/h دايماً /1000 (من غير heuristic خطر)
5. استخراج cooling_type و package_recipe/bonding_tech من الـ .stk
6. إضافة filler_blocks dataset المفقود
7. [FIX] إضافة كل الـ package/global attrs المطلوبة إجباريًا من validate_inference_h5.py
   (rdl_*, interposer_*, c4_*, substrate_*, has_c4) — كانت ناقصة بالكامل قبل كده،
   وكانت بتخلي أي إنبوت يفشل في الـ validation مهما كان مظبوط.
8. [FIX] feature0 (العمود الأول في block_features_merged) اتصحح ليبقى raw power [W]
   مش power_density — ده مطابق للـ contract التاريخي المؤكد من
   build_graph_PINN_ready_final_cohort_gated.py:
       "the historical generator wrote raw block feature column 0 directly to
        3D-ICE `power values`. solver-faithful P_real [W] = raw block_features[:, 0]"
   واسم العمود فضل power_density_W_mm2 بس للتوافق مع الـ schema القديم.
9. [FIX] total_power_W بقت = sum(feature0 * area) بدل sum(feature0) عشان تطابق
   نفس المعادلة اللي بيتأكد منها الـ validator (legacy global_features[2] semantics).
10. [FIX] إضافة ml_feature0_numeric_semantics و global_feature2_numeric_semantics
    و solver_input_total_power_W كـ attrs صريحة، مطلوبين إجباريًا من الـ validator.
"""
import os
import sys
import re
import glob
import json
import numpy as np
import h5py

# ==============================================================================
# المراجع الثابتة (من generate_thermal_dataset_v3.py - مؤكدة من السكرينشوتات)
# ==============================================================================
COOLING_TYPES = {
    "air_passive": 100.0,
    "air_forced": 1000.0,
    "liquid_microchannel": 10000.0,
    "exotic_jet_cooling": 50000.0,
}
COOLING_ORDER = list(COOLING_TYPES.keys())

PACKAGE_RECIPES = {
    "bare_stack_microbump": dict(
        bonding_tech="microbump_underfill", bond_kappa_w_mk=1.2, bond_thickness_um=15.0,
        has_rdl=False, has_interposer=False, has_substrate=False,
    ),
    "bare_stack_hybrid": dict(
        bonding_tech="hybrid_cu_cu", bond_kappa_w_mk=8.0, bond_thickness_um=3.0,
        has_rdl=False, has_interposer=False, has_substrate=False,
    ),
    "cowos_style_si_interposer": dict(
        bonding_tech="microbump_underfill", bond_kappa_w_mk=1.2, bond_thickness_um=12.0,
        has_rdl=True, rdl_kappa_w_mk=15.0, rdl_thickness_um=10.0,
        has_interposer=True, interposer_type="si", interposer_kappa_w_mk=130.0, interposer_thickness_um=100.0,
        has_substrate=True, c4_kappa_w_mk=0.8, c4_thickness_um=50.0,
        substrate_kappa_w_mk=4.0, substrate_thickness_um=800.0,
    ),
    "cowos_style_organic_interposer": dict(
        bonding_tech="microbump_underfill", bond_kappa_w_mk=1.2, bond_thickness_um=12.0,
        has_rdl=True, rdl_kappa_w_mk=15.0, rdl_thickness_um=10.0,
        has_interposer=True, interposer_type="organic", interposer_kappa_w_mk=3.0, interposer_thickness_um=80.0,
        has_substrate=True, c4_kappa_w_mk=0.8, c4_thickness_um=50.0,
        substrate_kappa_w_mk=4.0, substrate_thickness_um=800.0,
    ),
    "foveros_style_hybrid_substrate": dict(
        bonding_tech="hybrid_cu_cu", bond_kappa_w_mk=8.0, bond_thickness_um=2.0,
        has_rdl=True, rdl_kappa_w_mk=25.0, rdl_thickness_um=8.0,
        has_interposer=False,
        has_substrate=True, c4_kappa_w_mk=1.0, c4_thickness_um=40.0,
        substrate_kappa_w_mk=5.0, substrate_thickness_um=1000.0,
    ),
}
PACKAGE_RECIPE_NAMES = list(PACKAGE_RECIPES.keys())


def _nearest_key(value, mapping):
    """يرجع أقرب مفتاح في القاموس لقيمة value"""
    return min(mapping.keys(), key=lambda k: abs(mapping[k] - value))


# ==============================================================================
# PARSING .stk
# ==============================================================================
def parse_stk_file(stk_path):
    with open(stk_path, "r") as f:
        content = f.read()

    # -- المواد: name -> kappa_w_mk (converted native um->m: *1e6), vhc --
    materials = {}
    for m in re.finditer(
        r"material\s+([\w\d_-]+)\s*:\s*thermal conductivity\s+([\d\.\+eE-]+)\s*;\s*volumetric heat capacity\s+([\d\.\+eE-]+)\s*;",
        content,
    ):
        name, kappa_native, vhc = m.group(1), float(m.group(2)), float(m.group(3))
        materials[name] = {"kappa_w_mk": kappa_native * 1e6, "vhc": vhc}

    # -- top heat sink --
    hs_m = re.search(r"top heat sink\s*:\s*heat transfer coefficient\s+([\d\.\+eE-]+)\s*;\s*temperature\s+([\d\.\+eE-]+)\s*;", content)
    h_native = float(hs_m.group(1)) if hs_m else None
    ambient_temp_K = float(hs_m.group(2)) if hs_m else 300.0
    h_SI = h_native * 1e12 if h_native is not None else None

    # -- dimensions --
    dim_m = re.search(r"dimensions\s*:\s*chip length\s+([\d\.\+eE-]+)\s*,\s*width\s+([\d\.\+eE-]+)\s*;\s*cell length\s+([\d\.\+eE-]+)\s*,\s*width\s+([\d\.\+eE-]+)\s*;", content)
    chip_w_um, chip_h_um, cell_um = None, None, None
    if dim_m:
        chip_w_um = float(dim_m.group(1))
        chip_h_um = float(dim_m.group(2))
        cell_um = float(dim_m.group(3))

    # -- die type definitions: source material + layer stack (thickness, material) --
    die_types = {}
    for dm in re.finditer(r"die\s+([\w\d_-]+)\s*:\s*(.*?)(?=\ndie\s+[\w\d_-]+\s*:|\nstack\s*:|$)", content, re.DOTALL):
        dtype_name, dtype_body = dm.group(1), dm.group(2)
        src_m = re.search(r"source\s+\d+\s+([\w\d_-]+)\s*;", dtype_body)
        source_mat = src_m.group(1) if src_m else None
        layers = re.findall(r"layer\s+([\d\.\+eE-]+)\s+([\w\d_-]+)\s*;", dtype_body)
        die_types[dtype_name] = {
            "source_material": source_mat,
            "layers": [(float(th), mat) for th, mat in layers],  # [(thickness_um, material_name), ...]
        }

    # -- stack section: die ID TYPE floorplan "file" --
    stack_m = re.search(r"stack\s*:\s*(.*?)(?=\n\s*solver|\n\s*output|$)", content, re.DOTALL)
    stack_body = stack_m.group(1) if stack_m else ""
    die_entries = re.findall(r"die\s+([\w\d_-]+)\s+([\w\d_-]+)\s+floorplan\s+\"([^\"]+)\"\s*;", stack_body)

    layers_ordered = []
    pkg_entry = None
    layer_idx = 0
    for die_id, die_type, flp_filename in die_entries:
        if "pkg" in flp_filename.lower() or "pkg" in die_id.lower() or "PKG" in die_type:
            pkg_entry = {"die_id": die_id, "die_type": die_type, "flp_filename": flp_filename}
            continue
        layers_ordered.append({
            "layer_idx": layer_idx,
            "die_id": die_id,
            "die_type": die_type,
            "flp_filename": flp_filename,
        })
        layer_idx += 1

    return {
        "materials": materials,
        "h_native": h_native,
        "h_SI": h_SI,
        "ambient_temp_K": ambient_temp_K,
        "chip_w_um": chip_w_um,
        "chip_h_um": chip_h_um,
        "cell_um": cell_um,
        "die_types": die_types,
        "layers_ordered": layers_ordered,
        "pkg_entry": pkg_entry,
    }


# ==============================================================================
# PARSING .flp
# ==============================================================================
def parse_flp_file(flp_path):
    """يرجع 3 لستات: real_blocks, filler_blocks, tsv_pads (كل عنصر dict بالـ geometry والـ power)"""
    real_blocks, filler_blocks, tsv_pads = [], [], []
    if not os.path.exists(flp_path):
        return real_blocks, filler_blocks, tsv_pads

    with open(flp_path, "r") as f:
        content = f.read()

    block_matches = re.findall(r"([\w\d_-]+)\s*:\s*([^:]+?)(?=\n[\w\d_-]+\s*:|\Z)", content)

    for b_name, b_body in block_matches:
        b_name = b_name.strip()
        pos_m = re.search(r"position\s+([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)", b_body)
        dim_m = re.search(r"dimension\s+([\d\.\+eE-]+)\s*,\s*([\d\.\+eE-]+)", b_body)
        pwr_m = re.search(r"power\s+values\s+([^;]+);", b_body)
        if not (pos_m and dim_m):
            continue

        # الوحدات في .flp دايماً micrometers -> نحولها لـ mm بقسمة ثابتة على 1000
        x_mm = float(pos_m.group(1)) / 1000.0
        y_mm = float(pos_m.group(2)) / 1000.0
        w_mm = float(dim_m.group(1)) / 1000.0
        h_mm = float(dim_m.group(2)) / 1000.0

        power_val = 0.0
        if pwr_m:
            p_tokens = re.findall(r"[\d\.\+eE-]+", pwr_m.group(1))
            if p_tokens:
                power_val = float(p_tokens[0])

        item = {"name": b_name, "x_mm": x_mm, "y_mm": y_mm, "w_mm": w_mm, "h_mm": h_mm, "power_W": power_val}

        if b_name.startswith("FILL_"):
            filler_blocks.append(item)
        elif b_name.startswith("TSV"):
            tsv_pads.append(item)
        else:
            real_blocks.append(item)

    return real_blocks, filler_blocks, tsv_pads


# ==============================================================================
# تحديد cooling_type و package_recipe
# ==============================================================================
def resolve_cooling(h_SI):
    if h_SI is None:
        return "air_forced", COOLING_ORDER.index("air_forced")
    name = _nearest_key(h_SI, COOLING_TYPES)
    return name, COOLING_ORDER.index(name)


def resolve_package_recipe(stk_data):
    """
    مهم جداً: قائمة الـ materials في .stk ثابتة دايماً (مكتبة تعريفات عامة) بغض
    النظر عن التصميم الفعلي، فمينفعش نحدد has_rdl/has_interposer/has_substrate
    من مجرد وجود اسم المادة في القائمة العامة (هيطلع True دايماً). لازم نفحص
    طبقات الـ PKG_DIETYPE نفسه (die الـ package/substrate) ونشوف هو فعلاً
    بيستخدم المواد دي في الـ layers بتاعته ولا لأ.
    """
    materials = stk_data["materials"]

    bond_kappa = None
    for dt in stk_data["die_types"].values():
        for th, mat in dt["layers"]:
            if mat == "BOND_MAT" and "BOND_MAT" in materials:
                bond_kappa = materials["BOND_MAT"]["kappa_w_mk"]
                break
        if bond_kappa is not None:
            break

    # فحص طبقات الـ PKG die type تحديداً (مش القائمة العامة للمواد)
    pkg_entry = stk_data["pkg_entry"]
    pkg_layer_materials = set()
    if pkg_entry is not None:
        pkg_die_type = stk_data["die_types"].get(pkg_entry["die_type"], {})
        pkg_layer_materials = {mat for th, mat in pkg_die_type.get("layers", [])}

    has_rdl = "RDL_MAT" in pkg_layer_materials
    has_interposer = "INTERPOSER_MAT" in pkg_layer_materials
    has_substrate = "SUBSTRATE_MAT" in pkg_layer_materials

    interposer_type = "none"
    if has_interposer and "INTERPOSER_MAT" in materials:
        interposer_type = "si" if abs(materials["INTERPOSER_MAT"]["kappa_w_mk"] - 130.0) < 10 else "organic"

    # نفتش عن أقرب recipe مطابق للـ flags دي (مع الفصل بين si/organic كفيصل حاسم)
    best_name, best_score = None, -1
    for name, r in PACKAGE_RECIPES.items():
        score = 0
        if r.get("has_rdl", False) == has_rdl:
            score += 1
        if r.get("has_interposer", False) == has_interposer:
            score += 1
        if r.get("has_substrate", False) == has_substrate:
            score += 1
        if bond_kappa is not None and abs(r["bond_kappa_w_mk"] - bond_kappa) < 0.5:
            score += 2
        if has_interposer and r.get("interposer_type") == interposer_type:
            score += 3
        if score > best_score:
            best_score, best_name = score, name

    return {
        "package_recipe": best_name or "bare_stack_microbump",
        "package_recipe_code": PACKAGE_RECIPE_NAMES.index(best_name) if best_name else 0,
        "bonding_tech": PACKAGE_RECIPES[best_name]["bonding_tech"] if best_name else "microbump_underfill",
        "has_rdl": int(has_rdl),
        "has_interposer": int(has_interposer),
        "has_substrate": int(has_substrate),
        "interposer_type": interposer_type,
    }


# ==============================================================================
# البناء الرئيسي
# ==============================================================================
def build_v3_h5(input_dir, output_h5_path, sim_name="sim_550000"):
    stk_files = glob.glob(os.path.join(input_dir, "*.stk"))
    if not stk_files:
        raise FileNotFoundError("لم يتم العثور على ملف .stk في المجلد: " + input_dir)

    stk_data = parse_stk_file(stk_files[0])
    materials = stk_data["materials"]

    merged_features = []   # real blocks فقط: power_density,area,layer_idx,kappa,block_type,has_tsv,is_hotspot,x,y,w,h,xc,yc,zc,thick
    block_names = []
    filler_rows = []       # x,y,w,h,layer_idx,z_center,thickness
    tsv_pads_rows = []     # x,y,w,h,layer_idx,z_center,thickness
    layer_thicknesses_um = []
    n_dies_per_layer = []
    total_power_w = 0.0

    current_z_mm = 0.0

    for layer_entry in stk_data["layers_ordered"]:
        layer_idx = layer_entry["layer_idx"]
        die_type = stk_data["die_types"].get(layer_entry["die_type"], {})
        source_mat = die_type.get("source_material", "LOGIC_MAT")
        block_type = 2.0 if source_mat == "MEMORY_MAT" else 1.0
        kappa_w_mk = materials.get(source_mat, {}).get("kappa_w_mk", 130.0)

        # سمك الطبقة = سيليكون + bond (لو موجود) -- z_includes_bond_thickness=1
        total_th_um = sum(th for th, mat in die_type.get("layers", []))
        if total_th_um <= 0:
            total_th_um = 100.0
        total_th_mm = total_th_um / 1000.0
        z_center_mm = current_z_mm + (total_th_mm / 2.0)
        layer_thicknesses_um.append(total_th_um)

        flp_path = os.path.join(input_dir, layer_entry["flp_filename"])
        real_blocks, filler_blocks, tsv_pads = parse_flp_file(flp_path)
        n_dies_per_layer.append(len(real_blocks))

        for b in real_blocks:
            w, h = b["w_mm"], b["h_mm"]
            area = w * h
            power = b["power_W"]
            # global_features[2] in the trained contract uses the legacy H5 convention:
            # sum(raw_feature_0 * area), not the physical solver-input total.
            total_power_w += power * area
            # IMPORTANT: match the historical V3/Fine-tuning ML contract exactly.
            # The field is *named* power_density_W_mm2 for backward compatibility,
            # but the historical generator/model saw the raw block power numeric value [W].
            # Do NOT divide by area here, otherwise inference distribution differs from training.
            power_density = power
            x_min, y_min = b["x_mm"], b["y_mm"]
            x_center, y_center = x_min + w / 2.0, y_min + h / 2.0

            merged_features.append([
                power_density, area, float(layer_idx), kappa_w_mk, block_type,
                0.0, 0.0,  # has_tsv, is_hotspot -- غير معروفين من الـ floorplan لوحده
                x_min, y_min, w, h, x_center, y_center, z_center_mm, total_th_mm,
            ])
            # اسم البلوك في .flp أصلاً بييجي فيه بادئة الطبقة (زي "L0_Die0")،
            # فمينفعش نضيف بادئة تانية فوقها (كان بيطلع "L0_L0_Die0" غلط)
            block_names.append(b["name"].encode("utf-8"))

        for fb in filler_blocks:
            w, h = fb["w_mm"], fb["h_mm"]
            x_min, y_min = fb["x_mm"], fb["y_mm"]
            filler_rows.append([x_min, y_min, w, h, float(layer_idx), z_center_mm, total_th_mm])

        for t in tsv_pads:
            w, h = t["w_mm"], t["h_mm"]
            x_min, y_min = t["x_mm"], t["y_mm"]
            tsv_pads_rows.append([x_min, y_min, w, h, float(layer_idx), z_center_mm, total_th_mm])

        current_z_mm += total_th_mm

    merged_arr = np.array(merged_features, dtype=np.float32)
    filler_arr = np.array(filler_rows, dtype=np.float32) if filler_rows else np.zeros((0, 7), dtype=np.float32)
    tsv_arr = np.array(tsv_pads_rows, dtype=np.float32) if tsv_pads_rows else np.zeros((0, 7), dtype=np.float32)

    cooling_name, cooling_code = resolve_cooling(stk_data["h_SI"])
    pkg_info = resolve_package_recipe(stk_data)
    bond_kappa = materials.get("BOND_MAT", {}).get("kappa_w_mk", 1.2)

    with h5py.File(output_h5_path, "w") as h5f:
        sims_grp = h5f.create_group("simulations")
        sim_0 = sims_grp.create_group(sim_name)

        sim_0.create_dataset("block_features_merged", data=merged_arr, compression="gzip")
        sim_0.create_dataset("block_names", data=block_names)
        sim_0.create_dataset("filler_blocks", data=filler_arr, compression="gzip")
        sim_0.create_dataset("tsv_pads", data=tsv_arr, compression="gzip")

        sim_0.attrs["ambient_temp_K"] = stk_data["ambient_temp_K"]
        sim_0.attrs["block_features_columns"] = "power_density_W_mm2,area_mm2,layer_idx,kappa_W_mK,block_type(1=logic,2=memory),has_tsv(0/1),is_hotspot(0/1)"
        sim_0.attrs["block_features_merged_columns"] = "power_density_W_mm2,area_mm2,layer_idx,kappa_W_mK,block_type(1=logic,2=memory),has_tsv(0/1),is_hotspot(0/1),x_mm,y_mm,width_mm,height_mm,x_center_mm,y_center_mm,z_center_mm,thickness_mm"
        sim_0.attrs["filler_blocks_columns"] = "x_mm,y_mm,width_mm,height_mm,layer_idx,z_center_mm,thickness_mm"
        sim_0.attrs["tsv_pads_columns"] = "x_mm,y_mm,width_mm,height_mm,layer_idx,z_center_mm,thickness_mm"
        sim_0.attrs["filler_kappa_w_mk"] = materials.get("LOGIC_MAT", {}).get("kappa_w_mk", 130.0)
        sim_0.attrs["filler_temp_note"] = (
            "Filler cell temperatures are available directly in point_cloud (real solver output, non-zero). "
            "node_temperature_labels intentionally contains ONLY real (non-filler) blocks -- do not zero-fill "
            "filler temperature when building GNN target y; source it from point_cloud and keep target_mask=False "
            "for filler nodes in the physics loss."
        )
        sim_0.attrs["data_source"] = "3dice_real_fea_v3_self_contained"

        sim_0.attrs["chip_w_um"] = stk_data["chip_w_um"] or 0.0
        sim_0.attrs["chip_h_um"] = stk_data["chip_h_um"] or 0.0
        sim_0.attrs["cell_um"] = stk_data["cell_um"] or 50.0

        sim_0.attrs["cooling_htc_w_m2k"] = stk_data["h_SI"] or COOLING_TYPES["air_forced"]
        sim_0.attrs["cooling_type"] = cooling_name
        sim_0.attrs["cooling_type_code"] = cooling_code

        sim_0.attrs["bonding_tech"] = pkg_info["bonding_tech"]
        sim_0.attrs["bond_kappa_w_mk"] = bond_kappa
        sim_0.attrs["bond_thickness_um"] = PACKAGE_RECIPES[pkg_info["package_recipe"]]["bond_thickness_um"]
        sim_0.attrs["package_recipe"] = pkg_info["package_recipe"]
        sim_0.attrs["package_recipe_code"] = pkg_info["package_recipe_code"]
        sim_0.attrs["has_rdl"] = pkg_info["has_rdl"]
        sim_0.attrs["has_interposer"] = pkg_info["has_interposer"]
        sim_0.attrs["has_package_substrate"] = pkg_info["has_substrate"]
        sim_0.attrs["interposer_type"] = pkg_info["interposer_type"]

        # Preserve the same package-global inputs consumed by the 27-feature
        # training graph contract. These are deterministic from package_recipe.
        recipe = PACKAGE_RECIPES[pkg_info["package_recipe"]]
        sim_0.attrs["rdl_thickness_um"] = float(recipe.get("rdl_thickness_um", 0.0))
        sim_0.attrs["rdl_kappa_w_mk"] = float(recipe.get("rdl_kappa_w_mk", 0.0))
        sim_0.attrs["interposer_thickness_um"] = float(recipe.get("interposer_thickness_um", 0.0))
        sim_0.attrs["interposer_kappa_w_mk"] = float(recipe.get("interposer_kappa_w_mk", 0.0))
        sim_0.attrs["has_c4"] = int(recipe.get("has_substrate", False) and "c4_thickness_um" in recipe)
        sim_0.attrs["c4_thickness_um"] = float(recipe.get("c4_thickness_um", 0.0))
        sim_0.attrs["c4_kappa_w_mk"] = float(recipe.get("c4_kappa_w_mk", 0.0))
        sim_0.attrs["substrate_thickness_um"] = float(recipe.get("substrate_thickness_um", 0.0))
        sim_0.attrs["substrate_kappa_w_mk"] = float(recipe.get("substrate_kappa_w_mk", 0.0))

        sim_0.attrs["layer_thicknesses_um"] = np.array(layer_thicknesses_um, dtype=np.float32)
        sim_0.attrs["n_layers"] = len(stk_data["layers_ordered"])
        sim_0.attrs["n_dies_per_layer"] = np.array(n_dies_per_layer, dtype=np.int32)
        sim_0.attrs["n_tsv_pairs"] = len(tsv_pads_rows)
        sim_0.attrs["has_hotspot"] = 0
        sim_0.attrs["hotspot_indices_per_layer_json"] = json.dumps([[] for _ in stk_data["layers_ordered"]])
        sim_0.attrs["hotspot_multiplier"] = 1.0
        sim_0.attrs["non_uniform_material"] = 1
        sim_0.attrs["pass_fail"] = 0
        sim_0.attrs["sim_id"] = sim_name
        sim_0.attrs["total_power_W"] = total_power_w
        sim_0.attrs["workload_factor"] = 0.75

        # ملاحظة صريحة: دي مدخلات inference، مش نتيجة solver
        sim_0.attrs["is_inference_input_not_solver_output"] = 1
        sim_0.attrs["ml_feature0_numeric_semantics"] = "historical_v3_raw_power_W_despite_legacy_name"
        sim_0.attrs["global_feature2_numeric_semantics"] = "historical_v3_legacy_sum_feature0_times_area"
        sim_0.attrs["solver_input_total_power_W"] = float(sum(float(row[0]) for row in merged_features))
        sim_0.attrs["max_temp_K"] = -1.0  # placeholder -- يتحسب بواسطة GNN، مش موجود هنا


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 convert_user_inputs_to_h5_FIXED.py <input_dir> <output.h5> [sim_name]")
        sys.exit(1)
    sim_name = sys.argv[3] if len(sys.argv) > 3 else "sim_550000"
    build_v3_h5(os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]), sim_name)
    print(f"تم إنشاء: {sys.argv[2]}")


if __name__ == "__main__":
    main()