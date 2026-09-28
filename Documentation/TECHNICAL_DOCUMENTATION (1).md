# Technical Documentation — 3D-IC Thermal Surrogate GNN

## 1. System Overview

A physics-informed Graph Neural Network that predicts steady-state per-block temperature for 3D-IC packages, replacing a full 3D-ICE physics simulation with a single forward pass. See `README.md` for the high-level architecture diagrams; this document covers module-by-module and function-by-function detail.

## 2. Directory / Module Map (as uploaded, flat)

```text
Dataset.py                                   — dataset assembly, scaler fitting, distribution-matched split
process_one_graph.py                         — inference-time (single graph) preprocessing pipelines
Train_data_processin.py                      — training-set batch preprocessing pipelines
Val_data_processing.py                       — validation-set batch preprocessing pipelines
Test_data_processing.py                      — test-set batch preprocessing pipelines
prepare_Data.py                              — orchestrates preprocessing + DataLoader construction for train/val/test
Training_v1_playground.py                    — physics-informed training loop (current/best model)
Testing_phase.py                             — full test-set evaluation
FASTAPI.py                                   — inference API (older config: ReLU model, ultra_super_power_V2000 checkpoint)
Final_app.py                                 — model/checkpoint loader (newer config: LeakyReLU, Model_v7_playground)
Stack_floorplan_to_graph.py                  — .stk/.flp -> H5 -> graph, one-shot inference wrapper
Extract_ground_truth_temperatures.py         — parses 3D-ICE tflp_*.txt output into a name->temperature map
Test_inference_mode.py                       — manual end-to-end sanity check script (prediction vs ground truth)
plot_predictionsVSGroundTruth.py             — custom thermal-map visualization
app.py                                       — Streamlit frontend
```

Referenced but not included in this documentation pass (present on disk under other names, not among the files provided for this task): `convert_user_inputs_to_h5.py`, `validate_inference_h5.py`, `build_graph_for_inference.py`, `build_graph_PINN_ready_final_cohort_gated.py`. `[Needs clarification]` for their internals — module docs below describe only what other files' calls into them reveal.

## 3. Module-by-Module Documentation

### `Dataset.py`

**Responsibility**: builds the train/val/test splits and fits every scaler/one-hot cardinality used everywhere else in the pipeline.

**Key steps, in order**:
1. Loads 4 raw `.pt` graph-dataset files (paths hardcoded to `D:/Thermal surrogate project/Datasets/...`) — a base dataset plus three supplementary batches explicitly targeting the 400-500 K range, and concatenates them.
2. `remove_graphs_outside_temperature_range(dataset, min_temp=300.0, max_temp=500.0)` — drops any graph with even one node outside range, printing counts removed/remaining.
3. Builds `bins = [300, 350, 400, 450, 500]` and, for `seed in range(100)`, does `train_test_split(indices, test_size=0.20)` then splits the remaining 20% 50/50 into val/test, computes each split's per-bin percentage distribution via `get_distribution()`, and scores the seed by the mean absolute difference between train/val, train/test, and val/test distributions. Keeps the seed with the lowest (best-matched) score.
4. Builds `Nodes_data`, `Edges_data`, `global_data` as flat pandas DataFrames by concatenating every graph's `x`, `edge_attr`, `global_features` (train split only) — used purely to fit scalers on the full population of training-set values.
5. Fits and dumps (via `joblib.dump`, default filenames, current working directory) three scaler dicts and two one-hot metadata dicts:
   - `scallers_Node_data` = `{std_power_density, std_area, std_kappa, std_exposed_area_mm2}` (area/kappa/exposed-area fit on `log1p`-transformed values)
   - `one_hot_Node_data` = `{Number_of_layers, Number_of_block_types}`
   - `scallers_Edege_data` = `{std_resistance, std_geometric_distance, std_contact_area, std_thermal_path_length}` (resistance/contact-area/path-length fit on `log1p`)
   - `one_hot_edge_data` = `{Number_of_edge_types}`
   - `scallers_global_data` = `{std_total_power_W, std_cooling_htc, std_bond_kappa_w_mk, std_bond_thickness_mm, std_ambient_temp_K, std_workload_factor, std_n_tsv_pairs, std_hotspot_multiplier, std_chip_width_mm, std_chip_height_mm}`
   - `one_hot_global_data` = `{Number_of_cooling_types, Number_of_package_recipe_code}`
6. Fits `std_target_temp` (a `StandardScaler`) on `log(y)` across all training-graph node temperatures, and dumps `normalized_temp_scaller = {"std_target_temp": std_target_temp}`.
7. Contains local copies of `Train_Nodes_data_pipleline`, `Train_Edges_data_pipleline`, `Train_Global_data_pipleline`, `Train_Global_data_pipleline_new`, `normalize_temp` — functionally identical to the versions in `Train_data_processin.py`, defined here so the fitted scalers can be validated/used immediately after fitting in the same script. **Duplicated logic across files** rather than a shared import — worth consolidating.

**Called by**: `prepare_Data.py` imports `train_data, test_data, val_data` directly from this module (i.e., running `prepare_Data.py` executes all of `Dataset.py`'s module-level code as an import side effect).

---

### `process_one_graph.py` / `Train_data_processin.py` / `Val_data_processing.py` / `Test_data_processing.py`

**Responsibility**: apply the fitted scalers/one-hot encodings to graphs. `process_one_graph.py` operates on **one** graph (used at inference); the other three operate on a **list** of graphs (used at training/eval time), and are otherwise structurally identical to each other and to `process_one_graph.py` (same column indices, same transforms), differing only in variable name prefixes (`Train_`/`val_`/`test_`) and in `process_one_graph.py` loading scalers from an absolute `ARTIFACT_DIR` (`C:/msys64/home/dell`) rather than the current working directory.

**Functions (per file, same signatures modulo the Train/Val/test/single-graph naming)**:

- **`{Train,Val,test}_Nodes_data_pipleline(data)`**
  - `keep = [0, 1, 3, 5, 6, 12]` — power_density, area, kappa, has_tsv, is_hotspot, exposed_area (columns 2=layer_idx and 4=block_type are consumed by one-hot encoding instead of being kept raw).
  - One-hot encodes column 2 (`layer_idx`) and column 4 (`block_type_code`).
  - Computes `relative_depth`: `layer_idx / max(layer_idx)` in Val/Test/single-graph versions, but `layer_idx / 4` (a **hardcoded constant**, not the graph's own max) in `Train_data_processin.py`'s function. `[Needs clarification]`: this is an inconsistency between the train-time and val/test/inference-time normalization of the same feature — confirm this is intentional (e.g., 4 is a known fixed max layer count across the whole dataset) rather than a bug, since a per-graph `max()` and a fixed `/4` only agree if every graph has exactly 4 layers.
  - Scales power_density, `log1p`+scales area/kappa/exposed_area via the loaded `StandardScaler`s.
  - Computes `relative_width`, `relative_height`, `relative_x`, `relative_y` by dividing raw node position/size columns by the graph's own `global_features[0, 25]` (chip width) / `[0, 26]` (chip height).
  - Concatenates: kept raw (6) + one-hot layer + one-hot block type + 5 engineered features = 18 total (`in_features` in the model config).
  - Returns the graph(s) with `graph.x` replaced in place.

- **`{Train,Val,test}_Edges_data_pipleline(data)`**
  - `keep = [0,1,2,3,4,8,9,10]`.
  - One-hot encodes column 7 (`edge_type`).
  - `log1p`+scales thermal_resistance (col 0), contact_overlap_area (col 4), thermal_path_length (col 8); scales geometric_distance (col 2) directly (no `log1p`).
  - Concatenates: kept raw (8) + one-hot edge type = 11 total (`in_features_edge`).

- **`{Train,Val,test}_Global_data_pipleline(data)`** (older version)
  - `KEEP_GLOBAL = [0, 2, 3, 8, 11, 12]`.
  - One-hot encodes cooling type (col 7).
  - `log1p`+scales total_power (col 2), cooling_htc (col 8), bond_kappa (col 11); scales bond_thickness (col 12) directly; divides n_layers (col 3) by 4.
  - Computes `R_pkg` = sum of (thickness / conductivity), clamped to avoid division by ~0, across 4 package layers (RDL: cols 14/15, Interposer: 17/18, C4: 20/21, Substrate: 23/24).
  - Concatenates kept (6) + `R_pkg` (1) + one-hot cooling type = variable width depending on `Number_of_cooling_types`.

- **`{Train,val,test}_Global_data_pipleline_new(data)`** (current version, used in `set_up_file`-equivalent live paths)
  - `KEEP_GLOBAL = [0, 2, 3, 5, 8, 11, 12, 13, 16, 19, 22]` — adds ambient_temp (0), workload_factor (1, kept but not listed since it's transformed separately — see below), n_tsv_pairs is transformed but not directly in `KEEP_GLOBAL` list (transformed value lives at column 4, not in the keep list — the transform is applied but the **raw untransformed** slot for n_tsv_pairs is not retained in the final concat unless captured by one of the listed indices; verify against your own column schema if extending this) `[Needs clarification: cross-check exact column-to-semantic mapping against your data-generation schema before modifying]`.
  - Also one-hot encodes `package_recipe_code` (col 9) in addition to cooling type (col 7).
  - Transforms (all in-place on `graph.global_features`): ambient_temp (col 0, scaled, no log), workload_factor (col 1, scaled, no log), total_power (col 2, log1p+scaled), n_layers (col 3, /4), n_tsv_pairs (col 4, log1p+scaled), hotspot_multiplier (col 6, log1p+scaled), cooling_htc (col 8, log1p+scaled), bond_kappa (col 11, log1p+scaled), bond_thickness (col 12, scaled), chip_width (col 25, scaled), chip_height (col 26, scaled).
  - Computes the same `R_pkg` feature as the older pipeline.
  - Concatenates: `KEEP_GLOBAL` columns (11) + `R_pkg` (1) + one-hot cooling type + one-hot package recipe = 21 total (`global_features` in model config), given the specific cardinalities in this dataset.

- **`{Train,val,test}_normalize_temp(data)`** (not present in `process_one_graph.py` — single-graph inference has no ground truth to normalize)
  - `graph.T_real_normalized = StandardScaler.transform(log(graph.y))`, squeezed to 1D. This is the actual training target; `graph.y` is left untouched in real Kelvin for evaluation/plotting.

**Called by**: `prepare_Data.py` (Train/Val/Test batch versions), `Training_v1_playground.py` (imports these but has the calls commented out in the uploaded version, using `loaders["train"].dataset` — already-preprocessed — directly instead; see that module's notes), `FASTAPI.py` and `Test_inference_mode.py` (the single-graph `process_one_graph.py` versions, via `Train_Nodes_data_pipleline` etc. imported from the `Preprocessing` package).

---

### `prepare_Data.py`

**Responsibility**: orchestration script that takes the raw split produced by `Dataset.py`, runs it through the batch preprocessing pipelines, builds `DataLoader`s, and reports diagnostics.

**Flow**:
1. Imports `train_data, test_data, val_data` from `Dataset.py` (executing that module fully).
2. Saves the three raw (pre-preprocessing) splits to disk as `.pt` files.
3. Runs `Train_Nodes_data_pipleline` -> `Train_Edges_data_pipleline` -> `Train_Global_data_pipleline_new` -> `Train_normalize_temp` on the train split, and the equivalent Val/Test functions on their splits.
4. `analyze_temperature_distribution(dataset)` — prints node counts and percentages per 50 K bin (300-350...450-500) for train and val (not test).
5. Builds `loaders = {"train": DataLoader(..., batch_size=8, shuffle=True), "val": DataLoader(..., batch_size=8, shuffle=True), "test": DataLoader(..., batch_size=8, shuffle=True)}` and dumps it via `joblib.dump(value=loaders, filename="loaders")` — this `loaders` file is the shared artifact every downstream training/testing script loads.
6. Prints a loader-shape sanity check (first train batch's `x`, `edge_attr`, `global_features`, and `T_real_normalized`/`y` shapes).
7. `analyze_target(data, name)` — prints count/min/max/mean/std/median/percentiles (1/5/25/50/75/95/99) of the raw Kelvin target for train and val.

**Note**: despite `Dataset.py`'s own split-search using an 80/10/10 target, `prepare_Data.py`'s `DataLoader`s use `batch_size=8`, matching `Training_v1_playground.py`'s `CONFIG["batch_size"]`, not the `BATCH_SIZE = 16` constant defined (but not directly used for the final loaders) inside `Dataset.py`.

---

### `Training_v1_playground.py`

**Responsibility**: the physics-informed training loop for the current/best model line (`Model_v7_playground.pt`).

**Data loading**: loads the pre-built `loaders` artifact (`load(filename="loaders")`) rather than re-running the Train/Val/Test preprocessing pipelines directly — `Fine_tuning_data = list(loaders["train"].dataset)`. A large commented-out block shows an earlier version that additionally loaded and preprocessed several extra hard/high-temperature datasets and combined them with the original train set for "fine-tuning" (oversampling harder cases); as uploaded, that block is disabled and training runs on `loaders["train"].dataset` alone, filtered again by `remove_graphs_outside_temperature_range` (belt-and-suspenders re-check on data already filtered in `Dataset.py`).

**Classes**:
- **`RangeBalancedMLLoss(nn.Module)`** — `forward(prediction_normalized, target_normalized, target_kelvin)`: per-node squared error in normalized space, weighted by `soft_range_weights(target_kelvin, ranges, margin_K)` and averaged per range, then averaged across the (up to 4) non-empty ranges.
- **`RangeBalancedRealSpaceLoss(nn.Module)`** — same range-balancing structure, but the per-node loss is `F.huber_loss` on `(prediction_kelvin - target_kelvin) / scale_K` (delta=1.0, `scale_K=30.0` from `CONFIG["real_space_loss_scale_K"]`).
- **`ML_only_model(nn.Module)`** — see Section 5 of `README.md`; this file's copy uses `nn.LeakyReLU(negative_slope=0.01)` everywhere (input projection, edge MLP, global encoder, FC head, and the explicit `F.leaky_relu` call inside the message-passing loop).

**Key functions**:
- `soft_range_weights(target_kelvin, ranges, margin_K)` — returns a `[num_ranges, N]` weight tensor; each node's weights sum to 1; nodes within `margin_K/2` of an internal boundary are linearly split between the two neighboring ranges (see docstring example table in the source for exact blending values at 390/395/400/405/410 K with a 400 K boundary and 10 K margin).
- `calculate_global_energy_loss(T_pred_kelvin, batch)` — sums predicted power dissipation per graph (`G_amb * (T_pred - T_ambient)`, using `batch.G_amb_W_K` and de-normalized ambient temperature from `global_features[:, 0]`), compares to `batch.physics_node_power_W` summed per graph, and returns the mean squared normalized residual over graphs flagged `batch.pinn_global_energy_eligible`. Returns a zero loss (with correct dtype/device) if no graph in the batch is eligible.
- `normalized_to_kelvin(prediction_normalized)` — `exp(prediction_normalized * target_temp_scale + target_temp_mean)`, no Jensen correction (used inside the training loop's real-space loss, where the correction is not applied — the correction is a validation/inference-time reporting adjustment, not part of the optimized objective).
- `apply_jensen_correction(prediction_normalized, residual_log_var)` — as `normalized_to_kelvin` but adds `0.5 * residual_log_var` to the log-prediction before exponentiating, if `CONFIG["use_lognormal_correction"]` is true.
- `estimate_jensen_sigma_per_range(model, loader)` — one forward pass over `loader` (no grad), computes `(target - prediction) * target_temp_scale` as the log-space residual, computes its variance separately within each of the 4 temperature ranges (based on `batch.y`, i.e., ground-truth Kelvin, not prediction), returns a `[4]` tensor. Called once per epoch on the **training** loader only (never validation), so the correction is estimated without touching held-out data.
- `assert_all_nodes_are_targets(loader)` / `assert_physics_fields_exist(loader)` — sanity checks run once before training starts, raising `RuntimeError` if `T_real_normalized`/`y` don't cover every node, or if `physics_node_power_W`, `G_amb_W_K`, `pinn_global_energy_eligible` are missing from a batch.
- `compute_temperature_distribution(loader)` — prints node counts/percentages per configured range and warns if any nodes fall outside all 4 ranges.
- `calculate_basic_metrics`, `calculate_range_metrics`, `calculate_fine_temperature_diagnostics` — RMSE/MAE/bias (overall, per 4 ranges, and per 10 K fine-grained bin respectively) from raw prediction/target Kelvin tensors.
- `train_one_epoch(model, loader, ml_criterion, real_criterion, optimizer, physics_lambda)` — standard forward/backward loop; total loss = `ml_loss + CONFIG["real_space_loss_weight"] * real_loss + physics_lambda * physics_loss`; gradient-clips at `CONFIG["grad_clip"]` (35); accumulates normalized and real-space metrics and the global-energy residual for eligible graphs; returns a metrics dict including the raw prediction/target tensors for downstream range analysis.
- `validate(model, loader, per_range_var)` — no-grad pass; applies the Jensen correction per node using the **target's** range (a lookup, not a leak into gradients, since `validate()` never backprops); computes the same metric set as training, plus physics loss and global residual, for reporting only (physics loss is not part of any validation "objective" used for model selection — model selection is by `real_rmse`).
- `train_model(model, train_loader, val_loader)` — the main loop: runs sanity asserts, builds both loss objects, `AdamW` optimizer, `ReduceLROnPlateau` scheduler; per epoch: ramps `physics_lambda` through the warmup window, trains one epoch, estimates per-range Jensen variance from the **training** loader, validates, computes the train/val RMSE gap, computes per-range and per-10K-bin diagnostics for both splits, prints a detailed report, saves a checkpoint (`torch.save`) whenever `val_metrics["real_rmse"]` improves (a rich checkpoint including optimizer/scheduler state, full `CONFIG`, per-range Jensen variance, and dataset/training provenance strings), appends a JSON-serializable history record and rewrites `CONFIG["results_path"]` every epoch, steps the LR scheduler on validation RMSE, then checks two stopping conditions: the gap-based overfitting guard (`gap > 25%` after warmup + 10-epoch buffer) and ordinary early stopping (`epochs_without_improvement >= 12`).

**Checkpoint file**: `CONFIG["checkpoint_path"] = "Model_v7_playground.pt"`, results JSON at `CONFIG["results_path"] = "Model_v7_playground.json"` — both written to the current working directory the script is run from.

---

### `Testing_phase.py`

**Responsibility**: full held-out test-set evaluation of a chosen checkpoint.

**Flow**: rebuilds `ML_only_model` (this file's copy uses `nn.ReLU()`, matching `FASTAPI.py`'s copy rather than `Training_v1_playground.py`'s `LeakyReLU` — `[Needs clarification]`, see README Section 8.2/11), loads `CONFIG["checkpoint_path"]` (`Model_v7_playground.pt` per the path in this file, though the comment block above it also lists loading `ML_with_physics_model_ultra_super_power_V2000.pt` as a prior option), loads `loaders["test"]`.

**`test_full_dataset(model, test_loader, target_temp_mean, target_temp_scale, device)`**:
- Runs inference batch by batch, denormalizing with plain `exp()` (no Jensen correction applied in this script — `[Needs clarification]`: confirm whether the numbers this script reports are meant to be directly compared to Jensen-corrected validation numbers, since the two use different denormalization).
- Splits each batch's predictions/ground truth back into per-graph tensors using `batch.batch` (the PyG batch-index vector), then concatenates everything back into one flat prediction/ground-truth pair for whole-test-set metrics.
- Computes and prints: overall RMSE/MAE/bias/max-error; percentage of nodes with `|error| > 5/8/10 K`; and, per one of the 4 fixed temperature ranges (defined locally in this script, hardcoded, not imported from `Training_v1_playground.CONFIG`), node count, RMSE, MAE, bias, max error, and `%>8K`/`%>10K`.
- Returns `(results_dict, all_predictions_list, all_ground_truth_list)`, where `results_dict` mirrors the printed structure (`overall` and `per_range` sub-dicts) — suitable for saving to JSON if you extend this script to do so (it currently only prints).

---

### `FASTAPI.py`

**Responsibility**: (older-configuration) inference API.

**Model/checkpoint**: defines its own `ML_only_model` using `nn.ReLU()`; `CONFIG["checkpoint_path"] = "C:/msys64/home/dell/ML_with_physics_model_ultra_super_power_V2000.pt"`. Loads the checkpoint, builds the model, loads `state_dict`, sets `model.eval()`. Also loads `normalized_temp_scaller` from `ARTIFACT_DIR` to reconstruct `target_temp_mean`/`target_temp_scale` for de-normalization (no Jensen correction applied here either — plain `torch.exp`).

**Helper**: `save_uploaded_folder(Files: list[UploadFile]) -> str` — writes every uploaded file into a fresh temp directory (`tempfile.mkdtemp(prefix="thermal_")`), returns the directory path.

**Endpoints**:
- `GET /plot/{plot_name}` — returns `FileResponse(os.path.join(PLOT_DIR, plot_name))`. No existence check before serving — a request for a non-existent plot will surface FastAPI/Starlette's default file-not-found behavior rather than a custom error.
- `POST /predict` — `Files: list[UploadFile] = File(...)`. Flow: `save_uploaded_folder` -> `build_one_graph(input_dir)` (from `Stack_floorplan_to_graph.py`, imported here from an `Inference_mode` package) -> `Train_Nodes_data_pipleline` -> `Train_Edges_data_pipleline` -> `Train_Global_data_pipleline_new` (all imported from a `Preprocessing` package — i.e., the single-graph versions in `process_one_graph.py`, despite the `Train_` prefix in their names) -> builds a `data` dict expanding `global_features` to one row per node (`graph.global_features.expand(graph.x.size(0), -1)`, since the model's forward pass expects one global-feature row per node, matching how `batch.global_features[batch.batch]` is constructed during training) -> `model(...)` under `torch.no_grad()` -> `torch.exp(output * target_temp_scale + target_temp_mean)` -> generates a unique plot filename (`prediction_{uuid4().hex}.png`), calls `plot_graph(graph, result)`, saves the current matplotlib figure to `PLOT_DIR`, closes it -> returns JSON with `status`, `num_nodes`, `min/max/mean_temperature_K`, the full `temperatures_K` list, and `plot_name`.

**No error handling** around graph building, model inference, or plotting in either endpoint — an invalid upload (missing `.flp`, malformed `.stk`, an H5 validation failure inside `build_one_graph`) will surface as an unhandled exception (`500` by default) rather than a descriptive API error.

---

### `Final_app.py`

**Responsibility**: (newer-configuration) model/checkpoint loader, imported elsewhere as `Backend.Final_app` (per `Test_inference_mode.py`'s `from Backend.Final_app import model, device`) — i.e., this file is intended to be the shared model-loading module for other scripts, not a standalone FastAPI app despite its name and its `print("Final app in action...")` banner.

**Differences from `FASTAPI.py`'s copy**:
- Uses `nn.LeakyReLU(negative_slope=0.01)` throughout (matching `Training_v1_playground.py`).
- `CONFIG["checkpoint_path"] = "C:/msys64/home/dell/Model_v7_playground.pt"` (the latest playground checkpoint), with a comment block listing the full lineage of prior checkpoints considered.
- Also loads `loaders = load(filename="loaders")` at module level, unused in the active (non-commented) code path.
- Defines `PROJECT_ROOT` and appends it to `sys.path`, and imports `plot_graph` from a `Plotting_results` package — implying this file expects to run as part of a proper package structure (`Plotting_results/`, `Preprocessing/`, `Backend/`), unlike `FASTAPI.py`, which imports from `Inference_mode`/`Preprocessing` packages with slightly different names.
- The bulk of the file (everything after loading the model and target-temp scaler) is a **triple-quoted, fully commented-out block**: a one-off manual test that grabs a single preprocessed batch from `loaders["test"]`, runs inference, prints per-node prediction vs. ground truth with error, and calls `plot_graph` + `plt.show()`. This is dead code kept as a reference/history of a manual sanity check, not part of the active module.

**Does not itself define any FastAPI routes** — as uploaded, this file only prepares `model` and `device` as importable module-level objects.

---

### `Stack_floorplan_to_graph.py`

**Responsibility**: single entry point, `build_one_graph(input_dir, sim_name=None, contact_tolerance=0.001, max_recovery_gap=None, converter_path=..., validator_path=..., graph_builder_path=..., reference_builder_path=...)`, that turns a folder of `.stk`/`.flp` files into one ready `torch_geometric.data.Data` graph.

**How it avoids reimplementing logic**: rather than importing the three dependency scripts normally, it dynamically loads each one from an explicit file path via `importlib.util.spec_from_file_location` / `module_from_spec` / `exec_module` (helper `_load_module`), registering each under a synthetic module name (`sv_converter`, `sv_validator`, `sv_graph_builder`) in `sys.modules`. This lets the wrapper pin exact file paths (defaulting to files that sit next to this script) without depending on package/import-path setup elsewhere.

**Flow inside `build_one_graph`**:
1. Resolves `input_dir` to an absolute path; generates a random `sim_name` if none given.
2. Loads the three dependency modules.
3. Inside a `tempfile.TemporaryDirectory()`: calls `converter.build_v3_h5(str(input_dir), str(h5_path), sim_name)` to produce an H5 file.
4. Opens the H5 file, calls `validator.validate_sim(sim)`; if it returns any errors, raises `ValueError` immediately (graph construction never proceeds on an invalid H5).
5. Loads the reference contract via `graph_builder._load_reference(reference_path)` (pointing, by default, at `build_graph_PINN_ready_final_cohort_gated.py` — the same script used to build the **training** data), then calls `graph_builder.build_graph(sim, h5f, ref, contact_tolerance=..., max_recovery_gap=...)`, followed by `graph_builder.validate_inference_graph(graph, ref)` — a second, graph-level validation against the reference contract.
6. Returns the resulting `Data` graph (documented shape: `x [N,13]`, `edge_index [2,E]`, `edge_attr [E,11]`, `global_features [1,27]` — i.e., the **raw**, pre-preprocessing feature counts, matching the columns consumed by `process_one_graph.py`).

**CLI**: `python stack_floorplan_to_graph.py <input_dir> [--output graph.pt] [--sim-name ...] [--reference-builder ...]` — prints node/edge counts and shapes, optionally saves the graph via `torch.save`.

---

### `Extract_ground_truth_temperatures.py`

**Responsibility**: parses 3D-ICE's raw `tflp_*.txt` output files into ground-truth per-block temperatures, for validation/testing only (a genuinely new design has no such files, since the whole point of the surrogate model is to avoid running the simulator).

**Functions**:
- `parse_tflp_file(tflp_path) -> Dict[str, float]` — reads a single `tflp_*.txt` file; finds the header line (starts with `%` and contains `Time`) to recover block names (each column name has a trailing `(K)` stripped); takes the **last** data row (documented as "average, final" steady-state; if the file is transient with multiple rows, the last is treated as closest to steady state); returns `{block_name: temperature_K}`. Raises `ValueError` on a missing header, no data rows, or a column/value count mismatch.
- `build_ground_truth(input_dir, pattern="tflp_*.txt") -> Dict[str, float]` — globs all matching files in a directory, parses each, and merges into one dict; raises `ValueError` if the same block name appears in more than one file (would indicate a data problem, since each block should be reported by exactly one layer's file).
- `build_ground_truth_in_graph_order(input_dir) -> list[float]` — reconstructs the ground truth in the **exact node order the graph builder uses**: a full pass over all layers collecting every *real* block first (in `stk_data["layers_ordered"]` / `parse_flp_file` order), then a **second, separate** full pass over all layers collecting every *filler* block. The module docstring explicitly warns that interleaving real+filler per layer (instead of two full passes) silently produces a wrong node-to-temperature alignment — this function exists specifically to prevent that class of bug when comparing model output to ground truth.
- `main()` — CLI: `python extract_ground_truth_temperatures.py <input_dir> [--output out.json]`, prints a sample of loaded temperatures, optionally dumps to JSON.

---

### `Test_inference_mode.py`

**Responsibility**: a manual, single-simulation, end-to-end validation script (not a reusable function/module — a linear script meant to be run and inspected).

**Flow**: imports `model, device` from `Backend.Final_app`; hardcodes one `input_dir` pointing at a specific prior simulation batch on the developer's local machine; calls `build_one_graph(input_dir)` and `build_ground_truth_in_graph_order(input_dir)`; prints graph shape info and the raw ground-truth list; loads the temperature scaler; runs the three inference-time preprocessing functions (`Train_Nodes_data_pipleline`, `Train_Edges_data_pipleline`, `Train_Global_data_pipleline_new` — again, the single-graph versions despite the `Train_` name) on the graph; builds the same `data` dict shape as `FASTAPI.py`'s `/predict`; runs the model under `torch.no_grad()`; denormalizes with plain `exp()` (no Jensen correction); overwrites `graph.y` with the real ground-truth tensor (so `plot_graph` can show both prediction and truth); calls `plot_graph(graph, result)` and `plt.show()`.

---

### `plot_predictionsVSGroundTruth.py`

**Responsibility**: `plot_graph(graph, result, cmap="turbo", fontsize=7.5, node_size=2200, figsize=(16,7), error_ring_K=10, color_by="truth", show_cross_layer=True, save=None) -> (fig, ax)`.

**Internals**:
- `_arr(t)` — converts a tensor (detaching/moving to CPU first if it has `.detach`) or array-like into a plain `numpy.ndarray`, or returns `None` unchanged.
- Ground truth is read from `graph.y` if present, else `graph.real_y`; if its length doesn't match the number of predictions, attempts to scatter it into the right positions using `graph.target_mask` (a legacy-graph compatibility path).
- Counts and warns about exact-zero values in either prediction or ground truth (`n_zero`), noting these are excluded from the color-range calculation as likely data issues.
- Node layout: uses `graph.pos` (3D) if present, projecting z into a discretized `layer` index via `np.searchsorted` on unique rounded z-values, and horizontally offsetting each layer's nodes by a gap proportional to the x-extent so layers appear as separate columns; falls back to an evenly-spaced circular layout if `graph.pos` is absent.
- Color value selection (`color_by`): `"prediction"` uses raw predictions, `"error"` uses `prediction - truth` with a symmetric `coolwarm` scale sized to the max absolute error, and the default (anything else, effectively `"truth"`) uses ground truth where available and falls back to prediction per node. Color range is clipped to the 1st-99th percentile of finite, non-zero values (falling back to `[min-1, max+1]` if the percentile range is degenerate, or `[0,1]` if there are no valid values at all).
- Draws deduplicated edges (`seen` set on `(min(s,d), max(s,d))` pairs) — lateral edges (same layer) solid/darker, cross-layer edges faint/thin, the latter skippable via `show_cross_layer=False`.
- Draws nodes as a single `scatter` call, then overlays a second, larger, unfilled scatter with a red edge (`#e53e3e`) around any node with `|error| > error_ring_K`.
- Annotates each node with `f"{true:.1f}\n{pred:.1f}"` (or just the prediction if no truth), choosing white or black text based on the perceptual luminance of that node's fill color (so labels stay legible against both light and dark cells).
- Adds a layer-label callout above each layer's column when more than one layer is present.
- Title includes RMSE/MAE/max-error/bias computed over whatever nodes have both a finite prediction and truth, or a fallback "no ground truth" note.
- `save` parameter, if given, calls `fig.savefig(save, dpi=150, bbox_inches="tight")` before returning.

---

### `app.py` (Streamlit frontend)

**Responsibility**: upload UI and result viewer.

**Flow**: sets page config/logo from a local asset path (`D:/Thermal surrogate project/assets/si-vision-logo (2).png` — `[Needs clarification]`: this hardcoded Windows path must exist on whatever machine actually serves the Streamlit app); file uploader restricted to `.stk`/`.flp`, multiple files allowed; lists uploaded filenames; on button click, packages every uploaded file as a `(fieldname, (filename, bytes, mimetype))` tuple under the form field name `"Files"` (matching `FASTAPI.py`'s `Files: list[UploadFile] = File(...)` parameter name) and POSTs to `http://127.0.0.1:8000/predict`; on success, fetches the plot image from `http://127.0.0.1:8000/plot/{plot_name}` and displays it via `st.image`, plus the min/max/mean temperature and the full per-node temperature list; on a non-200 response, shows the status code and raw response text; wraps the whole request in a `try/except` that surfaces any connection error (e.g., backend not running) as a Streamlit error message.

## 4. End-to-End Execution Flow

**Offline (one time, or whenever new simulation data arrives)**:
1. Run 3D-ICE simulations to produce `.stk`/`.flp` inputs and `tflp_*.txt` outputs for many designs.
2. Convert each design to a graph via the training-time graph builder (`build_graph_PINN_ready_final_cohort_gated.py`, not included in this documentation pass) and assemble raw `.pt` graph-dataset files.
3. Run `Dataset.py` (directly, or via `prepare_Data.py`'s import): filters, finds the best-matched train/val/test split, fits every scaler/one-hot cardinality, persists them via `joblib`.
4. Run `prepare_Data.py`: preprocesses all three splits, builds and persists the `loaders` `DataLoader` bundle.
5. Run `Training_v1_playground.py`: trains `ML_only_model` with the physics-informed loss, saving the best checkpoint (`Model_v7_playground.pt`) and a full per-epoch diagnostics JSON.
6. Run `Testing_phase.py` against the held-out test loader for final reported metrics.

**Online (per user request)**:
1. User uploads `.stk`/`.flp` files via `app.py`.
2. `app.py` POSTs to `FASTAPI.py`'s `/predict`.
3. `Stack_floorplan_to_graph.build_one_graph()` converts the upload into one graph, reusing the exact training-time contract.
4. The single-graph preprocessing functions (`process_one_graph.py`, imported under their `Train_...` names) apply the same fitted scalers/one-hot encodings.
5. The trained model produces a per-node normalized log-temperature prediction; `exp()` (± Jensen correction, depending on which backend module is actually wired in — see Section 3's `FASTAPI.py`/`Final_app.py` notes) converts it to Kelvin.
6. `plot_graph()` renders and saves a visualization; the API returns per-node temperatures, summary stats, and the plot filename.
7. `app.py` displays the plot and stats to the user.

**Validation (developer-run, ad hoc)**: `Test_inference_mode.py` performs steps 3-5 above against a design whose real 3D-ICE ground truth is known (extracted via `Extract_ground_truth_temperatures.py`), then plots prediction vs. truth side by side for visual/numeric sanity-checking outside the formal `Testing_phase.py` test-set run.

## 5. Configuration Reference

| Parameter | Location | Meaning |
|---|---|---|
| `in_features` | Model `CONFIG` (all files) | Node feature dimension after preprocessing (18) |
| `in_features_edge` | Model `CONFIG` | Edge feature dimension after preprocessing (11) |
| `global_features` | Model `CONFIG` | Global feature dimension after preprocessing (21) |
| `hidden_features` | Model `CONFIG` | GNN hidden width (64) |
| `edge_mlp_hidden` | Model `CONFIG` | Hidden width of the NNConv edge MLP (32) |
| `num_layers` | Model `CONFIG` | Number of stacked message-passing blocks (4) |
| `gat_heads` | Model `CONFIG` | GATv2Conv attention heads (2, averaged via `concat=False`) |
| `checkpoint_path` | Model `CONFIG` | Which `.pt` file to load/save — differs between `FASTAPI.py` and `Final_app.py`/`Testing_phase.py`/`Training_v1_playground.py` (see Section 3 notes) |
| `temperature_ranges` | `Training_v1_playground.CONFIG` | The 4 fixed 50 K bins used for range-balanced losses and diagnostics |
| `boundary_margin_K` | `Training_v1_playground.CONFIG` | Width of the soft-blend zone around each internal range boundary (10.0 K) |
| `real_space_loss_weight` / `real_space_loss_scale_K` | `Training_v1_playground.CONFIG` | Weight and Huber scale for the Kelvin-space loss term (1, 30.0 K) |
| `lambda_physics` / `physics_warmup_epochs` | `Training_v1_playground.CONFIG` | Final physics-loss weight and its linear warmup length (0.05, 20 epochs) |
| `lr` / `weight_decay` / `grad_clip` | `Training_v1_playground.CONFIG` | AdamW learning rate, weight decay, and gradient-norm clip value (1e-3, 3e-4, 35) |
| `scheduler_factor` / `scheduler_patience` | `Training_v1_playground.CONFIG` | `ReduceLROnPlateau` settings (0.5, 1 epoch) |
| `early_stopping_patience` | `Training_v1_playground.CONFIG` | Epochs without val-RMSE improvement before stopping (12) |
| `max_safe_train_val_gap_pct` / `gap_guard_buffer_epochs` | `Training_v1_playground.CONFIG` | Hard overfitting-gap stop threshold and its post-warmup grace period (25%, 10 epochs) |
| `use_lognormal_correction` | Model `CONFIG` (all files) | Whether to apply the Jensen bias correction when converting log-space predictions to Kelvin |
| `diagnostic_bin_size_K` | `Training_v1_playground.CONFIG` | Width of the fine-grained diagnostic bins (10 K) |
| `ARTIFACT_DIR` | `process_one_graph.py`, `FASTAPI.py`, `Final_app.py` | Absolute path where scalers/one-hot metadata/checkpoints are loaded from at inference time (`C:/msys64/home/dell`) |
| `PLOT_DIR` | `FASTAPI.py` | Local folder where generated prediction plots are saved and served from (`"plots"`) |

## 6. API / Interface Reference

Implemented in `FASTAPI.py` (see also `Final_app.py`'s note on being a model-loading module rather than a route-defining one).

| Endpoint | Method | Input | Output | Errors |
|---|---|---|---|---|
| `/predict` | POST | Multipart form, field name `Files`, one or more `.stk`/`.flp` files | JSON: `status`, `num_nodes`, `min_temperature_K`, `max_temperature_K`, `mean_temperature_K`, `temperatures_K` (list), `plot_name` | No explicit handling — an invalid upload, H5 validation failure, or graph-construction error propagates as an unhandled exception (FastAPI default 500) |
| `/plot/{plot_name}` | GET | Plot filename (from a prior `/predict` response) | The image file (`FileResponse`) | No existence check — a bad filename surfaces FastAPI/Starlette's default not-found behavior |

## 7. Testing Strategy (as implemented)

There is no automated unit-test framework (`pytest`/`unittest`) among the uploaded files. What exists instead:
- **`Testing_phase.py`**: a full, deterministic held-out test-set evaluation (RMSE/MAE/bias/max-error, overall and per temperature range) — the closest thing to a formal test suite, but it prints results rather than asserting thresholds or writing them to a file.
- **`Test_inference_mode.py`**: a manual, one-simulation, ad hoc integration check comparing live inference output against known 3D-ICE ground truth — a developer sanity script, not an automated test.
- **Sanity assertions inside training** (`assert_all_nodes_are_targets`, `assert_physics_fields_exist` in `Training_v1_playground.py`): fail fast (via `RuntimeError`) if the data going into a training run doesn't have the shape/fields the loss functions require.

## 8. Production Readiness Notes

### Already implemented
- Physics-informed loss with a real energy-conservation constraint, not just data fitting.
- Careful, diagnosed, and documented handling of a real overfitting episode, including a training-time guard beyond ordinary early stopping.
- A dedicated wrapper (`Stack_floorplan_to_graph.py`) specifically built to keep inference-time graph construction consistent with training-time graph construction.
- Rich checkpoints (optimizer/scheduler state, full config, provenance strings) enabling reproducibility and resumption.
- A dedicated ground-truth extraction and node-ordering utility with an explicit, documented warning about a real ordering bug it exists to prevent.

### Needs improvement
- Two inference backends (`FASTAPI.py`, `Final_app.py`) disagree on model activation and checkpoint — must be reconciled before presenting a single "production" story.
- Hardcoded, machine-specific absolute paths throughout (`ARTIFACT_DIR`, dataset paths in `Dataset.py`, the logo path in `app.py`, checkpoint paths) — not portable without edits.
- No automated test suite; `Testing_phase.py` prints rather than persists or asserts against results.
- No error handling in either FastAPI endpoint.
- Duplicated preprocessing-function definitions across `Dataset.py` and `Train_data_processin.py` (identical logic, not shared via a common import).
- A real, uncommented inconsistency in `relative_depth`'s normalization (`/max(layer_idx)` vs. a hardcoded `/4`) between the training pipeline and the val/test/inference pipelines (Section 3) — worth verifying before relying on this feature's exact meaning across splits.
- `Testing_phase.py` denormalizes with plain `exp()` (no Jensen correction) while `Training_v1_playground.py`'s `validate()` applies the correction — the two may not be reporting metrics on a consistent basis; worth confirming which convention your presented numbers use.

### Recommended next steps
- Consolidate on one backend module and one checkpoint as the stated production configuration.
- Externalize all hardcoded paths into environment variables or a config file.
- Persist `Testing_phase.py`'s results to a JSON file alongside training's `results_path`, for direct before/after comparison across checkpoints.
- Add a small automated regression test that reruns `Test_inference_mode.py`'s comparison on a fixed reference simulation and asserts the RMSE stays within a known-good bound, catching future silent regressions from preprocessing or model changes.
