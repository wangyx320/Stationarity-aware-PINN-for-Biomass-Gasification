#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ============================================================
# M4 Phase A — MLP-Eq systematic logarithmic Grid + Pareto analysis
# Using ONE predefined fixed Train/Validation/Test split
#
# Final M4 loss:
#     L = L_data + lambda_eq * L_eq_const
#
# CLEAN M3 -> M4 ABLATION
# -----------------------
# 1) Reuse the completed M3 architecture / optimizer settings unchanged.
# 2) M4 changes ONLY lambda_eq.
# 3) Every lambda_eq candidate is trained for the FULL MAX_EPOCHS trajectory.
#    No early stopping is used during lambda_eq search.
# 4) For every fixed lambda_eq, choose its representative checkpoint by:
#
#        e*(lambda) = argmin_e Validation Mean NRMSE(lambda, e)
#
#    where Mean NRMSE is averaged over H2 / CO / CO2 / CH4.
# 5) Every representative model is mapped into the two-objective
#    Validation space:
#
#        Objective 1: minimize Validation Mean NRMSE
#        Objective 2: minimize Validation Mean Eeq_SS
#
#    where Mean Eeq_SS is averaged over H2 / CO / CO2.
# 6) Retain Pareto non-dominated representative models.
# 7) Robust-standardize BOTH Pareto objectives within the Pareto set:
#
#        z = (f - median(f)) / IQR(f)
#
# 8) Define the equal-weight compromise score:
#
#        S = z_NRMSE + z_Eeq
#
#    and identify the Pareto candidate with minimum S.
# 9) TCR is a supporting diagnostic only and is NOT an optimization objective.
# 10) This Phase-A cell NEVER evaluates Test.
#
# SEARCH STRATEGY
# ---------------
#     lambda_eq = 0 plus 101 positive log-spaced values from 1e-4 to 10.
#     The positive grid contains 100 equal intervals in log10 space
#     (20 intervals per decade across five decades).
#
# TWO-CELL WORKFLOW
# -----------------
# Cell 1 / Phase A:
#     - run/reuse the full predefined lambda_eq grid;
#     - choose one representative epoch per lambda_eq using Validation NRMSE;
#     - construct the Validation Pareto front;
#     - calculate robust standardized compromise scores;
#     - save the selected-compromise flag;
#     - STOP before Test.
#
# Cell 2 / Phase B:
#     - read the saved Pareto table;
#     - independently recompute/verify the same robust compromise score;
#     - fix lambda_eq and epoch;
#     - retrain on Training + Validation;
#     - evaluate Test once.
#
# M4 ablation rules:
#     - equilibrium constraint only for process_stage == "ss_stage"
#     - all ss_stage samples use constant physics weight = 1
#     - H2 / CO / CO2 receive the equilibrium constraint
#     - CH4 is data-driven only
#     - startup/off samples receive no equilibrium constraint
#     - no gradient loss
#     - no stationarity weighting
# ============================================================

from pathlib import Path
import re
import json
import random
import joblib
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# 1. Basic settings
# ============================================================

RANDOM_STATE = 42

# ------------------------------------------------------------
# Systematic logarithmic Grid + Pareto settings
# ------------------------------------------------------------

# Full trajectory for every lambda_eq candidate.
MAX_EPOCHS = 1500

# Broad positive search interval.
LAMBDA_EQ_MIN = 1e-4
LAMBDA_EQ_MAX = 10.0

# 101 positive values = 100 equal log10 intervals across 5 decades.
# This is 20 intervals per decade.
# Together with lambda_eq = 0, 102 representative candidates are evaluated.
N_COARSE_LOG_POINTS = 101

POSITIVE_COARSE_LAMBDA_GRID = np.logspace(
    np.log10(LAMBDA_EQ_MIN),
    np.log10(LAMBDA_EQ_MAX),
    N_COARSE_LOG_POINTS,
)

COARSE_LAMBDA_GRID = np.concatenate(
    [
        np.array([0.0], dtype=float),
        POSITIVE_COARSE_LAMBDA_GRID,
    ]
)

# ------------------------------------------------------------
# Pareto compromise-selection settings
# ------------------------------------------------------------

# Reuse the expensive full-grid results whenever they already exist.
REUSE_EXISTING_SEARCH_RESULTS = True

# Equal importance for prediction accuracy and equilibrium consistency
# AFTER robust standardization within the Pareto set.
PARETO_WEIGHT_NRMSE = 1.0
PARETO_WEIGHT_PHYSICS = 1.0

# Numerical matching tolerance used only for floating-point identity checks.
PARETO_LAMBDA_MATCH_RTOL = 1e-10
PARETO_LAMBDA_MATCH_ATOL = 1e-14

# GCR settings fixed by the adopted definition.
GCR_DELTA_T = 5.0
GCR_EPS_FRACTION = 0.01

# Experimental data
DATA_PATH = Path(
    "../20250606_oxygen enrich_whole process_"
    "linear interpolation.csv"
)

# Predefined fixed split
FIXED_SPLIT_PATH = Path(
    "../fixed_split_results/fixed_split_index.csv"
)

# FactSage equilibrium data
FACTSAGE_PATH = Path(
    "../Fig_FactSage_data.xlsx"
)

# Completed M3 files
BASELINE_MLP_BEST_PARAMS_PATH = Path(
    "../MLP/MLP_TPE_results/mlp_best_parameters.json"
)
BASELINE_MLP_RUN_CONFIG_PATH = Path(
    "../MLP/MLP_TPE_results/mlp_run_configuration.json"
)

# Final consolidated M4 output folder
OUTPUT_DIR = Path(
    "MLP_Eq_Pareto_results"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("PyTorch device")
print("======================================")
print(DEVICE)


# ============================================================
# 2. Reproducibility
# ============================================================


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve repeatability.
    # Exact determinism can still depend on PyTorch/CUDA versions.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(
    RANDOM_STATE
)


# ============================================================
# 3. Features and targets
# ============================================================

FEATURES = [
    "Feeding rate",
    "Oxygen concentration",
    "50mm",
    "100mm",
    "200mm",
    "300mm",
    "400mm",
    "500mm",
    "600mm",
    "700mm",
    "800mm",
    "900mm",
    "950mm",
]


TEMP_COLS = [
    "50mm",
    "100mm",
    "200mm",
    "300mm",
    "400mm",
    "500mm",
    "600mm",
    "700mm",
    "800mm",
    "900mm",
    "950mm",
]


TARGETS = [
    "H2",
    "CO",
    "CO2",
    "CH4",
]


# The adopted physics-constraint framework uses H2 / CO / CO2 only.
# CH4 is intentionally excluded from Eq / TCR / GCR, even if a CH4
# row exists in the FactSage workbook, to remain identical to M3.
FACTSAGE_TARGETS = [
    "H2",
    "CO",
    "CO2",
]


# ============================================================
# 4. Check input files
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Experimental data file not found:\n"
        f"{DATA_PATH.resolve()}"
    )


if not FIXED_SPLIT_PATH.exists():
    raise FileNotFoundError(
        "Fixed split file not found:\n"
        f"{FIXED_SPLIT_PATH.resolve()}"
    )


if not FACTSAGE_PATH.exists():
    raise FileNotFoundError(
        "FactSage file not found:\n"
        f"{FACTSAGE_PATH.resolve()}"
    )


if not BASELINE_MLP_BEST_PARAMS_PATH.exists():
    raise FileNotFoundError(
        "Baseline MLP best-parameter file not found:\n"
        f"{BASELINE_MLP_BEST_PARAMS_PATH.resolve()}\n\n"
        "Run the final M3 MLP first, or correct "
        "BASELINE_MLP_BEST_PARAMS_PATH."
    )

if not BASELINE_MLP_RUN_CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Baseline MLP run-configuration file not found:\n"
        f"{BASELINE_MLP_RUN_CONFIG_PATH.resolve()}\n\n"
        "The completed M3 run must provide mlp_run_configuration.json, "
        "or BASELINE_MLP_RUN_CONFIG_PATH must be corrected."
    )


print("\n======================================")
print("Input files")
print("======================================")
print("Experimental data:", DATA_PATH.resolve())
print("Fixed split:", FIXED_SPLIT_PATH.resolve())
print("FactSage:", FACTSAGE_PATH.resolve())
print("M3 best parameters:", BASELINE_MLP_BEST_PARAMS_PATH.resolve())
print("M3 run configuration:", BASELINE_MLP_RUN_CONFIG_PATH.resolve())
print("Output folder:", OUTPUT_DIR.resolve())


# ============================================================
# 4.1 Load fixed M3 MLP hyperparameters
# ============================================================

with open(
    BASELINE_MLP_BEST_PARAMS_PATH,
    "r",
    encoding="utf-8",
) as file:
    baseline_mlp_parameter_payload = json.load(file)


BASE_MLP_PARAMETER_KEYS = [
    "n_hidden_layers",
    "hidden_units",
    "activation",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
]


missing_baseline_parameters = [
    key
    for key in BASE_MLP_PARAMETER_KEYS
    if key not in baseline_mlp_parameter_payload
]


if missing_baseline_parameters:
    raise ValueError(
        "The M3 best-parameter file is missing: "
        f"{missing_baseline_parameters}"
    )


BASE_MLP_PARAMS = {
    key: baseline_mlp_parameter_payload[key]
    for key in BASE_MLP_PARAMETER_KEYS
}




# ------------------------------------------------------------
# Load the already-completed M3 Validation reference.
# M3 is NOT trained again in this M4 script.
# ------------------------------------------------------------
with open(
    BASELINE_MLP_RUN_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    baseline_mlp_run_config = json.load(file)

required_m3_reference_keys = [
    "best_validation_mean_nrmse",
    "best_validation_epoch",
]

missing_m3_reference_keys = [
    key
    for key in required_m3_reference_keys
    if key not in baseline_mlp_run_config
]

if missing_m3_reference_keys:
    raise ValueError(
        "The M3 run-configuration file is missing: "
        f"{missing_m3_reference_keys}"
    )

m3_reference_val_mean_nrmse = float(
    baseline_mlp_run_config["best_validation_mean_nrmse"]
)

m3_reference_best_epoch = int(
    baseline_mlp_run_config["best_validation_epoch"]
)

if not np.isfinite(m3_reference_val_mean_nrmse):
    raise ValueError(
        "M3 best_validation_mean_nrmse is not finite."
    )

if m3_reference_val_mean_nrmse <= 0:
    raise ValueError(
        "M3 best_validation_mean_nrmse must be positive."
    )

if m3_reference_best_epoch <= 0:
    raise ValueError(
        "M3 best_validation_epoch must be positive."
    )

print("\nFixed M3 MLP hyperparameters reused by M4:")
for key, value in BASE_MLP_PARAMS.items():
    print(f"{key}: {value}")

print("\n======================================")
print("Saved M3 validation reference")
print("======================================")
print("M3 best Validation Mean NRMSE:", m3_reference_val_mean_nrmse)
print("M3 best Validation epoch:", m3_reference_best_epoch)
print(
    "M4 Pareto objectives: minimize Validation Mean NRMSE "
    "and minimize Validation Mean Eeq_SS."
)


# ============================================================
# 5. Load original experimental data
# ============================================================

df = pd.read_csv(
    DATA_PATH
).copy()


# ------------------------------------------------------------
# Add original row index
# This MUST match original_row in fixed_split_index.csv
# ------------------------------------------------------------

df["original_row"] = np.arange(
    len(df)
)


required_data_columns = set(
    FEATURES
    + TARGETS
    + [
        "Time",
        "Flow rate",
    ]
)


missing_data_columns = sorted(
    required_data_columns
    - set(df.columns)
)


if missing_data_columns:
    raise ValueError(
        "Missing columns in experimental data: "
        f"{missing_data_columns}"
    )


print("\n======================================")
print("Original experimental data")
print("======================================")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 6. Load predefined fixed split
# ============================================================

split_df = pd.read_csv(
    FIXED_SPLIT_PATH
)


required_split_columns = [
    "original_row",
    "Time",
    "run_id",
    "process_stage",
    "block_uid",
    "stage_block_id",
    "dynamic_state",
    "is_local_steady",
    "is_local_transient",
    "D_thermal",
    "S_stationarity",
    "physics_weight",
    "split",
]


missing_split_columns = [
    column
    for column in required_split_columns
    if column not in split_df.columns
]


if missing_split_columns:
    raise ValueError(
        "Missing columns in fixed split file: "
        f"{missing_split_columns}"
    )


# ============================================================
# 7. Verify fixed split matches original data
# ============================================================

if len(split_df) != len(df):
    raise ValueError(
        "Fixed split row count does not match "
        "experimental data.\n"
        f"Experimental data: {len(df)}\n"
        f"Fixed split: {len(split_df)}"
    )


if split_df["original_row"].duplicated().any():
    raise ValueError(
        "Duplicate original_row values found "
        "in fixed_split_index.csv."
    )


expected_rows = np.arange(
    len(df)
)


actual_rows = (
    split_df["original_row"]
    .sort_values()
    .to_numpy(dtype=int)
)


if not np.array_equal(
    expected_rows,
    actual_rows,
):
    raise ValueError(
        "original_row in fixed split does not "
        "exactly cover the original dataset."
    )


# ------------------------------------------------------------
# Verify Time alignment before merge
# ------------------------------------------------------------

split_time_check = (
    split_df[
        [
            "original_row",
            "Time",
        ]
    ]
    .sort_values("original_row")
)


original_time = df["Time"].to_numpy(
    dtype=float
)


fixed_time = split_time_check["Time"].to_numpy(
    dtype=float
)


if not np.allclose(
    original_time,
    fixed_time,
    rtol=0.0,
    atol=1e-8,
):
    raise ValueError(
        "Time values in fixed split do not match "
        "the original experimental dataset."
    )


# ============================================================
# 8. Merge fixed split information
# ============================================================

# Do not merge duplicate Time column.
split_columns_to_merge = [
    "original_row",
    "run_id",
    "process_stage",
    "block_uid",
    "stage_block_id",
    "dynamic_state",
    "is_local_steady",
    "is_local_transient",
    "D_thermal",
    "S_stationarity",
    "physics_weight",
    "split",
]


df = (
    df.merge(
        split_df[
            split_columns_to_merge
        ],
        on="original_row",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    .sort_values("original_row")
    .reset_index(drop=True)
)


# ============================================================
# 9. Verify predefined split
# ============================================================

valid_split_names = {
    "train",
    "validation",
    "test",
}


actual_split_names = set(
    df["split"].unique()
)


if actual_split_names != valid_split_names:
    raise ValueError(
        "Unexpected split labels.\n"
        f"Actual: {actual_split_names}"
    )


train_idx = (
    df.index[
        df["split"] == "train"
    ]
    .to_numpy()
)


val_idx = (
    df.index[
        df["split"] == "validation"
    ]
    .to_numpy()
)


test_idx = (
    df.index[
        df["split"] == "test"
    ]
    .to_numpy()
)


if (
    len(train_idx)
    + len(val_idx)
    + len(test_idx)
    != len(df)
):
    raise ValueError(
        "Fixed split does not cover all samples."
    )


if (
    np.intersect1d(train_idx, val_idx).size
    or np.intersect1d(train_idx, test_idx).size
    or np.intersect1d(val_idx, test_idx).size
):
    raise ValueError(
        "Train, Validation and Test overlap."
    )


# ============================================================
# 10. Print fixed split summary
# ============================================================

split_counts = (
    df["split"]
    .value_counts()
    .reindex(
        [
            "train",
            "validation",
            "test",
        ]
    )
)


print("\n======================================")
print("Predefined fixed split")
print("======================================")
print(split_counts)


print("\nPercentages:")
print(
    (
        split_counts
        / len(df)
        * 100
    ).round(2)
)


stage_split_table = pd.crosstab(
    df["process_stage"],
    df["split"],
)


print("\nProcess stage × split:")
print(stage_split_table)


ss_all_df = df.loc[
    df["process_stage"] == "ss_stage"
]


dynamic_split_table = pd.crosstab(
    ss_all_df["dynamic_state"],
    ss_all_df["split"],
)


print("\nSS dynamic state × split:")
print(dynamic_split_table)


# ============================================================
# 11. Representative temperature
# Trep = Tmax
# ============================================================

df["Trep"] = (
    df[TEMP_COLS]
    .max(axis=1)
)


df["Tmax_sensor"] = (
    df[TEMP_COLS]
    .idxmax(axis=1)
)


# ============================================================
# 12. Prepare Training / Validation / Test data
# ============================================================

X_train = (
    df.loc[
        train_idx,
        FEATURES,
    ]
    .copy()
)


y_train = (
    df.loc[
        train_idx,
        TARGETS,
    ]
    .copy()
)


X_val = (
    df.loc[
        val_idx,
        FEATURES,
    ]
    .copy()
)


y_val = (
    df.loc[
        val_idx,
        TARGETS,
    ]
    .copy()
)


X_test = (
    df.loc[
        test_idx,
        FEATURES,
    ]
    .copy()
)


y_test = (
    df.loc[
        test_idx,
        TARGETS,
    ]
    .copy()
)


print("\n======================================")
print("Model data shapes")
print("======================================")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 13. Training target ranges
# NRMSE = RMSE / training target range
# ============================================================

target_range = (
    y_train.max(axis=0)
    - y_train.min(axis=0)
)


if (target_range <= 0).any():
    raise ValueError(
        "At least one target has zero or negative "
        "training range."
    )


print("\nTraining target ranges:")
print(target_range)


# ============================================================
# 14. Read FactSage equilibrium data (same parser as M3)
# ============================================================

factsage_raw_df = pd.read_excel(
    FACTSAGE_PATH,
    sheet_name=0,
    header=None,
)



def normalize_factsage_gas_name(
    value,
):
    gas_name = (
        str(value)
        .strip()
        .replace("₂", "2")
        .replace("₄", "4")
        .replace(" ", "")
        .upper()
    )

    mapping = {
        "H2": "H2",
        "CO": "CO",
        "CO2": "CO2",
    }

    return mapping.get(
        gas_name
    )


factsage_records = []
row_index = 0


while row_index < len(
    factsage_raw_df
):
    first_cell = factsage_raw_df.iat[
        row_index,
        0,
    ]

    oxygen_match = re.match(
        r"^\s*(\d+(?:\.\d+)?)"
        r"\s*%\s*O2\s*$",
        str(first_cell),
        flags=re.IGNORECASE,
    )

    if oxygen_match is None:
        row_index += 1
        continue

    oxygen_value = float(
        oxygen_match.group(1)
    )

    temperature_series = pd.to_numeric(
        factsage_raw_df.iloc[
            row_index,
            1:,
        ],
        errors="coerce",
    )

    valid_temperature = temperature_series.notna()

    factsage_temperatures = (
        temperature_series.loc[
            valid_temperature
        ]
        .to_numpy(dtype=float)
    )

    if len(factsage_temperatures) < 2:
        raise ValueError(
            f"FactSage O2={oxygen_value}% "
            "has fewer than two temperatures."
        )

    gas_curves = {}
    gas_row = row_index + 1

    while gas_row < len(
        factsage_raw_df
    ):
        gas_label = factsage_raw_df.iat[
            gas_row,
            0,
        ]

        if pd.isna(gas_label):
            break

        next_oxygen_match = re.match(
            r"^\s*(\d+(?:\.\d+)?)"
            r"\s*%\s*O2\s*$",
            str(gas_label),
            flags=re.IGNORECASE,
        )

        if next_oxygen_match is not None:
            break

        gas_name = normalize_factsage_gas_name(
            gas_label
        )

        if gas_name is not None:
            gas_values = (
                pd.to_numeric(
                    factsage_raw_df.iloc[
                        gas_row,
                        1:,
                    ],
                    errors="coerce",
                )
                .loc[valid_temperature]
                .to_numpy(dtype=float)
            )

            if len(gas_values) != len(
                factsage_temperatures
            ):
                raise ValueError(
                    f"FactSage {gas_name} length "
                    f"mismatch at O2={oxygen_value}%."
                )

            if not np.isfinite(
                gas_values
            ).all():
                raise ValueError(
                    f"FactSage {gas_name} has missing "
                    f"values at O2={oxygen_value}%."
                )

            gas_curves[
                gas_name
            ] = gas_values

        gas_row += 1

    missing_gases = [
        target
        for target in FACTSAGE_TARGETS
        if target not in gas_curves
    ]

    if missing_gases:
        raise ValueError(
            f"FactSage O2={oxygen_value}% "
            f"is missing gases: {missing_gases}"
        )

    if not np.all(
        np.diff(
            factsage_temperatures
        ) > 0
    ):
        raise ValueError(
            "FactSage temperatures must increase "
            f"for O2={oxygen_value}%."
        )

    for position, temperature in enumerate(
        factsage_temperatures
    ):
        record = {
            "Oxygen concentration": oxygen_value,
            "Temperature": float(temperature),
        }

        for target in FACTSAGE_TARGETS:
            record[
                f"{target}_eq"
            ] = float(
                gas_curves[target][position]
            )

        factsage_records.append(
            record
        )

    row_index = gas_row


# ============================================================
# 15. Verify FactSage table
# ============================================================

factsage_long_df = pd.DataFrame(
    factsage_records
)


if factsage_long_df.empty:
    raise ValueError(
        "No FactSage data could be parsed."
    )


if factsage_long_df.duplicated(
    subset=[
        "Oxygen concentration",
        "Temperature",
    ]
).any():
    raise ValueError(
        "FactSage contains duplicate "
        "O2-temperature combinations."
    )


available_oxygen = set(
    factsage_long_df[
        "Oxygen concentration"
    ].unique()
)


required_oxygen = {
    21.0,
    23.0,
    25.0,
}


if not required_oxygen.issubset(
    available_oxygen
):
    raise ValueError(
        "FactSage must contain "
        "21%, 23% and 25% O2."
    )


factsage_long_df = (
    factsage_long_df
    .sort_values(
        [
            "Oxygen concentration",
            "Temperature",
        ]
    )
    .reset_index(drop=True)
)


print("\nFactSage summary:")
print(
    factsage_long_df.groupby(
        "Oxygen concentration"
    )["Temperature"].agg(
        [
            "count",
            "min",
            "max",
        ]
    )
)


# ============================================================
# 16. FactSage interpolation
# ============================================================


def interpolate_factsage_values(
    oxygen_values,
    temperature_values,
    target,
):
    if target not in FACTSAGE_TARGETS:
        raise ValueError(
            f"{target} is not a FactSage target."
        )

    oxygen_values = np.asarray(
        oxygen_values,
        dtype=float,
    )

    temperature_values = np.asarray(
        temperature_values,
        dtype=float,
    )

    if len(oxygen_values) != len(
        temperature_values
    ):
        raise ValueError(
            "oxygen_values and temperature_values "
            "must have the same length."
        )

    interpolated_values = np.full(
        len(temperature_values),
        np.nan,
        dtype=float,
    )

    for oxygen_value in np.unique(
        oxygen_values
    ):
        sample_mask = np.isclose(
            oxygen_values,
            oxygen_value,
        )

        curve = (
            factsage_long_df.loc[
                np.isclose(
                    factsage_long_df[
                        "Oxygen concentration"
                    ],
                    oxygen_value,
                )
            ]
            .sort_values("Temperature")
        )

        if curve.empty:
            raise ValueError(
                "No FactSage curve available for "
                f"O2={oxygen_value}%."
            )

        grid_temperature = curve[
            "Temperature"
        ].to_numpy(dtype=float)

        grid_values = curve[
            f"{target}_eq"
        ].to_numpy(dtype=float)

        requested_temperature = temperature_values[
            sample_mask
        ]

        outside_range = (
            (
                requested_temperature
                < grid_temperature.min()
            )
            |
            (
                requested_temperature
                > grid_temperature.max()
            )
        )

        if outside_range.any():
            invalid_temperature = requested_temperature[
                outside_range
            ]

            raise ValueError(
                f"Trep for O2={oxygen_value}% "
                "is outside FactSage range: "
                f"{invalid_temperature.tolist()}"
            )

        interpolated_values[
            sample_mask
        ] = np.interp(
            requested_temperature,
            grid_temperature,
            grid_values,
        )

    if not np.isfinite(
        interpolated_values
    ).all():
        raise ValueError(
            "FactSage interpolation failed "
            f"for {target}."
        )

    return interpolated_values



# ============================================================
# 17. Build constant-equilibrium targets for M4 training
# ============================================================

# IMPORTANT:
#     L_eq_const is applied only to ss_stage samples.
#     Within ss_stage every sample gets the SAME weight = 1.
#     We DO NOT use physics_weight, D_thermal, S_stationarity,
#     is_local_steady, or is_local_transient in the M4 loss.
#
# This means local-steady and local-transient samples inside ss_stage
# are treated equally by the equilibrium constraint. The later M6 model
# will introduce stationarity weighting.

EQ_TARGET_POSITIONS = [
    TARGETS.index(target)
    for target in FACTSAGE_TARGETS
]


def build_equilibrium_targets_for_rows(
    metadata_frame,
):
    """
    Return:
        equilibrium_physical: shape (n_samples, 3), ordered H2/CO/CO2
        equilibrium_mask:     True only for process_stage == 'ss_stage'

    Startup/off rows contain NaN equilibrium targets and are excluded
    from L_eq_const by equilibrium_mask.
    """

    metadata_frame = metadata_frame.copy().reset_index(drop=True)

    equilibrium_mask = (
        metadata_frame["process_stage"]
        .eq("ss_stage")
        .to_numpy(dtype=bool)
    )

    equilibrium_physical = np.full(
        (
            len(metadata_frame),
            len(FACTSAGE_TARGETS),
        ),
        np.nan,
        dtype=float,
    )

    if equilibrium_mask.any():
        oxygen_ss = (
            metadata_frame.loc[
                equilibrium_mask,
                "Oxygen concentration",
            ]
            .to_numpy(dtype=float)
        )

        temperature_ss = (
            metadata_frame.loc[
                equilibrium_mask,
                "Trep",
            ]
            .to_numpy(dtype=float)
        )

        for k, target in enumerate(
            FACTSAGE_TARGETS
        ):
            equilibrium_physical[
                equilibrium_mask,
                k,
            ] = interpolate_factsage_values(
                oxygen_values=oxygen_ss,
                temperature_values=temperature_ss,
                target=target,
            )

    return (
        equilibrium_physical,
        equilibrium_mask,
    )



def scale_equilibrium_targets(
    equilibrium_physical,
    equilibrium_mask,
    y_scaler,
):
    """
    Scale H2/CO/CO2 equilibrium targets using the SAME target scaler
    used by the neural-network outputs.

    Non-physics rows are filled with 0 only to keep TensorDataset finite;
    they are excluded from L_eq_const by equilibrium_mask.
    """

    equilibrium_physical = np.asarray(
        equilibrium_physical,
        dtype=float,
    )

    equilibrium_mask = np.asarray(
        equilibrium_mask,
        dtype=bool,
    )

    equilibrium_scaled = np.zeros_like(
        equilibrium_physical,
        dtype=float,
    )

    if equilibrium_mask.any():
        target_means = y_scaler.mean_[
            EQ_TARGET_POSITIONS
        ]

        target_scales = y_scaler.scale_[
            EQ_TARGET_POSITIONS
        ]

        equilibrium_scaled[
            equilibrium_mask
        ] = (
            equilibrium_physical[
                equilibrium_mask
            ]
            - target_means
        ) / target_scales

    return equilibrium_scaled


train_metadata_for_physics = (
    df.loc[train_idx]
    .copy()
    .reset_index(drop=True)
)


(
    eq_train_physical,
    eq_train_mask,
) = build_equilibrium_targets_for_rows(
    train_metadata_for_physics
)


print("\n======================================")
print("M4 constant-equilibrium training subset")
print("======================================")
print("Training samples:", len(train_metadata_for_physics))
print("SS-stage physics samples:", int(eq_train_mask.sum()))
print(
    "Startup/off data-only samples:",
    int((~eq_train_mask).sum()),
)


# ------------------------------------------------------------
# Build Validation equilibrium targets ONCE.
# These are used only for model selection/diagnostics, never for
# gradient updates. Test remains untouched.
# ------------------------------------------------------------

val_metadata_for_physics = (
    df.loc[val_idx]
    .copy()
    .reset_index(drop=True)
)

(
    eq_val_physical,
    eq_val_mask,
) = build_equilibrium_targets_for_rows(
    val_metadata_for_physics
)

if int(eq_val_mask.sum()) == 0:
    raise ValueError(
        "No ss_stage samples are available in Validation; "
        "physics-aware M4 selection cannot be performed."
    )

print("Validation SS-stage physics samples:", int(eq_val_mask.sum()))


# ============================================================
# 18. Standardization for MLP-Eq lambda optimization
#
# IMPORTANT:
#     During grid-search lambda optimization, scalers are fitted on
#     TRAIN only. Validation and Test do not contribute to them.
# ============================================================

x_scaler_search = StandardScaler()
y_scaler_search = StandardScaler()


X_train_scaled = x_scaler_search.fit_transform(
    X_train[FEATURES].to_numpy(dtype=float)
)


X_val_scaled = x_scaler_search.transform(
    X_val[FEATURES].to_numpy(dtype=float)
)


y_train_scaled = y_scaler_search.fit_transform(
    y_train[TARGETS].to_numpy(dtype=float)
)


y_val_scaled = y_scaler_search.transform(
    y_val[TARGETS].to_numpy(dtype=float)
)


eq_train_scaled = scale_equilibrium_targets(
    equilibrium_physical=eq_train_physical,
    equilibrium_mask=eq_train_mask,
    y_scaler=y_scaler_search,
)


# ============================================================
# 19. MLP definition (unchanged from M3)
# ============================================================


def get_activation(name):
    if name == "relu":
        return nn.ReLU()

    if name == "tanh":
        return nn.Tanh()

    if name == "silu":
        return nn.SiLU()

    raise ValueError(
        f"Unsupported activation: {name}"
    )


class MultiOutputMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units,
        n_hidden_layers,
        activation,
        dropout,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(n_hidden_layers):
            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_units,
                )
            )

            layers.append(
                get_activation(
                    activation
                )
            )

            if dropout > 0:
                layers.append(
                    nn.Dropout(
                        p=dropout
                    )
                )

            current_dim = hidden_units

        layers.append(
            nn.Linear(
                current_dim,
                output_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 20. MLP-Eq helper functions
# ============================================================


def build_mlp_from_params(params):
    model = MultiOutputMLP(
        input_dim=len(FEATURES),
        output_dim=len(TARGETS),
        hidden_units=int(params["hidden_units"]),
        n_hidden_layers=int(params["n_hidden_layers"]),
        activation=params["activation"],
        dropout=float(params["dropout"]),
    )

    return model.to(DEVICE)



def make_train_loader(
    X_scaled,
    y_scaled,
    eq_scaled,
    eq_mask,
    batch_size,
    seed,
):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    eq_tensor = torch.tensor(eq_scaled, dtype=torch.float32)
    eq_mask_tensor = torch.tensor(eq_mask, dtype=torch.bool)

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
        eq_tensor,
        eq_mask_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )




def predict_scaled(model, X_scaled):
    model.eval()

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction = model(X_tensor).cpu().numpy()

    return prediction



def predict_physical(
    model,
    X_physical,
    x_scaler,
    y_scaler,
):
    if isinstance(X_physical, pd.DataFrame):
        X_array = X_physical[FEATURES].to_numpy(dtype=float)
    else:
        X_array = np.asarray(X_physical, dtype=float)

    X_scaled = x_scaler.transform(X_array)
    prediction_scaled = predict_scaled(model, X_scaled)
    return y_scaler.inverse_transform(prediction_scaled)



def calculate_mean_nrmse(
    y_true_physical,
    y_pred_physical,
):
    nrmse_values = []

    for j, target in enumerate(TARGETS):
        rmse = np.sqrt(
            mean_squared_error(
                y_true_physical[:, j],
                y_pred_physical[:, j],
            )
        )

        nrmse_values.append(
            rmse / float(target_range[target])
        )

    return float(np.mean(nrmse_values))



def calculate_validation_equilibrium_metrics(
    y_true_physical,
    y_pred_physical,
    equilibrium_physical,
    equilibrium_mask,
):
    """
    Validation physics metrics used for M4 model selection.

    Eeq_SS_j = RMSE(model, FactSage) / training target range
    on ALL Validation ss_stage samples.

    TCR_j = fraction of Validation ss_stage samples satisfying
        |prediction - equilibrium| <= |measurement - equilibrium|.

    M4 selection uses Mean_Eeq_SS as the PRIMARY physics metric.
    Mean_TCR is saved as a supporting diagnostic only.
    """

    y_true_physical = np.asarray(y_true_physical, dtype=float)
    y_pred_physical = np.asarray(y_pred_physical, dtype=float)
    equilibrium_physical = np.asarray(equilibrium_physical, dtype=float)
    equilibrium_mask = np.asarray(equilibrium_mask, dtype=bool)

    if not equilibrium_mask.any():
        raise ValueError("Validation equilibrium mask contains no ss_stage samples.")

    metrics = {}
    eeq_values = []
    tcr_values = []

    for k, target in enumerate(FACTSAGE_TARGETS):
        target_position = TARGETS.index(target)

        measured_ss = y_true_physical[
            equilibrium_mask,
            target_position,
        ]

        predicted_ss = y_pred_physical[
            equilibrium_mask,
            target_position,
        ]

        equilibrium_ss = equilibrium_physical[
            equilibrium_mask,
            k,
        ]

        rmse_model_vs_eq = np.sqrt(
            np.mean((predicted_ss - equilibrium_ss) ** 2)
        )

        eeq_ss = float(
            rmse_model_vs_eq / float(target_range[target])
        )

        experimental_equilibrium_error = np.abs(
            measured_ss - equilibrium_ss
        )

        model_equilibrium_error = np.abs(
            predicted_ss - equilibrium_ss
        )

        tcr = float(
            np.mean(
                model_equilibrium_error
                <= experimental_equilibrium_error
            )
        )

        metrics[f"{target}_Eeq_SS"] = eeq_ss
        metrics[f"{target}_TCR"] = tcr

        eeq_values.append(eeq_ss)
        tcr_values.append(tcr)

    metrics["Mean_Eeq_SS"] = float(np.mean(eeq_values))
    metrics["Mean_TCR"] = float(np.mean(tcr_values))

    return metrics






def train_fixed_epochs(
    params,
    X_scaled,
    y_scaled,
    eq_scaled,
    eq_mask,
    n_epochs,
    seed=RANDOM_STATE,
):
    """
    Final M4 training on TRAIN + VALIDATION for exactly n_epochs.
    No Test information is used.
    """

    set_global_seed(seed)

    model = build_mlp_from_params(params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    criterion = nn.MSELoss()
    lambda_eq = float(params["lambda_eq"])

    train_loader = make_train_loader(
        X_scaled,
        y_scaled,
        eq_scaled,
        eq_mask,
        batch_size=int(params["batch_size"]),
        seed=seed,
    )

    history_records = []

    for epoch in range(1, int(n_epochs) + 1):
        model.train()

        epoch_data_loss_sum = 0.0
        epoch_data_sample_count = 0
        epoch_eq_loss_sum = 0.0
        epoch_eq_sample_count = 0

        for (
            X_batch,
            y_batch,
            eq_batch,
            eq_mask_batch,
        ) in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            eq_batch = eq_batch.to(DEVICE)
            eq_mask_batch = eq_mask_batch.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(X_batch)
            loss_data = criterion(prediction, y_batch)

            n_eq_batch = int(eq_mask_batch.sum().item())

            if n_eq_batch > 0:
                prediction_eq_targets = prediction[
                    eq_mask_batch
                ][:, EQ_TARGET_POSITIONS]

                equilibrium_batch = eq_batch[eq_mask_batch]

                loss_eq = criterion(
                    prediction_eq_targets,
                    equilibrium_batch,
                )
            else:
                loss_eq = torch.zeros(
                    (),
                    dtype=prediction.dtype,
                    device=DEVICE,
                )

            loss_total = loss_data + lambda_eq * loss_eq
            loss_total.backward()
            optimizer.step()

            batch_n = len(X_batch)
            epoch_data_loss_sum += float(loss_data.item()) * batch_n
            epoch_data_sample_count += batch_n

            if n_eq_batch > 0:
                epoch_eq_loss_sum += float(loss_eq.item()) * n_eq_batch
                epoch_eq_sample_count += n_eq_batch

        train_data_loss = (
            epoch_data_loss_sum / epoch_data_sample_count
        )

        if epoch_eq_sample_count > 0:
            train_eq_loss = (
                epoch_eq_loss_sum / epoch_eq_sample_count
            )
        else:
            train_eq_loss = np.nan

        weighted_eq_contribution = (
            lambda_eq * train_eq_loss
            if np.isfinite(train_eq_loss)
            else np.nan
        )

        train_total_loss = (
            train_data_loss + weighted_eq_contribution
            if np.isfinite(weighted_eq_contribution)
            else train_data_loss
        )

        history_records.append(
            {
                "Epoch": epoch,
                "Train_Data_MSE_scaled": train_data_loss,
                "Train_Eq_MSE_scaled": train_eq_loss,
                "Lambda_Eq": lambda_eq,
                "Weighted_Eq_Contribution": weighted_eq_contribution,
                "Train_Total_Loss": train_total_loss,
            }
        )

    return model, pd.DataFrame(history_records)




# ============================================================
# 21. Fixed-lambda full-trajectory search helper
# ============================================================


def calculate_target_nrmse_dict(
    y_true_physical,
    y_pred_physical,
):
    """Per-target Validation NRMSE for reporting."""

    y_true_physical = np.asarray(
        y_true_physical,
        dtype=float,
    )
    y_pred_physical = np.asarray(
        y_pred_physical,
        dtype=float,
    )

    result = {}

    for j, target in enumerate(TARGETS):
        rmse = np.sqrt(
            mean_squared_error(
                y_true_physical[:, j],
                y_pred_physical[:, j],
            )
        )
        result[f"{target}_Val_NRMSE"] = float(
            rmse / float(target_range[target])
        )

    return result



def train_fixed_lambda_full_trajectory(
    lambda_eq,
    seed=RANDOM_STATE,
):
    """
    Train one fixed lambda_eq for exactly MAX_EPOCHS on TRAIN only.

    No tolerance-based early stopping is used.
    Validation is evaluated after every epoch.

    The representative checkpoint for this lambda is chosen later by:
        minimum Validation Mean NRMSE.
    """

    set_global_seed(seed)

    params_local = dict(BASE_MLP_PARAMS)
    params_local["lambda_eq"] = float(lambda_eq)

    model = build_mlp_from_params(params_local)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params_local["learning_rate"]),
        weight_decay=float(params_local["weight_decay"]),
    )

    criterion = nn.MSELoss()

    train_loader = make_train_loader(
        X_train_scaled,
        y_train_scaled,
        eq_train_scaled,
        eq_train_mask,
        batch_size=int(params_local["batch_size"]),
        seed=seed,
    )

    history_records = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        epoch_data_loss_sum = 0.0
        epoch_data_sample_count = 0
        epoch_eq_loss_sum = 0.0
        epoch_eq_sample_count = 0

        for (
            X_batch,
            y_batch,
            eq_batch,
            eq_mask_batch,
        ) in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            eq_batch = eq_batch.to(DEVICE)
            eq_mask_batch = eq_mask_batch.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(X_batch)
            loss_data = criterion(
                prediction,
                y_batch,
            )

            n_eq_batch = int(
                eq_mask_batch.sum().item()
            )

            if n_eq_batch > 0:
                prediction_eq = prediction[
                    eq_mask_batch
                ][:, EQ_TARGET_POSITIONS]

                equilibrium_eq = eq_batch[
                    eq_mask_batch
                ]

                loss_eq = criterion(
                    prediction_eq,
                    equilibrium_eq,
                )
            else:
                loss_eq = torch.zeros(
                    (),
                    dtype=prediction.dtype,
                    device=DEVICE,
                )

            loss_total = (
                loss_data
                + float(lambda_eq) * loss_eq
            )

            loss_total.backward()
            optimizer.step()

            batch_n = len(X_batch)

            epoch_data_loss_sum += (
                float(loss_data.item()) * batch_n
            )
            epoch_data_sample_count += batch_n

            if n_eq_batch > 0:
                epoch_eq_loss_sum += (
                    float(loss_eq.item()) * n_eq_batch
                )
                epoch_eq_sample_count += n_eq_batch

        train_data_loss = (
            epoch_data_loss_sum
            / epoch_data_sample_count
        )

        if epoch_eq_sample_count > 0:
            train_eq_loss = (
                epoch_eq_loss_sum
                / epoch_eq_sample_count
            )
        else:
            train_eq_loss = np.nan

        weighted_eq_contribution = (
            float(lambda_eq) * train_eq_loss
            if np.isfinite(train_eq_loss)
            else np.nan
        )

        train_total_loss = (
            train_data_loss + weighted_eq_contribution
            if np.isfinite(weighted_eq_contribution)
            else train_data_loss
        )

        # Validation prediction at this epoch.
        val_prediction_scaled = predict_scaled(
            model,
            X_val_scaled,
        )

        val_prediction_physical = (
            y_scaler_search.inverse_transform(
                val_prediction_scaled
            )
        )

        val_mean_nrmse = calculate_mean_nrmse(
            y_true_physical=(
                y_val[TARGETS].to_numpy(dtype=float)
            ),
            y_pred_physical=val_prediction_physical,
        )

        physics_metrics = (
            calculate_validation_equilibrium_metrics(
                y_true_physical=(
                    y_val[TARGETS].to_numpy(dtype=float)
                ),
                y_pred_physical=val_prediction_physical,
                equilibrium_physical=eq_val_physical,
                equilibrium_mask=eq_val_mask,
            )
        )

        target_nrmse = calculate_target_nrmse_dict(
            y_true_physical=(
                y_val[TARGETS].to_numpy(dtype=float)
            ),
            y_pred_physical=val_prediction_physical,
        )

        record = {
            "Lambda_Eq": float(lambda_eq),
            "Epoch": int(epoch),
            "Train_Data_MSE_scaled": float(train_data_loss),
            "Train_Eq_MSE_scaled": (
                float(train_eq_loss)
                if np.isfinite(train_eq_loss)
                else np.nan
            ),
            "Weighted_Eq_Contribution": (
                float(weighted_eq_contribution)
                if np.isfinite(weighted_eq_contribution)
                else np.nan
            ),
            "Train_Total_Loss": float(train_total_loss),
            "Val_Mean_NRMSE": float(val_mean_nrmse),
            "Val_NRMSE_Change_vs_M3_percent": float(
                (
                    val_mean_nrmse
                    / m3_reference_val_mean_nrmse
                    - 1.0
                )
                * 100.0
            ),
            "Val_Mean_Eeq_SS": float(
                physics_metrics["Mean_Eeq_SS"]
            ),
            "Val_Mean_TCR": float(
                physics_metrics["Mean_TCR"]
            ),
        }

        record.update(target_nrmse)

        for target in FACTSAGE_TARGETS:
            record[f"Val_{target}_Eeq_SS"] = float(
                physics_metrics[f"{target}_Eeq_SS"]
            )
            record[f"Val_{target}_TCR"] = float(
                physics_metrics[f"{target}_TCR"]
            )

        history_records.append(record)

    history_df = pd.DataFrame(history_records)

    del model
    del optimizer
    del train_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history_df



def representative_from_history(history_df):
    """
    One fixed lambda_eq -> one representative checkpoint.

    Primary epoch rule:
        minimum Validation Mean NRMSE.

    Eeq is used only as a deterministic tie-breaker.
    """

    return (
        history_df
        .sort_values(
            [
                "Val_Mean_NRMSE",
                "Val_Mean_Eeq_SS",
            ],
            ascending=[True, True],
        )
        .iloc[0]
        .copy()
    )


# ============================================================
# 22. Grid + Pareto search
# ============================================================

GRID_HISTORY_DIR = (
    OUTPUT_DIR
    / "grid_lambda_histories"
)

GRID_HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPRESENTATIVES_PATH = (
    OUTPUT_DIR
    / "mlp_eq_grid_all_representatives.csv"
)

PARETO_PATH = (
    OUTPUT_DIR
    / "mlp_eq_pareto_non_dominated.csv"
)

PARETO_PLOT_PATH = (
    OUTPUT_DIR
    / "mlp_eq_pareto_non_dominated.png"
)


def lambda_history_filename(lambda_eq):
    """
    Stable file name for one fixed-lambda full trajectory.
    """

    if np.isclose(
        float(lambda_eq),
        0.0,
    ):
        label = "0"
    else:
        label = (
            f"{float(lambda_eq):.12g}"
            .replace("+", "")
            .replace("-", "m")
            .replace(".", "p")
        )

    return (
        f"lambda_{label}_history.csv"
    )


def evaluate_lambda_once(
    lambda_eq,
    cache,
):
    """
    Train/evaluate one lambda_eq exactly once.

    Each lambda_eq is represented ONLY by its prediction-best epoch:
        e*(lambda) = argmin Validation Mean NRMSE.
    """

    lambda_eq = float(
        lambda_eq
    )

    cache_key = round(
        lambda_eq,
        14,
    )

    if cache_key in cache:
        return cache[
            cache_key
        ]

    print(
        f"\n[Grid] lambda_eq = "
        f"{lambda_eq:.12g}"
    )

    history_df = (
        train_fixed_lambda_full_trajectory(
            lambda_eq=lambda_eq,
            seed=RANDOM_STATE,
        )
    )

    representative = (
        representative_from_history(
            history_df
        )
    )

    val_mean_nrmse = float(
        representative[
            "Val_Mean_NRMSE"
        ]
    )

    val_mean_eeq = float(
        representative[
            "Val_Mean_Eeq_SS"
        ]
    )

    val_mean_tcr = float(
        representative[
            "Val_Mean_TCR"
        ]
    )

    record = {
        "Lambda_Eq": (
            lambda_eq
        ),
        "Epoch": int(
            representative[
                "Epoch"
            ]
        ),
        "Val_Mean_NRMSE": (
            val_mean_nrmse
        ),
        "Val_Mean_Eeq_SS": (
            val_mean_eeq
        ),
        "Val_Mean_TCR": (
            val_mean_tcr
        ),
        "NRMSE_Change_vs_M3_percent": float(
            (
                val_mean_nrmse
                / m3_reference_val_mean_nrmse
                - 1.0
            )
            * 100.0
        ),
    }

    for target in TARGETS:
        record[
            f"{target}_Val_NRMSE"
        ] = float(
            representative[
                f"{target}_Val_NRMSE"
            ]
        )

    for target in FACTSAGE_TARGETS:

        record[
            f"{target}_Val_Eeq_SS"
        ] = float(
            representative[
                f"Val_{target}_Eeq_SS"
            ]
        )

        record[
            f"{target}_Val_TCR"
        ] = float(
            representative[
                f"Val_{target}_TCR"
            ]
        )

    history_path = (
        GRID_HISTORY_DIR
        / lambda_history_filename(
            lambda_eq
        )
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    record[
        "History_File"
    ] = str(
        history_path
    )

    print(
        "  representative epoch =",
        record[
            "Epoch"
        ],
    )

    print(
        "  Val Mean NRMSE =",
        f"{val_mean_nrmse:.10f}",
    )

    print(
        "  Val Mean Eeq_SS =",
        f"{val_mean_eeq:.10f}",
    )

    print(
        "  Val Mean TCR =",
        f"{val_mean_tcr:.6f}",
    )

    cache[
        cache_key
    ] = record

    del history_df
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


def cache_from_existing_representatives(
    representatives_df,
):
    """
    Rebuild the in-memory cache from a previously completed search.
    """

    required_columns = {
        "Lambda_Eq",
        "Epoch",
        "Val_Mean_NRMSE",
        "Val_Mean_Eeq_SS",
        "Val_Mean_TCR",
        "History_File",
    }

    missing_columns = (
        required_columns
        - set(
            representatives_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Existing representative file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    cache = {}

    for _, row in representatives_df.iterrows():

        record = row.to_dict()

        lambda_eq = float(
            record[
                "Lambda_Eq"
            ]
        )

        cache[
            round(
                lambda_eq,
                14,
            )
        ] = record

    return cache


def records_to_dataframe(
    cache,
):
    return (
        pd.DataFrame(
            list(
                cache.values()
            )
        )
        .sort_values(
            "Lambda_Eq"
        )
        .reset_index(
            drop=True
        )
    )


def pareto_non_dominated_mask(
    dataframe,
    objective_columns,
):
    """
    Pareto non-dominated mask for minimization objectives.

    Candidate j dominates candidate i when:
        all objectives_j <= objectives_i
    and:
        at least one objective_j < objectives_i
    """

    objective_values = (
        dataframe[
            objective_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        objective_values
    ).all():
        raise ValueError(
            "Pareto objectives contain NaN or infinite values."
        )

    n_rows = len(
        objective_values
    )

    non_dominated = np.ones(
        n_rows,
        dtype=bool,
    )

    for i in range(
        n_rows
    ):

        for j in range(
            n_rows
        ):

            if i == j:
                continue

            no_worse = np.all(
                objective_values[j]
                <= objective_values[i]
            )

            strictly_better = np.any(
                objective_values[j]
                < objective_values[i]
            )

            if (
                no_worse
                and strictly_better
            ):
                non_dominated[i] = False
                break

    return non_dominated


def verify_expected_grid_coverage(
    representatives_df,
):
    """
    Ensure every predefined lambda_eq in COARSE_LAMBDA_GRID exists.
    """

    evaluated = (
        representatives_df[
            "Lambda_Eq"
        ]
        .to_numpy(
            dtype=float
        )
    )

    missing = []

    for expected_lambda in COARSE_LAMBDA_GRID:

        match = np.isclose(
            evaluated,
            float(
                expected_lambda
            ),
            rtol=PARETO_LAMBDA_MATCH_RTOL,
            atol=PARETO_LAMBDA_MATCH_ATOL,
        )

        if not match.any():
            missing.append(
                float(
                    expected_lambda
                )
            )

    return missing


# ------------------------------------------------------------
# 22.1 Run or reuse the full logarithmic grid
# ------------------------------------------------------------

print("\n======================================")
print("M4 GRID + PARETO SEARCH")
print("======================================")

print(
    "Positive lambda_eq range:",
    (
        LAMBDA_EQ_MIN,
        LAMBDA_EQ_MAX,
    ),
)

print(
    "Positive grid points:",
    N_COARSE_LOG_POINTS,
)

print(
    "Total candidates including lambda_eq=0:",
    len(
        COARSE_LAMBDA_GRID
    ),
)

print(
    "Epochs per lambda_eq:",
    MAX_EPOCHS,
)

print(
    "Representative epoch rule: "
    "minimum Validation Mean NRMSE"
)

print(
    "Pareto Objective 1: "
    "minimize Validation Mean NRMSE"
)

print(
    "Pareto Objective 2: "
    "minimize Validation Mean Eeq_SS"
)

print(
    "Test used during search: False"
)


lambda_cache = {}


if (
    REUSE_EXISTING_SEARCH_RESULTS
    and REPRESENTATIVES_PATH.exists()
):

    print(
        "\nExisting representative results found."
    )

    print(
        "Loading:",
        REPRESENTATIVES_PATH.resolve(),
    )

    existing_representatives_df = pd.read_csv(
        REPRESENTATIVES_PATH
    )

    lambda_cache = (
        cache_from_existing_representatives(
            existing_representatives_df
        )
    )

    existing_coverage_missing = (
        verify_expected_grid_coverage(
            existing_representatives_df
        )
    )

    if existing_coverage_missing:
        print(
            "Existing search is incomplete; "
            "missing candidates will be evaluated."
        )
        print(
            existing_coverage_missing
        )
    else:
        print(
            "Existing search covers the complete "
            "predefined lambda_eq grid."
        )


for lambda_eq in COARSE_LAMBDA_GRID:

    evaluate_lambda_once(
        lambda_eq=lambda_eq,
        cache=lambda_cache,
    )


lambda_representatives_df = (
    records_to_dataframe(
        lambda_cache
    )
)


missing_after_search = (
    verify_expected_grid_coverage(
        lambda_representatives_df
    )
)

if missing_after_search:
    raise RuntimeError(
        "Full grid search is incomplete after evaluation. "
        f"Missing lambda_eq values: {missing_after_search}"
    )


# ------------------------------------------------------------
# 22.2 M3-equivalent lambda_eq = 0 confirmation
# ------------------------------------------------------------

lambda_zero_matches = np.isclose(
    lambda_representatives_df[
        "Lambda_Eq"
    ].to_numpy(
        dtype=float
    ),
    0.0,
    rtol=0.0,
    atol=PARETO_LAMBDA_MATCH_ATOL,
)

if int(
    lambda_zero_matches.sum()
) != 1:
    raise RuntimeError(
        "Expected exactly one lambda_eq = 0 representative."
    )


m3_confirmation_row = (
    lambda_representatives_df.loc[
        lambda_zero_matches
    ]
    .iloc[0]
    .copy()
)


m3_confirmation_epoch = int(
    m3_confirmation_row[
        "Epoch"
    ]
)

m3_confirmation_val_nrmse = float(
    m3_confirmation_row[
        "Val_Mean_NRMSE"
    ]
)

m3_like_eeq_reference = float(
    m3_confirmation_row[
        "Val_Mean_Eeq_SS"
    ]
)


m3_confirmation_history_source = Path(
    m3_confirmation_row[
        "History_File"
    ]
)

if m3_confirmation_history_source.exists():

    pd.read_csv(
        m3_confirmation_history_source
    ).to_csv(
        OUTPUT_DIR
        / "mlp_eq_m3_full_trajectory_confirmation.csv",
        index=False,
    )


print("\n======================================")
print("M3-EQUIVALENT CONFIRMATION")
print("======================================")

print(
    "Saved M3 best epoch:",
    m3_reference_best_epoch,
)

print(
    "Saved M3 Val Mean NRMSE:",
    m3_reference_val_mean_nrmse,
)

print(
    "Full-trajectory lambda_eq=0 best epoch:",
    m3_confirmation_epoch,
)

print(
    "Full-trajectory lambda_eq=0 Val Mean NRMSE:",
    m3_confirmation_val_nrmse,
)


m3_confirmation_difference = abs(
    m3_confirmation_val_nrmse
    - m3_reference_val_mean_nrmse
)

if m3_confirmation_difference > 1e-8:

    print(
        "\nWARNING:"
        "\nThe full-trajectory lambda_eq=0 result differs from "
        "the saved M3 Validation reference by more than 1e-8."
    )

else:

    print(
        "\nM3-equivalent confirmation passed."
    )


# ------------------------------------------------------------
# 22.3 Pareto non-dominated representatives
# ------------------------------------------------------------

lambda_representatives_df[
    "Eeq_Improvement_vs_Lambda0_percent"
] = (
    (
        1.0
        - lambda_representatives_df[
            "Val_Mean_Eeq_SS"
        ]
        / m3_like_eeq_reference
    )
    * 100.0
)


pareto_mask = (
    pareto_non_dominated_mask(
        dataframe=(
            lambda_representatives_df
        ),
        objective_columns=[
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
    )
)


lambda_representatives_df[
    "Is_Pareto"
] = pareto_mask


pareto_df = (
    lambda_representatives_df.loc[
        pareto_mask
    ]
    .copy()
    .sort_values(
        [
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 22.3.1 Robust standardized Pareto compromise score
# ------------------------------------------------------------
#
# Both objectives are minimized:
#
#     f1 = Validation Mean NRMSE
#     f2 = Validation Mean Eeq_SS
#
# Robust scaling is computed ONLY within the Pareto set:
#
#     z = (f - median(f)) / IQR(f)
#
# Equal-weight compromise score:
#
#     S = z_NRMSE + z_Eeq
#
# Lower S = better prediction-physics compromise.
# ------------------------------------------------------------

pareto_nrmse_median = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].median()
)

pareto_nrmse_q1 = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].quantile(0.25)
)

pareto_nrmse_q3 = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].quantile(0.75)
)

pareto_nrmse_iqr = float(
    pareto_nrmse_q3
    - pareto_nrmse_q1
)


pareto_eeq_median = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].median()
)

pareto_eeq_q1 = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.25)
)

pareto_eeq_q3 = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.75)
)

pareto_eeq_iqr = float(
    pareto_eeq_q3
    - pareto_eeq_q1
)


if pareto_nrmse_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean NRMSE has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


if pareto_eeq_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean Eeq_SS has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


pareto_df[
    "Robust_Z_NRMSE"
] = (
    pareto_df[
        "Val_Mean_NRMSE"
    ]
    - pareto_nrmse_median
) / pareto_nrmse_iqr


pareto_df[
    "Robust_Z_Eeq_SS"
] = (
    pareto_df[
        "Val_Mean_Eeq_SS"
    ]
    - pareto_eeq_median
) / pareto_eeq_iqr


pareto_df[
    "Pareto_Compromise_Score"
] = (
    PARETO_WEIGHT_NRMSE
    * pareto_df[
        "Robust_Z_NRMSE"
    ]
    +
    PARETO_WEIGHT_PHYSICS
    * pareto_df[
        "Robust_Z_Eeq_SS"
    ]
)


selected_compromise_row = (
    pareto_df
    .sort_values(
        [
            "Pareto_Compromise_Score",
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )
    .iloc[0]
    .copy()
)


selected_compromise_lambda_eq = float(
    selected_compromise_row[
        "Lambda_Eq"
    ]
)


pareto_df[
    "Is_Selected_Compromise"
] = np.isclose(
    pareto_df[
        "Lambda_Eq"
    ].to_numpy(dtype=float),
    selected_compromise_lambda_eq,
    rtol=PARETO_LAMBDA_MATCH_RTOL,
    atol=PARETO_LAMBDA_MATCH_ATOL,
)


lambda_representatives_df.to_csv(
    REPRESENTATIVES_PATH,
    index=False,
)


pareto_df.to_csv(
    PARETO_PATH,
    index=False,
)


print("\n======================================")
print("M4 ROBUST PARETO COMPROMISE SELECTION")
print("======================================")

print(
    "Selection rule: minimum equal-weight robust standardized "
    "compromise score among Validation Pareto solutions"
)

print(
    "\nPareto NRMSE median:",
    pareto_nrmse_median,
)

print(
    "Pareto NRMSE IQR:",
    pareto_nrmse_iqr,
)

print(
    "Pareto Eeq_SS median:",
    pareto_eeq_median,
)

print(
    "Pareto Eeq_SS IQR:",
    pareto_eeq_iqr,
)

print(
    "\nSelected lambda_eq:",
    selected_compromise_lambda_eq,
)

print(
    "Selected epoch:",
    int(
        selected_compromise_row[
            "Epoch"
        ]
    ),
)

print(
    "Selected Validation Mean NRMSE:",
    float(
        selected_compromise_row[
            "Val_Mean_NRMSE"
        ]
    ),
)

print(
    "Selected Validation Mean Eeq_SS:",
    float(
        selected_compromise_row[
            "Val_Mean_Eeq_SS"
        ]
    ),
)

print(
    "Selected compromise score:",
    float(
        selected_compromise_row[
            "Pareto_Compromise_Score"
        ]
    ),
)


# ------------------------------------------------------------
# 22.4 Pareto-front figure
# ------------------------------------------------------------

plt.figure(
    figsize=(
        7.2,
        5.6,
    )
)


dominated_df = (
    lambda_representatives_df.loc[
        ~lambda_representatives_df[
            "Is_Pareto"
        ]
    ]
)

if not dominated_df.empty:

    plt.scatter(
        dominated_df[
            "Val_Mean_NRMSE"
        ],
        dominated_df[
            "Val_Mean_Eeq_SS"
        ],
        s=28,
        alpha=0.35,
        label="Dominated representatives",
    )


plt.plot(
    pareto_df[
        "Val_Mean_NRMSE"
    ],
    pareto_df[
        "Val_Mean_Eeq_SS"
    ],
    marker="o",
    linewidth=1.5,
    markersize=5,
    label="Pareto non-dominated front",
)


lambda_zero_pareto = pareto_df.loc[
    np.isclose(
        pareto_df[
            "Lambda_Eq"
        ],
        0.0,
        rtol=0.0,
        atol=PARETO_LAMBDA_MATCH_ATOL,
    )
]

if not lambda_zero_pareto.empty:

    plt.scatter(
        lambda_zero_pareto[
            "Val_Mean_NRMSE"
        ],
        lambda_zero_pareto[
            "Val_Mean_Eeq_SS"
        ],
        marker="s",
        s=75,
        label="M3-equivalent anchor (lambda=0)",
    )


plt.scatter(
    [
        float(
            selected_compromise_row[
                "Val_Mean_NRMSE"
            ]
        )
    ],
    [
        float(
            selected_compromise_row[
                "Val_Mean_Eeq_SS"
            ]
        )
    ],
    marker="*",
    s=180,
    label="Selected robust compromise",
    zorder=6,
)


for _, row in pareto_df.iterrows():

    plt.annotate(
        f"{float(row['Lambda_Eq']):.4g}",
        (
            float(
                row[
                    "Val_Mean_NRMSE"
                ]
            ),
            float(
                row[
                    "Val_Mean_Eeq_SS"
                ]
            ),
        ),
        xytext=(
            5,
            5,
        ),
        textcoords="offset points",
        fontsize=8,
    )


plt.xlabel(
    "Validation Mean NRMSE"
)

plt.ylabel(
    r"Validation Mean $E_{eq,SS}$"
)

plt.title(
    "M4 Validation Pareto Front"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.4,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PARETO_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# 22.5 Print Pareto table with robust compromise score
# ------------------------------------------------------------

PARETO_DISPLAY_COLUMNS = [
    "Lambda_Eq",
    "Epoch",
    "Val_Mean_NRMSE",
    "NRMSE_Change_vs_M3_percent",
    "Val_Mean_Eeq_SS",
    "Eeq_Improvement_vs_Lambda0_percent",
    "Robust_Z_NRMSE",
    "Robust_Z_Eeq_SS",
    "Pareto_Compromise_Score",
    "Is_Selected_Compromise",
    "Val_Mean_TCR",
]


print("\n======================================")
print("M4 PARETO NON-DOMINATED SOLUTIONS")
print("======================================")

print(
    pareto_df[
        PARETO_DISPLAY_COLUMNS
    ].to_string(
        index=False
    )
)


print("\nPareto figure:")
print(
    PARETO_PLOT_PATH.resolve()
)

print("\nPareto table:")
print(
    PARETO_PATH.resolve()
)


# ============================================================
# 23. Phase-A stop gate
# ============================================================

print("\n======================================")
print("M4 PHASE A COMPLETED")
print("======================================")

print(
    "\nGrid search / Pareto analysis is complete."
)

print(
    "The automatically identified robust-compromise lambda_eq is:"
)

print(
    selected_compromise_lambda_eq
)

print(
    "\nIMPORTANT: Test has NOT been evaluated by this Cell 1."
)

print(
    "Run Cell 2 to independently recompute/verify the same "
    "robust Pareto score, finalize M4, retrain on Training + "
    "Validation, and evaluate Test once."
)

print(
    "\nSaved representative table:"
)

print(
    REPRESENTATIVES_PATH.resolve()
)

print(
    "\nSaved Pareto table:"
)

print(
    PARETO_PATH.resolve()
)

print(
    "\nSaved Pareto figure:"
)

print(
    PARETO_PLOT_PATH.resolve()
)


# In[6]:


# ============================================================
# M4 Phase B — automatic finalization from the saved Pareto front
#
# Run this CELL ONLY AFTER Cell 1 / M4 Phase A has completed.
#
# Final selection rule:
#     1) use Pareto non-dominated Validation representatives only;
#     2) robust-standardize both objectives using median and IQR
#        calculated within the Pareto set;
#     3) use equal weights:
#
#            Score = z_NRMSE + z_Eeq
#
#     4) select the solution with minimum Score.
#
# Test is evaluated only AFTER lambda_eq and epoch are fixed.
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ------------------------------------------------------------
# 0. Make sure Phase A has already been run in this notebook
# ------------------------------------------------------------

required_names = [
    "OUTPUT_DIR",
    "BASE_MLP_PARAMS",
    "PARETO_PATH",
    "PARETO_PLOT_PATH",
    "GRID_HISTORY_DIR",
    "m3_reference_val_mean_nrmse",
    "m3_reference_best_epoch",
    "m3_confirmation_epoch",
    "m3_confirmation_val_nrmse",
    "train_fixed_epochs",
    "predict_physical",
    "calculate_mean_nrmse",
    "calculate_validation_equilibrium_metrics",
    "X_train_scaled",
    "y_train_scaled",
    "eq_train_scaled",
    "eq_train_mask",
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
    "train_idx",
    "val_idx",
    "test_idx",
    "df",
    "FEATURES",
    "TEMP_COLS",
    "TARGETS",
    "FACTSAGE_TARGETS",
    "EQ_TARGET_POSITIONS",
    "target_range",
    "x_scaler_search",
    "y_scaler_search",
    "eq_val_physical",
    "eq_val_mask",
    "interpolate_factsage_values",
    "factsage_long_df",
    "build_equilibrium_targets_for_rows",
    "scale_equilibrium_targets",
    "RANDOM_STATE",
    "DEVICE",
    "GCR_DELTA_T",
    "GCR_EPS_FRACTION",
    "split_counts",
    "BASELINE_MLP_BEST_PARAMS_PATH",
    "BASELINE_MLP_RUN_CONFIG_PATH",
    "MAX_EPOCHS",
    "LAMBDA_EQ_MIN",
    "LAMBDA_EQ_MAX",
    "N_COARSE_LOG_POINTS",
    "COARSE_LAMBDA_GRID",
]

missing_names = [
    name
    for name in required_names
    if name not in globals()
]

if missing_names:
    raise RuntimeError(
        "The M4 Pareto-search cell has not been fully run in the current "
        "kernel, or required variables are missing:\n"
        f"{missing_names}\n\n"
        "Run the first M4 cell with PARETO_SELECTED_LAMBDA_EQ = None first."
    )


# ------------------------------------------------------------
# 1. Read the saved Pareto non-dominated solutions
# ------------------------------------------------------------

pareto_selection_path = Path(
    PARETO_PATH
)

if not pareto_selection_path.exists():
    raise FileNotFoundError(
        "Pareto CSV not found:\n"
        f"{pareto_selection_path.resolve()}\n\n"
        "Run the first M4 Pareto-search cell first."
    )

pareto_selection_df = pd.read_csv(
    pareto_selection_path
).copy()

required_pareto_columns = [
    "Lambda_Eq",
    "Epoch",
    "Val_Mean_NRMSE",
    "Val_Mean_Eeq_SS",
    "Val_Mean_TCR",
    "NRMSE_Change_vs_M3_percent",
    "Eeq_Improvement_vs_Lambda0_percent",
    "History_File",
]

missing_pareto_columns = [
    column
    for column in required_pareto_columns
    if column not in pareto_selection_df.columns
]

if missing_pareto_columns:
    raise ValueError(
        "Pareto CSV is missing required columns: "
        f"{missing_pareto_columns}"
    )

if pareto_selection_df.empty:
    raise ValueError(
        "Pareto CSV contains no non-dominated solutions."
    )

objective_values = pareto_selection_df[
    [
        "Lambda_Eq",
        "Val_Mean_NRMSE",
        "Val_Mean_Eeq_SS",
    ]
].to_numpy(dtype=float)

if not np.isfinite(objective_values).all():
    raise ValueError(
        "Pareto CSV contains NaN or infinite values in the key columns."
    )


# ------------------------------------------------------------
# 2. Automatically select the robust Pareto compromise solution
# ------------------------------------------------------------

PARETO_WEIGHT_NRMSE = 1.0
PARETO_WEIGHT_PHYSICS = 1.0


pareto_nrmse_median = float(
    pareto_selection_df[
        "Val_Mean_NRMSE"
    ].median()
)

pareto_nrmse_q1 = float(
    pareto_selection_df[
        "Val_Mean_NRMSE"
    ].quantile(0.25)
)

pareto_nrmse_q3 = float(
    pareto_selection_df[
        "Val_Mean_NRMSE"
    ].quantile(0.75)
)

pareto_nrmse_iqr = float(
    pareto_nrmse_q3
    - pareto_nrmse_q1
)


pareto_eeq_median = float(
    pareto_selection_df[
        "Val_Mean_Eeq_SS"
    ].median()
)

pareto_eeq_q1 = float(
    pareto_selection_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.25)
)

pareto_eeq_q3 = float(
    pareto_selection_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.75)
)

pareto_eeq_iqr = float(
    pareto_eeq_q3
    - pareto_eeq_q1
)


if pareto_nrmse_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean NRMSE has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


if pareto_eeq_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean Eeq_SS has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


pareto_selection_df[
    "Robust_Z_NRMSE"
] = (
    pareto_selection_df[
        "Val_Mean_NRMSE"
    ]
    - pareto_nrmse_median
) / pareto_nrmse_iqr


pareto_selection_df[
    "Robust_Z_Eeq_SS"
] = (
    pareto_selection_df[
        "Val_Mean_Eeq_SS"
    ]
    - pareto_eeq_median
) / pareto_eeq_iqr


pareto_selection_df[
    "Pareto_Compromise_Score"
] = (
    PARETO_WEIGHT_NRMSE
    * pareto_selection_df[
        "Robust_Z_NRMSE"
    ]
    +
    PARETO_WEIGHT_PHYSICS
    * pareto_selection_df[
        "Robust_Z_Eeq_SS"
    ]
)


selected_row = (
    pareto_selection_df
    .sort_values(
        [
            "Pareto_Compromise_Score",
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )
    .iloc[0]
    .copy()
)


selected_lambda_for_flag = float(
    selected_row[
        "Lambda_Eq"
    ]
)

pareto_selection_df[
    "Is_Selected_Compromise"
] = np.isclose(
    pareto_selection_df[
        "Lambda_Eq"
    ].to_numpy(dtype=float),
    selected_lambda_for_flag,
    rtol=1e-10,
    atol=1e-14,
)


# Persist the updated Pareto table so metadata and Cell 3 see
# exactly the same scores and selected flag.
pareto_selection_df.to_csv(
    pareto_selection_path,
    index=False,
    encoding="utf-8-sig",
)


print("\n======================================")
print("M4 AUTOMATIC ROBUST PARETO SELECTION")
print("======================================")

print(
    "Selection rule: minimum equal-weight robust standardized "
    "compromise score among Pareto non-dominated Validation solutions"
)

print(
    "\nPareto NRMSE median:",
    pareto_nrmse_median,
)

print(
    "Pareto NRMSE IQR:",
    pareto_nrmse_iqr,
)

print(
    "Pareto Eeq_SS median:",
    pareto_eeq_median,
)

print(
    "Pareto Eeq_SS IQR:",
    pareto_eeq_iqr,
)

print(
    "\nSelected exact lambda_eq from CSV:",
    float(
        selected_row[
            "Lambda_Eq"
        ]
    ),
)

print(
    "Selected epoch:",
    int(
        selected_row[
            "Epoch"
        ]
    ),
)

print(
    "Selected Validation Mean NRMSE:",
    float(
        selected_row[
            "Val_Mean_NRMSE"
        ]
    ),
)

print(
    "Selected Validation Mean Eeq_SS:",
    float(
        selected_row[
            "Val_Mean_Eeq_SS"
        ]
    ),
)

print(
    "Selected Validation Mean TCR:",
    float(
        selected_row[
            "Val_Mean_TCR"
        ]
    ),
)

print(
    "Selected robust compromise score:",
    float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    ),
)


best_lambda_eq = float(
    selected_row[
        "Lambda_Eq"
    ]
)

best_epoch_from_search = int(
    selected_row[
        "Epoch"
    ]
)

best_val_mean_nrmse_from_search = float(
    selected_row[
        "Val_Mean_NRMSE"
    ]
)

best_val_mean_eeq_ss_from_search = float(
    selected_row[
        "Val_Mean_Eeq_SS"
    ]
)

best_val_mean_tcr_from_search = float(
    selected_row[
        "Val_Mean_TCR"
    ]
)


best_params = dict(
    BASE_MLP_PARAMS
)

best_params[
    "lambda_eq"
] = best_lambda_eq


print("\n======================================")
print("FINAL M4 PARETO SELECTION")
print("======================================")

print(
    "Selected lambda_eq:",
    best_lambda_eq,
)

print(
    "Selected epoch:",
    best_epoch_from_search,
)

print(
    "Selected Val Mean NRMSE:",
    best_val_mean_nrmse_from_search,
)

print(
    "NRMSE change vs M3 (%):",
    float(
        selected_row[
            "NRMSE_Change_vs_M3_percent"
        ]
    ),
)

print(
    "Selected Val Mean Eeq_SS:",
    best_val_mean_eeq_ss_from_search,
)

print(
    "Eeq improvement vs lambda=0 (%):",
    float(
        selected_row[
            "Eeq_Improvement_vs_Lambda0_percent"
        ]
    ),
)

print(
    "Selected Val Mean TCR:",
    best_val_mean_tcr_from_search,
)

print(
    "Selected Pareto compromise score:",
    float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    ),
)


# ------------------------------------------------------------
# 23.1 Save automatically selected Pareto parameter payload
# ------------------------------------------------------------

best_parameter_payload = dict(
    best_params
)

best_parameter_payload.update(
    {
        "search_method": (
            "systematic logarithmic grid search "
            "with Validation Pareto analysis"
        ),
        "selection_rule": (
            "for_each_lambda_select_min_validation_mean_NRMSE_epoch; "
            "construct_validation_Pareto_front_using_mean_NRMSE_and_mean_Eeq_SS; "
            "robust_standardize_Pareto_objectives_using_median_and_IQR; "
            "equal_weight_sum_of_robust_standardized_objectives; "
            "select_minimum_compromise_score"
        ),
        "pareto_objective_1": (
            "minimize_validation_mean_NRMSE"
        ),
        "pareto_objective_2": (
            "minimize_validation_mean_Eeq_SS"
        ),
        "manual_pareto_selection": False,
        "automatic_pareto_selection": True,
        "pareto_final_selection_criterion": (
            "minimum_equal_weight_robust_standardized_compromise_score"
        ),
        "pareto_robust_scaling": (
            "median_IQR_within_Pareto_non_dominated_set"
        ),
        "pareto_weight_nrmse": (
            PARETO_WEIGHT_NRMSE
        ),
        "pareto_weight_physics": (
            PARETO_WEIGHT_PHYSICS
        ),
        "pareto_nrmse_median": (
            pareto_nrmse_median
        ),
        "pareto_nrmse_iqr": (
            pareto_nrmse_iqr
        ),
        "pareto_physics_median": (
            pareto_eeq_median
        ),
        "pareto_physics_iqr": (
            pareto_eeq_iqr
        ),
        "selected_pareto_compromise_score": float(
            selected_row[
                "Pareto_Compromise_Score"
            ]
        ),
        "selected_lambda_eq": (
            best_lambda_eq
        ),
        "selected_epoch": (
            best_epoch_from_search
        ),
        "selected_validation_mean_nrmse": (
            best_val_mean_nrmse_from_search
        ),
        "selected_validation_mean_Eeq_SS": (
            best_val_mean_eeq_ss_from_search
        ),
        "selected_validation_mean_TCR": (
            best_val_mean_tcr_from_search
        ),
        "m3_reference_validation_mean_nrmse": (
            m3_reference_val_mean_nrmse
        ),
        "m3_reference_best_epoch": (
            m3_reference_best_epoch
        ),
        "m3_full_trajectory_best_epoch": (
            m3_confirmation_epoch
        ),
        "m3_full_trajectory_validation_mean_nrmse": (
            m3_confirmation_val_nrmse
        ),
        "lambda_eq_search_min": (
            LAMBDA_EQ_MIN
        ),
        "lambda_eq_search_max": (
            LAMBDA_EQ_MAX
        ),
        "n_positive_log_grid_points": (
            N_COARSE_LOG_POINTS
        ),
        "lambda_grid": (
            COARSE_LAMBDA_GRID.tolist()
        ),
        "pareto_table_path": str(
            PARETO_PATH
        ),
        "pareto_plot_path": str(
            PARETO_PLOT_PATH
        ),
        "uses_test_for_selection": False,
        "uses_accuracy_tolerance": False,
        "uses_early_stopping_during_lambda_search": False,
    }
)


with open(
    OUTPUT_DIR
    / "mlp_eq_best_parameters.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        best_parameter_payload,
        file,
        indent=4,
    )


selected_history_path = Path(
    selected_row[
        "History_File"
    ]
)

if not selected_history_path.exists():
    raise FileNotFoundError(
        "Selected lambda history file not found:\n"
        f"{selected_history_path}"
    )


selected_history_df = pd.read_csv(
    selected_history_path
)

selected_history_df.to_csv(
    OUTPUT_DIR
    / "mlp_eq_best_validation_history.csv",
    index=False,
)


model_selection_summary_df = pd.DataFrame(
    [
        {
            "Selection_Method": (
                "Validation Pareto robust standardized compromise selection"
            ),
            "Selected_Lambda_Eq": (
                best_lambda_eq
            ),
            "Selected_Epoch": (
                best_epoch_from_search
            ),
            "Selected_Val_Mean_NRMSE": (
                best_val_mean_nrmse_from_search
            ),
            "NRMSE_Change_vs_M3_percent": float(
                selected_row[
                    "NRMSE_Change_vs_M3_percent"
                ]
            ),
            "Selected_Val_Mean_Eeq_SS": (
                best_val_mean_eeq_ss_from_search
            ),
            "Selected_Pareto_Compromise_Score": float(
                selected_row[
                    "Pareto_Compromise_Score"
                ]
            ),
            "Eeq_Improvement_vs_Lambda0_percent": float(
                selected_row[
                    "Eeq_Improvement_vs_Lambda0_percent"
                ]
            ),
            "Selected_Val_Mean_TCR": (
                best_val_mean_tcr_from_search
            ),
            "M3_Reference_Val_Mean_NRMSE": (
                m3_reference_val_mean_nrmse
            ),
        }
    ]
)


model_selection_summary_df.to_csv(
    OUTPUT_DIR
    / "mlp_eq_model_selection_summary.csv",
    index=False,
)


# ============================================================
# 24. Rebuild robust-compromise Pareto M4 on TRAIN only
# ============================================================

best_mlp_validation, _ = train_fixed_epochs(
    params=best_params,
    X_scaled=X_train_scaled,
    y_scaled=y_train_scaled,
    eq_scaled=eq_train_scaled,
    eq_mask=eq_train_mask,
    n_epochs=best_epoch_from_search,
    seed=RANDOM_STATE,
)

train_pred = predict_physical(
    model=best_mlp_validation,
    X_physical=X_train,
    x_scaler=x_scaler_search,
    y_scaler=y_scaler_search,
)

val_pred = predict_physical(
    model=best_mlp_validation,
    X_physical=X_val,
    x_scaler=x_scaler_search,
    y_scaler=y_scaler_search,
)

validation_physics_details = (
    calculate_validation_equilibrium_metrics(
        y_true_physical=(
            y_val[TARGETS].to_numpy(dtype=float)
        ),
        y_pred_physical=val_pred,
        equilibrium_physical=eq_val_physical,
        equilibrium_mask=eq_val_mask,
    )
)

validation_best_score = calculate_mean_nrmse(
    y_true_physical=(
        y_val[TARGETS].to_numpy(dtype=float)
    ),
    y_pred_physical=val_pred,
)

validation_best_eeq_ss = float(
    validation_physics_details["Mean_Eeq_SS"]
)
validation_best_tcr = float(
    validation_physics_details["Mean_TCR"]
)
validation_best_epoch = best_epoch_from_search

# Reproducibility check against the search trajectory.
if not np.isclose(
    validation_best_score,
    best_val_mean_nrmse_from_search,
    rtol=0.0,
    atol=1e-8,
):
    print(
        "\nWARNING: selected-model rebuild Validation NRMSE differs "
        "slightly from the grid-search checkpoint."
    )
    print(
        "Grid search:",
        best_val_mean_nrmse_from_search,
        "Rebuild:",
        validation_best_score,
    )

validation_metrics = []

for j, target in enumerate(TARGETS):
    train_rmse = np.sqrt(
        mean_squared_error(
            y_train[target],
            train_pred[:, j],
        )
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val[target],
            val_pred[:, j],
        )
    )

    record = {
        "Target": target,
        "Train_R2": r2_score(
            y_train[target],
            train_pred[:, j],
        ),
        "Train_RMSE": train_rmse,
        "Train_MAE": mean_absolute_error(
            y_train[target],
            train_pred[:, j],
        ),
        "Train_NRMSE": train_rmse / target_range[target],
        "Val_R2": r2_score(
            y_val[target],
            val_pred[:, j],
        ),
        "Val_RMSE": val_rmse,
        "Val_MAE": mean_absolute_error(
            y_val[target],
            val_pred[:, j],
        ),
        "Val_NRMSE": val_rmse / target_range[target],
        "Val_Eeq_SS": np.nan,
        "Val_TCR": np.nan,
    }

    if target in FACTSAGE_TARGETS:
        record["Val_Eeq_SS"] = (
            validation_physics_details[
                f"{target}_Eeq_SS"
            ]
        )
        record["Val_TCR"] = (
            validation_physics_details[
                f"{target}_TCR"
            ]
        )

    validation_metrics.append(record)

validation_metrics_df = pd.DataFrame(
    validation_metrics
)

print("\nBest final-selection MLP-Eq — Training / Validation:")
print(validation_metrics_df)
print("\nSelected validation epoch:", validation_best_epoch)
print("Selected Validation Mean NRMSE:", validation_best_score)
print("Selected Validation Mean Eeq_SS:", validation_best_eeq_ss)
print("Selected Validation Mean TCR:", validation_best_tcr)

validation_metrics_df.to_csv(
    OUTPUT_DIR / "mlp_eq_best_validation_metrics.csv",
    index=False,
)


# ============================================================
# 27. Combine Training + Validation
# ============================================================

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0,
)
y_train_val = pd.concat(
    [y_train, y_val],
    axis=0,
)

print("\nFinal MLP-Eq training data:")
print("X_train_val:", X_train_val.shape)
print("y_train_val:", y_train_val.shape)


# ============================================================
# 28. Refit scalers on Training + Validation
# ============================================================

x_scaler_final = StandardScaler()
y_scaler_final = StandardScaler()

X_train_val_scaled = x_scaler_final.fit_transform(
    X_train_val[FEATURES].to_numpy(dtype=float)
)

y_train_val_scaled = y_scaler_final.fit_transform(
    y_train_val[TARGETS].to_numpy(dtype=float)
)

train_val_metadata_for_physics = pd.concat(
    [
        df.loc[train_idx],
        df.loc[val_idx],
    ],
    axis=0,
).copy().reset_index(drop=True)

(
    eq_train_val_physical,
    eq_train_val_mask,
) = build_equilibrium_targets_for_rows(
    train_val_metadata_for_physics
)

eq_train_val_scaled = scale_equilibrium_targets(
    equilibrium_physical=eq_train_val_physical,
    equilibrium_mask=eq_train_val_mask,
    y_scaler=y_scaler_final,
)

print("Final Training + Validation physics samples:")
print(
    "SS-stage constant-equilibrium samples:",
    int(eq_train_val_mask.sum()),
)
print(
    "Startup/off data-only samples:",
    int((~eq_train_val_mask).sum()),
)


# ============================================================
# 29. Train final M4 on Training + Validation
# ============================================================

final_epochs = int(best_epoch_from_search)

mlp_eq_final, final_training_history_df = train_fixed_epochs(
    params=best_params,
    X_scaled=X_train_val_scaled,
    y_scaled=y_train_val_scaled,
    eq_scaled=eq_train_val_scaled,
    eq_mask=eq_train_val_mask,
    n_epochs=final_epochs,
    seed=RANDOM_STATE,
)

final_training_history_df.to_csv(
    OUTPUT_DIR / "mlp_eq_final_training_history.csv",
    index=False,
)


# ============================================================
# 29. Final Test prediction
# ============================================================

y_test_pred = predict_physical(
    model=mlp_eq_final,
    X_physical=X_test,
    x_scaler=x_scaler_final,
    y_scaler=y_scaler_final,
)


# ============================================================
# 30. Overall Test metrics
# R2 / RMSE / MAE / NRMSE
# ============================================================

overall_metrics = []


for j, target in enumerate(
    TARGETS
):
    test_rmse = np.sqrt(
        mean_squared_error(
            y_test[target],
            y_test_pred[:, j],
        )
    )

    overall_metrics.append(
        {
            "Target": target,

            "R2": r2_score(
                y_test[target],
                y_test_pred[:, j],
            ),

            "RMSE": test_rmse,

            "MAE": mean_absolute_error(
                y_test[target],
                y_test_pred[:, j],
            ),

            "NRMSE": (
                test_rmse
                / target_range[target]
            ),
        }
    )


overall_metrics_df = pd.DataFrame(
    overall_metrics
)


# ============================================================
# 31. Extract Test metadata
# ============================================================

test_metadata_df = (
    df.loc[test_idx]
    .copy()
    .reset_index(drop=True)
)


if len(test_metadata_df) != len(y_test_pred):
    raise ValueError(
        "Test metadata and prediction counts do not match."
    )


# ============================================================
# 32. Define SS-stage Test mask
#
# NRMSE_SS / Eeq_SS / TCR / GCR use ALL Test samples inside:
#     process_stage == "ss_stage"
#
# This includes:
#     local_steady
#     transient
# ============================================================

ss_mask_test = (
    test_metadata_df["process_stage"]
    .eq("ss_stage")
    .to_numpy(dtype=bool)
)


local_steady_mask_test = (
    test_metadata_df["is_local_steady"]
    .to_numpy(dtype=bool)
)


transient_mask_test = (
    test_metadata_df["is_local_transient"]
    .to_numpy(dtype=bool)
)


n_test = len(
    test_metadata_df
)


n_test_ss = int(
    ss_mask_test.sum()
)


n_test_local_steady = int(
    local_steady_mask_test.sum()
)


n_test_transient = int(
    transient_mask_test.sum()
)


if n_test_ss == 0:
    raise ValueError(
        "No SS-stage samples found in Test."
    )


print("\n======================================")
print("Test subset counts")
print("======================================")
print("All Test:", n_test)
print("SS-stage Test:", n_test_ss)
print("Local-steady SS Test:", n_test_local_steady)
print("Transient SS Test:", n_test_transient)


# ============================================================
# 33. Calculate NRMSE_SS
# SS-stage Test = local steady + transient
# ============================================================

ss_metrics = []


for j, target in enumerate(
    TARGETS
):
    measured_ss = (
        y_test[target]
        .to_numpy(dtype=float)[
            ss_mask_test
        ]
    )

    predicted_ss = y_test_pred[
        ss_mask_test,
        j,
    ]

    rmse_ss = np.sqrt(
        np.mean(
            (
                measured_ss
                - predicted_ss
            ) ** 2
        )
    )

    nrmse_ss = (
        rmse_ss
        / target_range[target]
    )

    ss_metrics.append(
        {
            "Target": target,
            "NRMSE_SS": nrmse_ss,
        }
    )


ss_metrics_df = pd.DataFrame(
    ss_metrics
)


# ============================================================
# 34. FactSage data already parsed before training
# ============================================================

# The exact M3 FactSage parser/interpolator was moved before model
# training because M4 needs equilibrium targets inside the loss.


# ============================================================
# 35. Generate FactSage equilibrium values
# for ALL SS-stage Test samples
# ============================================================

ss_test_oxygen = (
    test_metadata_df.loc[
        ss_mask_test,
        "Oxygen concentration",
    ]
    .to_numpy(dtype=float)
)


ss_test_temperature = (
    test_metadata_df.loc[
        ss_mask_test,
        "Trep",
    ]
    .to_numpy(dtype=float)
)


factsage_equilibrium = {}


for target in FACTSAGE_TARGETS:
    factsage_equilibrium[
        target
    ] = interpolate_factsage_values(
        oxygen_values=ss_test_oxygen,
        temperature_values=ss_test_temperature,
        target=target,
    )


# ============================================================
# 36. Calculate Eeq_SS
#
# Eeq_SS =
#     RMSE(model prediction, FactSage equilibrium)
#     / training target range
#
# H2 / CO / CO2 only
# ============================================================

factsage_metrics = []


for target in FACTSAGE_TARGETS:
    target_position = TARGETS.index(
        target
    )

    predicted_ss = y_test_pred[
        ss_mask_test,
        target_position,
    ]

    equilibrium_ss = factsage_equilibrium[
        target
    ]

    rmse_model_vs_eq = np.sqrt(
        np.mean(
            (
                predicted_ss
                - equilibrium_ss
            ) ** 2
        )
    )

    eeq_ss = (
        rmse_model_vs_eq
        / target_range[target]
    )

    factsage_metrics.append(
        {
            "Target": target,
            "Eeq_SS": eeq_ss,
        }
    )


factsage_metrics_df = pd.DataFrame(
    factsage_metrics
)


# ============================================================
# 37. Calculate TCR
#
# Thermodynamic Consistency Rate
#
# A sample is counted as thermodynamically consistent when:
#
# |prediction - equilibrium|
# <=
# |measurement - equilibrium|
#
# Higher = better
# H2 / CO / CO2 only
# ============================================================

tcr_metrics = []


for target in FACTSAGE_TARGETS:
    target_position = TARGETS.index(
        target
    )

    measured_ss = (
        y_test[target]
        .to_numpy(dtype=float)[
            ss_mask_test
        ]
    )

    predicted_ss = y_test_pred[
        ss_mask_test,
        target_position,
    ]

    equilibrium_ss = factsage_equilibrium[
        target
    ]

    experimental_equilibrium_error = np.abs(
        measured_ss
        - equilibrium_ss
    )

    model_equilibrium_error = np.abs(
        predicted_ss
        - equilibrium_ss
    )

    thermodynamically_consistent = (
        model_equilibrium_error
        <= experimental_equilibrium_error
    )

    tcr = float(
        np.mean(
            thermodynamically_consistent
        )
    )

    tcr_metrics.append(
        {
            "Target": target,
            "TCR": tcr,
        }
    )


tcr_metrics_df = pd.DataFrame(
    tcr_metrics
)


# ============================================================
# 38. Calculate GCR
#
# Gradient Consistency Rate
#
# For each SS-stage Test sample:
#
# 1) Model gradient:
#    all 11 temperature sensors are shifted together by +5 degC
#    and -5 degC in ORIGINAL PHYSICAL INPUT SPACE.
#
#    g_model = [f(T + 5) - f(T - 5)] / 10
#
# 2) FactSage gradient:
#    Trep = Tmax
#
#    g_FS = [y_eq(Tmax + 5) - y_eq(Tmax - 5)] / 10
#
# 3) Near-zero FactSage gradients are excluded using:
#
#    epsilon_g,j = 0.01 * max_i |g_FS,ij|
#
# 4) A valid probe is directionally consistent when:
#
#    g_model * g_FS > 0
#
# 5) GCR_j = consistent valid probes / valid probes
#
# H2 / CO / CO2 only.
# CH4 = N/A.
# ============================================================

# ------------------------------------------------------------
# 38.1 Build physical-space +5 / -5 temperature perturbations
# ------------------------------------------------------------

X_ss_test_physical = (
    X_test.loc[
        ss_mask_test,
        FEATURES,
    ]
    .copy()
    .reset_index(drop=True)
)


X_ss_plus = X_ss_test_physical.copy()
X_ss_minus = X_ss_test_physical.copy()


X_ss_plus.loc[
    :,
    TEMP_COLS,
] = (
    X_ss_plus[
        TEMP_COLS
    ]
    + GCR_DELTA_T
)


X_ss_minus.loc[
    :,
    TEMP_COLS,
] = (
    X_ss_minus[
        TEMP_COLS
    ]
    - GCR_DELTA_T
)


# IMPORTANT:
# The +/- 5 degC perturbation is performed BEFORE StandardScaler.
# The perturbed physical inputs are then transformed by the final
# scaler fitted on Training + Validation.

model_prediction_plus = predict_physical(
    model=mlp_eq_final,
    X_physical=X_ss_plus,
    x_scaler=x_scaler_final,
    y_scaler=y_scaler_final,
)


model_prediction_minus = predict_physical(
    model=mlp_eq_final,
    X_physical=X_ss_minus,
    x_scaler=x_scaler_final,
    y_scaler=y_scaler_final,
)


model_gradients = (
    model_prediction_plus
    - model_prediction_minus
) / (
    2.0 * GCR_DELTA_T
)


# ------------------------------------------------------------
# 38.2 FactSage +5 / -5 central finite-difference gradients
# ------------------------------------------------------------

factsage_gradient_plus = {}
factsage_gradient_minus = {}
factsage_gradients = {}


for target in FACTSAGE_TARGETS:
    factsage_gradient_plus[
        target
    ] = interpolate_factsage_values(
        oxygen_values=ss_test_oxygen,
        temperature_values=(
            ss_test_temperature
            + GCR_DELTA_T
        ),
        target=target,
    )

    factsage_gradient_minus[
        target
    ] = interpolate_factsage_values(
        oxygen_values=ss_test_oxygen,
        temperature_values=(
            ss_test_temperature
            - GCR_DELTA_T
        ),
        target=target,
    )

    factsage_gradients[
        target
    ] = (
        factsage_gradient_plus[target]
        - factsage_gradient_minus[target]
    ) / (
        2.0 * GCR_DELTA_T
    )


# ------------------------------------------------------------
# 38.3 Probe-level GCR details
# ------------------------------------------------------------

gcr_probe_details_df = (
    test_metadata_df.loc[
        ss_mask_test,
        [
            "original_row",
            "Time",
            "run_id",
            "process_stage",
            "block_uid",
            "stage_block_id",
            "dynamic_state",
            "is_local_steady",
            "is_local_transient",
            "D_thermal",
            "S_stationarity",
            "physics_weight",
            "Feeding rate",
            "Flow rate",
            "Oxygen concentration",
            "Trep",
            "Tmax_sensor",
        ],
    ]
    .copy()
    .reset_index(drop=True)
)


gcr_metrics = []


for target in FACTSAGE_TARGETS:
    target_position = TARGETS.index(
        target
    )

    g_model = model_gradients[
        :,
        target_position,
    ]

    g_fs = factsage_gradients[
        target
    ]

    max_abs_g_fs = float(
        np.max(
            np.abs(g_fs)
        )
    )

    epsilon_g = (
        GCR_EPS_FRACTION
        * max_abs_g_fs
    )

    valid_probe = (
        np.abs(g_fs)
        > epsilon_g
    )

    gradient_consistent = (
        g_model
        * g_fs
        > 0.0
    )

    n_valid_probe = int(
        valid_probe.sum()
    )

    if n_valid_probe > 0:
        gcr_value = float(
            np.mean(
                gradient_consistent[
                    valid_probe
                ]
            )
        )
    else:
        gcr_value = np.nan

    gcr_metrics.append(
        {
            "Target": target,
            "GCR": gcr_value,
            "GCR_Epsilon": epsilon_g,
            "GCR_Total_SS_Probes": n_test_ss,
            "GCR_Valid_Probes": n_valid_probe,
            "GCR_Valid_Probe_Fraction": (
                n_valid_probe
                / n_test_ss
            ),
        }
    )

    gcr_probe_details_df[
        f"{target}_model_pred_plus"
    ] = model_prediction_plus[
        :,
        target_position,
    ]

    gcr_probe_details_df[
        f"{target}_model_pred_minus"
    ] = model_prediction_minus[
        :,
        target_position,
    ]

    gcr_probe_details_df[
        f"{target}_model_gradient"
    ] = g_model

    gcr_probe_details_df[
        f"{target}_FactSage_plus"
    ] = factsage_gradient_plus[
        target
    ]

    gcr_probe_details_df[
        f"{target}_FactSage_minus"
    ] = factsage_gradient_minus[
        target
    ]

    gcr_probe_details_df[
        f"{target}_FactSage_gradient"
    ] = g_fs

    gcr_probe_details_df[
        f"{target}_GCR_epsilon"
    ] = epsilon_g

    gcr_probe_details_df[
        f"{target}_GCR_valid_probe"
    ] = valid_probe

    gcr_probe_details_df[
        f"{target}_GCR_consistent"
    ] = (
        valid_probe
        & gradient_consistent
    )


# Target-wise GCR table for merge with final metrics.
gcr_metrics_df = pd.DataFrame(
    gcr_metrics
)


valid_gcr_values = (
    gcr_metrics_df[
        "GCR"
    ]
    .dropna()
    .to_numpy(dtype=float)
)


if len(valid_gcr_values) > 0:
    mean_gcr = float(
        np.mean(
            valid_gcr_values
        )
    )
else:
    mean_gcr = np.nan


# A compact GCR summary file also includes CH4 = N/A and Mean GCR.
gcr_summary_df = pd.DataFrame(
    [
        {
            "Target": row["Target"],
            "GCR": row["GCR"],
            "GCR_Epsilon": row["GCR_Epsilon"],
            "GCR_Total_SS_Probes": row["GCR_Total_SS_Probes"],
            "GCR_Valid_Probes": row["GCR_Valid_Probes"],
            "GCR_Valid_Probe_Fraction": row["GCR_Valid_Probe_Fraction"],
        }
        for row in gcr_metrics
    ]
    + [
        {
            "Target": "CH4",
            "GCR": np.nan,
            "GCR_Epsilon": np.nan,
            "GCR_Total_SS_Probes": n_test_ss,
            "GCR_Valid_Probes": np.nan,
            "GCR_Valid_Probe_Fraction": np.nan,
        },
        {
            "Target": "Mean_GCR",
            "GCR": mean_gcr,
            "GCR_Epsilon": np.nan,
            "GCR_Total_SS_Probes": n_test_ss,
            "GCR_Valid_Probes": np.nan,
            "GCR_Valid_Probe_Fraction": np.nan,
        },
    ]
)


print("\n======================================")
print("GCR SUMMARY")
print("======================================")
print(gcr_summary_df)


# ============================================================
# 39. Combine FINAL MLP-Eq metrics
# ============================================================

final_metrics_df = (
    overall_metrics_df
    .merge(
        ss_metrics_df,
        on="Target",
        how="left",
        validate="one_to_one",
    )
    .merge(
        factsage_metrics_df,
        on="Target",
        how="left",
        validate="one_to_one",
    )
    .merge(
        tcr_metrics_df,
        on="Target",
        how="left",
        validate="one_to_one",
    )
    .merge(
        gcr_metrics_df[
            [
                "Target",
                "GCR",
            ]
        ],
        on="Target",
        how="left",
        validate="one_to_one",
    )
)


# CH4:
# Eeq_SS = NaN
# TCR    = NaN
# GCR    = NaN
# because CH4 is not included in the current FactSage
# equilibrium / gradient constraint.

final_metrics_df = final_metrics_df[
    [
        "Target",
        "R2",
        "RMSE",
        "MAE",
        "NRMSE",
        "NRMSE_SS",
        "Eeq_SS",
        "TCR",
        "GCR",
    ]
]


print("\n======================================")
print("FINAL MLP-Eq TEST METRICS")
print("======================================")
print(final_metrics_df)


print("\nMean GCR across H2 / CO / CO2:")
print(mean_gcr)


# ============================================================
# 40. Save final MLP-Eq metrics
# ============================================================

final_metrics_df.to_csv(
    OUTPUT_DIR / "mlp_eq_final_metrics.csv",
    index=False,
)


gcr_summary_df.to_csv(
    OUTPUT_DIR / "mlp_eq_gcr_summary.csv",
    index=False,
)


gcr_probe_details_df.to_csv(
    OUTPUT_DIR / "mlp_eq_gcr_probe_details.csv",
    index=False,
)


# ============================================================
# 41. Save Test predictions
# ============================================================

test_predictions_df = (
    test_metadata_df[
        [
            "original_row",
            "Time",
            "run_id",
            "process_stage",
            "block_uid",
            "stage_block_id",
            "dynamic_state",
            "is_local_steady",
            "is_local_transient",
            "D_thermal",
            "S_stationarity",
            "physics_weight",
            "Feeding rate",
            "Flow rate",
            "Oxygen concentration",
            "Trep",
            "Tmax_sensor",
        ]
    ]
    .copy()
)


for j, target in enumerate(
    TARGETS
):
    measured_values = (
        y_test[target]
        .to_numpy(dtype=float)
    )

    predicted_values = y_test_pred[
        :,
        j,
    ]

    test_predictions_df[
        f"{target}_measured"
    ] = measured_values

    test_predictions_df[
        f"{target}_predicted"
    ] = predicted_values

    test_predictions_df[
        f"{target}_error"
    ] = (
        predicted_values
        - measured_values
    )


# FactSage values only exist for SS-stage rows.
# Startup/off remain NaN.
for target in FACTSAGE_TARGETS:
    equilibrium_full_test = np.full(
        n_test,
        np.nan,
        dtype=float,
    )

    equilibrium_full_test[
        ss_mask_test
    ] = factsage_equilibrium[
        target
    ]

    test_predictions_df[
        f"{target}_FactSage_eq"
    ] = equilibrium_full_test


# Reset index before saving so output row numbering is clean.
test_predictions_df = test_predictions_df.reset_index(
    drop=True
)


test_predictions_df.to_csv(
    OUTPUT_DIR / "mlp_eq_test_predictions.csv",
    index=False,
)


# ============================================================
# 42. Save final MLP-Eq model and scalers
# ============================================================

model_checkpoint = {
    "model_state_dict": {
        key: value.detach().cpu()
        for key, value in mlp_eq_final.state_dict().items()
    },
    "features": FEATURES,
    "targets": TARGETS,
    "best_params": best_params,
    "baseline_mlp_best_params_path": str(
        BASELINE_MLP_BEST_PARAMS_PATH.resolve()
    ),
    "baseline_mlp_run_config_path": str(
        BASELINE_MLP_RUN_CONFIG_PATH.resolve()
    ),
    "baseline_mlp_params": BASE_MLP_PARAMS,
    "search_method": (
        "systematic logarithmic grid search "
        "with Validation Pareto analysis"
    ),
    "lambda_eq": float(
        best_params["lambda_eq"]
    ),
    "selection_rule": (
        "for_each_lambda_select_min_validation_mean_NRMSE_epoch; "
        "construct_validation_Pareto_front_using_mean_NRMSE_and_mean_Eeq_SS; "
        "robust_standardize_Pareto_objectives_using_median_and_IQR; "
        "equal_weight_sum_of_robust_standardized_objectives; "
        "select_minimum_compromise_score"
    ),
    "manual_pareto_selection": False,
    "automatic_pareto_selection": True,
    "pareto_final_selection_criterion": (
        "minimum_equal_weight_robust_standardized_compromise_score"
    ),
    "pareto_robust_scaling": (
        "median_IQR_within_Pareto_non_dominated_set"
    ),
    "pareto_weight_nrmse": PARETO_WEIGHT_NRMSE,
    "pareto_weight_physics": PARETO_WEIGHT_PHYSICS,
    "pareto_nrmse_median": pareto_nrmse_median,
    "pareto_nrmse_iqr": pareto_nrmse_iqr,
    "pareto_physics_median": pareto_eeq_median,
    "pareto_physics_iqr": pareto_eeq_iqr,
    "selected_pareto_compromise_score": float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    ),
    "uses_accuracy_tolerance": False,
    "uses_test_for_selection": False,
    "m3_reference_validation_mean_nrmse": (
        m3_reference_val_mean_nrmse
    ),
    "selected_validation_mean_nrmse": (
        best_val_mean_nrmse_from_search
    ),
    "selected_validation_mean_Eeq_SS": (
        best_val_mean_eeq_ss_from_search
    ),
    "selected_validation_mean_TCR": (
        best_val_mean_tcr_from_search
    ),
    "equilibrium_constraint_stage": "ss_stage",
    "equilibrium_constraint_targets": FACTSAGE_TARGETS,
    "equilibrium_weighting": "constant_1_within_ss_stage",
    "uses_physics_weight_in_training": False,
    "uses_gradient_loss_in_training": False,
    "final_epochs": int(final_epochs),
    "random_state": RANDOM_STATE,
    "gcr_delta_T": GCR_DELTA_T,
    "gcr_eps_fraction": GCR_EPS_FRACTION,
}


torch.save(
    model_checkpoint,
    OUTPUT_DIR / "mlp_eq_final_model.pt",
)


joblib.dump(
    x_scaler_final,
    OUTPUT_DIR / "mlp_eq_x_scaler.pkl",
)


joblib.dump(
    y_scaler_final,
    OUTPUT_DIR / "mlp_eq_y_scaler.pkl",
)


# ============================================================
# 43. Save FactSage long table
# ============================================================

factsage_long_df.to_csv(
    OUTPUT_DIR
    / "mlp_eq_factsage_equilibrium_H2_CO_CO2_long.csv",
    index=False,
)


# ============================================================
# 44. Save Test subset summary
# ============================================================

test_subset_summary_df = pd.DataFrame(
    {
        "Subset": [
            "All Test",
            "SS-stage Test",
            "Local-steady SS Test",
            "Transient SS Test",
        ],
        "N": [
            n_test,
            n_test_ss,
            n_test_local_steady,
            n_test_transient,
        ],
    }
)


test_subset_summary_df.to_csv(
    OUTPUT_DIR / "mlp_eq_test_subset_summary.csv",
    index=False,
)




# ============================================================
# 45. Save compact run configuration
# ============================================================

run_configuration = {
    "random_state": RANDOM_STATE,
    "device": str(DEVICE),
    "search_method": (
        "systematic logarithmic grid search "
        "with Validation Pareto analysis"
    ),
    "max_epochs_per_lambda": MAX_EPOCHS,
    "lambda_eq_search_min": LAMBDA_EQ_MIN,
    "lambda_eq_search_max": LAMBDA_EQ_MAX,
    "n_positive_log_grid_points": N_COARSE_LOG_POINTS,
    "lambda_grid": COARSE_LAMBDA_GRID.tolist(),
    "selection_rule": (
        "for_each_lambda_select_min_validation_mean_NRMSE_epoch; "
        "construct_validation_Pareto_front_using_mean_NRMSE_and_mean_Eeq_SS; "
        "robust_standardize_Pareto_objectives_using_median_and_IQR; "
        "equal_weight_sum_of_robust_standardized_objectives; "
        "select_minimum_compromise_score"
    ),
    "pareto_objective_1": (
        "minimize_validation_mean_NRMSE"
    ),
    "pareto_objective_2": (
        "minimize_validation_mean_Eeq_SS"
    ),
    "manual_pareto_selection": False,
    "automatic_pareto_selection": True,
    "pareto_final_selection_criterion": (
        "minimum_equal_weight_robust_standardized_compromise_score"
    ),
    "pareto_robust_scaling": (
        "median_IQR_within_Pareto_non_dominated_set"
    ),
    "pareto_weight_nrmse": PARETO_WEIGHT_NRMSE,
    "pareto_weight_physics": PARETO_WEIGHT_PHYSICS,
    "pareto_nrmse_median": pareto_nrmse_median,
    "pareto_nrmse_iqr": pareto_nrmse_iqr,
    "pareto_physics_median": pareto_eeq_median,
    "pareto_physics_iqr": pareto_eeq_iqr,
    "selected_pareto_compromise_score": float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    ),
    "pareto_table_path": str(
        PARETO_PATH
    ),
    "pareto_plot_path": str(
        PARETO_PLOT_PATH
    ),
    "uses_accuracy_tolerance": False,
    "uses_test_for_selection": False,
    "uses_early_stopping_during_lambda_search": False,
    "m3_full_trajectory_best_epoch": int(
        m3_confirmation_epoch
    ),
    "m3_full_trajectory_validation_mean_nrmse": float(
        m3_confirmation_val_nrmse
    ),
    "m3_reference_validation_mean_nrmse": float(
        m3_reference_val_mean_nrmse
    ),
    "m3_reference_best_epoch": int(
        m3_reference_best_epoch
    ),
    "selected_validation_mean_nrmse": float(
        best_val_mean_nrmse_from_search
    ),
    "selected_validation_mean_Eeq_SS": float(
        best_val_mean_eeq_ss_from_search
    ),
    "selected_validation_mean_TCR": float(
        best_val_mean_tcr_from_search
    ),
    "best_validation_epoch": int(
        best_epoch_from_search
    ),
    "baseline_mlp_best_params_path": str(
        BASELINE_MLP_BEST_PARAMS_PATH.resolve()
    ),
    "baseline_mlp_run_config_path": str(
        BASELINE_MLP_RUN_CONFIG_PATH.resolve()
    ),
    "baseline_mlp_params": BASE_MLP_PARAMS,
    "lambda_eq": float(
        best_params["lambda_eq"]
    ),
    "equilibrium_constraint_stage": "ss_stage",
    "equilibrium_constraint_targets": FACTSAGE_TARGETS,
    "equilibrium_weighting": "constant_1_within_ss_stage",
    "uses_physics_weight_in_training": False,
    "uses_gradient_loss_in_training": False,
    "final_training_samples": int(
        len(X_train_val)
    ),
    "test_samples": int(n_test),
    "ss_stage_test_samples": int(n_test_ss),
    "gcr_delta_T_degC": GCR_DELTA_T,
    "gcr_eps_fraction": GCR_EPS_FRACTION,
    "mean_gcr_H2_CO_CO2": (
        None
        if np.isnan(mean_gcr)
        else float(mean_gcr)
    ),
}

with open(
    OUTPUT_DIR / "mlp_eq_run_configuration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_configuration,
        file,
        indent=4,
    )


# ============================================================
# 46. Final summary
# ============================================================

print("\n======================================")
print("M4 FINAL MLP-Eq PARETO WORKFLOW COMPLETED")
print("======================================")

print("\nFixed split:")
print(split_counts)

print("\nM3 reference Validation Mean NRMSE:")
print(
    m3_reference_val_mean_nrmse
)

print("\nSearch / selection method:")
print(
    "Systematic logarithmic Grid + Validation Pareto "
    "non-dominated analysis + robust standardized compromise selection"
)

print("\nSelected lambda_eq:")
print(
    best_params[
        "lambda_eq"
    ]
)

print("\nSelected Validation Mean NRMSE:")
print(
    best_val_mean_nrmse_from_search
)

print("\nSelected Validation Mean Eeq_SS:")
print(
    best_val_mean_eeq_ss_from_search
)

print("\nSelected Validation Mean TCR:")
print(
    best_val_mean_tcr_from_search
)

print("\nSelected Pareto compromise score:")
print(
    float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    )
)

print("\nSelected epoch / final epochs:")
print(
    final_epochs
)

print("\nPareto table:")
print(
    PARETO_PATH
)

print("\nPareto figure:")
print(
    PARETO_PLOT_PATH
)

print("\nTest subsets:")
print(
    test_subset_summary_df
)

print("\nFinal Test metrics:")
print(
    final_metrics_df
)

print(
    "\nGCR summary "
    "(reported only; not used for M4 selection):"
)

print(
    gcr_summary_df
)

print("\nMean GCR:")
print(
    mean_gcr
)

print("\nResults folder:")
print(
    OUTPUT_DIR
)

print("\nSaved files:")

saved_files = [
    "mlp_eq_grid_all_representatives.csv",
    "mlp_eq_pareto_non_dominated.csv",
    "mlp_eq_pareto_non_dominated.png",
    "mlp_eq_best_parameters.json",
    "mlp_eq_m3_full_trajectory_confirmation.csv",
    "mlp_eq_model_selection_summary.csv",
    "mlp_eq_best_validation_history.csv",
    "mlp_eq_best_validation_metrics.csv",
    "mlp_eq_final_training_history.csv",
    "mlp_eq_final_metrics.csv",
    "mlp_eq_test_predictions.csv",
    "mlp_eq_test_subset_summary.csv",
    "mlp_eq_gcr_summary.csv",
    "mlp_eq_gcr_probe_details.csv",
    "mlp_eq_final_model.pt",
    "mlp_eq_x_scaler.pkl",
    "mlp_eq_y_scaler.pkl",
    "mlp_eq_run_configuration.json",
    "mlp_eq_factsage_equilibrium_H2_CO_CO2_long.csv",
]

for filename in saved_files:
    print(
        OUTPUT_DIR
        / filename
    )

print("\nPer-lambda full histories:")
print(
    GRID_HISTORY_DIR
)


# In[7]:


# ============================================================
# M4 Pareto Front Figure + Robust Compromise Selection
#
# Objective 1:
#     minimize Validation Mean NRMSE
#
# Objective 2:
#     minimize Validation Mean Eeq_SS
#
# Robust final-selection rule:
#     1) identify Pareto non-dominated representatives;
#     2) robust-standardize both Pareto objectives using:
#
#            z = (f - median(f)) / IQR(f)
#
#     3) use equal weights:
#
#            Score = z_NRMSE + z_Eeq
#
#     4) select the Pareto solution with minimum Score.
#
# Input:
#     MLP_Eq_Pareto_results/
#     mlp_eq_grid_all_representatives.csv
#
# Output:
#     mlp_eq_pareto_non_dominated.csv
#     mlp_eq_pareto_front.png
#     mlp_eq_pareto_front.pdf
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths and selection settings
# ============================================================

RESULT_DIR = Path(
    "MLP_Eq_Pareto_results"
)

INPUT_CSV = (
    RESULT_DIR
    / "mlp_eq_grid_all_representatives.csv"
)

PARETO_CSV = (
    RESULT_DIR
    / "mlp_eq_pareto_non_dominated.csv"
)

FIGURE_PNG = (
    RESULT_DIR
    / "mlp_eq_pareto_front.png"
)

FIGURE_PDF = (
    RESULT_DIR
    / "mlp_eq_pareto_front.pdf"
)


# Equal objective weights AFTER robust standardization.
PARETO_WEIGHT_NRMSE = 1.0
PARETO_WEIGHT_PHYSICS = 1.0

# Numerical tolerance for identifying the selected lambda.
PARETO_LAMBDA_MATCH_RTOL = 1e-10
PARETO_LAMBDA_MATCH_ATOL = 1e-14


if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{INPUT_CSV.resolve()}"
    )


# ============================================================
# 2. Read all representative models
# ============================================================

df = pd.read_csv(
    INPUT_CSV
).copy()


required_columns = [
    "Lambda_Eq",
    "Epoch",
    "Val_Mean_NRMSE",
    "Val_Mean_Eeq_SS",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# Remove invalid rows if any.
df = (
    df.dropna(
        subset=[
            "Lambda_Eq",
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ]
    )
    .reset_index(drop=True)
)


objective_values = df[
    [
        "Val_Mean_NRMSE",
        "Val_Mean_Eeq_SS",
    ]
].to_numpy(dtype=float)


if not np.isfinite(
    objective_values
).all():
    raise ValueError(
        "Pareto objectives contain NaN or infinite values."
    )


# ============================================================
# 3. Identify Pareto non-dominated solutions
# ============================================================

def pareto_non_dominated_mask(
    dataframe,
    objective_columns,
):
    """
    Both objectives are MINIMIZED.

    Candidate j dominates candidate i if:

        objective_j <= objective_i
        for ALL objectives,

    and

        objective_j < objective_i
        for at least ONE objective.
    """

    values = (
        dataframe[
            objective_columns
        ]
        .to_numpy(dtype=float)
    )

    n_points = len(values)

    non_dominated = np.ones(
        n_points,
        dtype=bool,
    )

    for i in range(n_points):

        for j in range(n_points):

            if i == j:
                continue

            j_no_worse = np.all(
                values[j] <= values[i]
            )

            j_strictly_better = np.any(
                values[j] < values[i]
            )

            if (
                j_no_worse
                and j_strictly_better
            ):
                non_dominated[i] = False
                break

    return non_dominated


pareto_mask = (
    pareto_non_dominated_mask(
        dataframe=df,
        objective_columns=[
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
    )
)


df["Is_Pareto"] = pareto_mask


pareto_df = (
    df.loc[
        df["Is_Pareto"]
    ]
    .copy()
    .sort_values(
        [
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(drop=True)
)


dominated_df = (
    df.loc[
        ~df["Is_Pareto"]
    ]
    .copy()
)


if pareto_df.empty:
    raise RuntimeError(
        "No Pareto non-dominated solutions were found."
    )


# ============================================================
# 4. Calculate useful relative changes
# ============================================================

lambda0_rows = df.loc[
    np.isclose(
        df["Lambda_Eq"],
        0.0,
        rtol=0.0,
        atol=PARETO_LAMBDA_MATCH_ATOL,
    )
]


if len(lambda0_rows) != 1:
    raise ValueError(
        "Expected exactly one lambda_eq = 0 row."
    )


lambda0_row = (
    lambda0_rows.iloc[0]
)

m3_nrmse = float(
    lambda0_row[
        "Val_Mean_NRMSE"
    ]
)

lambda0_eeq = float(
    lambda0_row[
        "Val_Mean_Eeq_SS"
    ]
)


pareto_df[
    "NRMSE_Change_vs_M3_percent"
] = (
    (
        pareto_df[
            "Val_Mean_NRMSE"
        ]
        / m3_nrmse
        - 1.0
    )
    * 100.0
)


pareto_df[
    "Eeq_Improvement_vs_Lambda0_percent"
] = (
    (
        1.0
        - pareto_df[
            "Val_Mean_Eeq_SS"
        ]
        / lambda0_eeq
    )
    * 100.0
)


# ============================================================
# 5. Robust standardized Pareto compromise score
# ============================================================

pareto_nrmse_median = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].median()
)

pareto_nrmse_q1 = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].quantile(0.25)
)

pareto_nrmse_q3 = float(
    pareto_df[
        "Val_Mean_NRMSE"
    ].quantile(0.75)
)

pareto_nrmse_iqr = float(
    pareto_nrmse_q3
    - pareto_nrmse_q1
)


pareto_eeq_median = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].median()
)

pareto_eeq_q1 = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.25)
)

pareto_eeq_q3 = float(
    pareto_df[
        "Val_Mean_Eeq_SS"
    ].quantile(0.75)
)

pareto_eeq_iqr = float(
    pareto_eeq_q3
    - pareto_eeq_q1
)


if pareto_nrmse_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean NRMSE has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


if pareto_eeq_iqr <= 0.0:
    raise ValueError(
        "Pareto Validation Mean Eeq_SS has zero or negative IQR; "
        "robust scaling cannot be performed."
    )


pareto_df[
    "Robust_Z_NRMSE"
] = (
    pareto_df[
        "Val_Mean_NRMSE"
    ]
    - pareto_nrmse_median
) / pareto_nrmse_iqr


pareto_df[
    "Robust_Z_Eeq_SS"
] = (
    pareto_df[
        "Val_Mean_Eeq_SS"
    ]
    - pareto_eeq_median
) / pareto_eeq_iqr


pareto_df[
    "Pareto_Compromise_Score"
] = (
    PARETO_WEIGHT_NRMSE
    * pareto_df[
        "Robust_Z_NRMSE"
    ]
    +
    PARETO_WEIGHT_PHYSICS
    * pareto_df[
        "Robust_Z_Eeq_SS"
    ]
)


selected_row = (
    pareto_df
    .sort_values(
        [
            "Pareto_Compromise_Score",
            "Val_Mean_NRMSE",
            "Val_Mean_Eeq_SS",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )
    .iloc[0]
    .copy()
)


selected_lambda_eq = float(
    selected_row[
        "Lambda_Eq"
    ]
)


pareto_df[
    "Is_Selected_Compromise"
] = np.isclose(
    pareto_df[
        "Lambda_Eq"
    ].to_numpy(dtype=float),
    selected_lambda_eq,
    rtol=PARETO_LAMBDA_MATCH_RTOL,
    atol=PARETO_LAMBDA_MATCH_ATOL,
)


# ============================================================
# 6. Save Pareto table
# ============================================================

pareto_df.to_csv(
    PARETO_CSV,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n======================================"
)

print(
    "M4 PARETO NON-DOMINATED SOLUTIONS"
)

print(
    "======================================"
)


display_columns = [
    "Lambda_Eq",
    "Epoch",
    "Val_Mean_NRMSE",
    "NRMSE_Change_vs_M3_percent",
    "Val_Mean_Eeq_SS",
    "Eeq_Improvement_vs_Lambda0_percent",
    "Robust_Z_NRMSE",
    "Robust_Z_Eeq_SS",
    "Pareto_Compromise_Score",
    "Is_Selected_Compromise",
]


if "Val_Mean_TCR" in pareto_df.columns:
    display_columns.append(
        "Val_Mean_TCR"
    )


print(
    pareto_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "M4 ROBUST PARETO COMPROMISE SELECTION"
)

print(
    "======================================"
)

print(
    "Selection rule:"
)

print(
    "Minimum equal-weight robust standardized "
    "compromise score among Pareto solutions"
)

print(
    "\nPareto NRMSE median:",
    pareto_nrmse_median,
)

print(
    "Pareto NRMSE IQR:",
    pareto_nrmse_iqr,
)

print(
    "Pareto Eeq_SS median:",
    pareto_eeq_median,
)

print(
    "Pareto Eeq_SS IQR:",
    pareto_eeq_iqr,
)

print(
    "\nSelected lambda_eq:",
    selected_lambda_eq,
)

print(
    "Selected epoch:",
    int(
        selected_row[
            "Epoch"
        ]
    ),
)

print(
    "Selected Validation Mean NRMSE:",
    float(
        selected_row[
            "Val_Mean_NRMSE"
        ]
    ),
)

print(
    "Selected Validation Mean Eeq_SS:",
    float(
        selected_row[
            "Val_Mean_Eeq_SS"
        ]
    ),
)

print(
    "Selected compromise score:",
    float(
        selected_row[
            "Pareto_Compromise_Score"
        ]
    ),
)


# ============================================================
# 7. Elsevier-style figure settings
# ============================================================

# 14 cm width
CM_TO_INCH = 1.0 / 2.54

FIG_WIDTH_CM = 14.0
FIG_HEIGHT_CM = 9.0


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


fig, ax = plt.subplots(
    figsize=(
        FIG_WIDTH_CM * CM_TO_INCH,
        FIG_HEIGHT_CM * CM_TO_INCH,
    )
)


# ============================================================
# 8. Plot dominated representative models
# ============================================================

if not dominated_df.empty:

    ax.scatter(
        dominated_df[
            "Val_Mean_NRMSE"
        ],
        dominated_df[
            "Val_Mean_Eeq_SS"
        ],
        s=20,
        alpha=0.35,
        marker="o",
        label="Dominated solutions",
        zorder=1,
    )


# ============================================================
# 9. Plot Pareto front
# ============================================================

ax.plot(
    pareto_df[
        "Val_Mean_NRMSE"
    ],
    pareto_df[
        "Val_Mean_Eeq_SS"
    ],
    marker="o",
    markersize=5,
    linewidth=1.2,
    label="Pareto non-dominated front",
    zorder=3,
)


# ============================================================
# 10. Mark lambda = 0 baseline
# ============================================================

baseline_x = float(
    lambda0_row[
        "Val_Mean_NRMSE"
    ]
)

baseline_y = float(
    lambda0_row[
        "Val_Mean_Eeq_SS"
    ]
)


ax.scatter(
    baseline_x,
    baseline_y,
    marker="s",
    s=45,
    label=r"$\lambda_{eq}=0$ (M3)",
    zorder=4,
)


# ============================================================
# 11. Mark the selected robust compromise solution
# ============================================================

selected_x = float(
    selected_row[
        "Val_Mean_NRMSE"
    ]
)

selected_y = float(
    selected_row[
        "Val_Mean_Eeq_SS"
    ]
)


ax.scatter(
    selected_x,
    selected_y,
    marker="*",
    s=120,
    label="Selected robust compromise",
    zorder=6,
)


# ============================================================
# 12. Label every Pareto point with lambda_eq
# ============================================================

# Alternate label offsets to reduce overlap.
offsets = [
    (5, 5),
    (5, -12),
    (5, 7),
    (-28, 5),
]


for i, row in pareto_df.iterrows():

    lambda_value = float(
        row[
            "Lambda_Eq"
        ]
    )

    x_value = float(
        row[
            "Val_Mean_NRMSE"
        ]
    )

    y_value = float(
        row[
            "Val_Mean_Eeq_SS"
        ]
    )

    offset = offsets[
        i % len(offsets)
    ]

    ax.annotate(
        rf"$\lambda={lambda_value:.4g}$",
        xy=(
            x_value,
            y_value,
        ),
        xytext=offset,
        textcoords="offset points",
        fontsize=7,
        zorder=5,
    )


ax.annotate(
    (
        r"Selected "
        rf"$\lambda_{{eq}}={selected_lambda_eq:.4g}$"
    ),
    xy=(
        selected_x,
        selected_y,
    ),
    xytext=(
        8,
        12,
    ),
    textcoords="offset points",
    fontsize=7,
    zorder=7,
)


# ============================================================
# 13. Axis labels
# ============================================================

ax.set_xlabel(
    "Validation Mean NRMSE"
)

ax.set_ylabel(
    r"Validation Mean $E_{\mathrm{eq,SS}}$"
)


# No title for a manuscript figure.
# The explanation belongs in the figure caption.


# ============================================================
# 14. Grid
# ============================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.45,
)


# ============================================================
# 15. Legend
# ============================================================

legend = ax.legend(
    loc="best",
    frameon=True,
    fancybox=False,
)


legend.get_frame().set_linewidth(
    0.8
)


# ============================================================
# 16. Final layout
# ============================================================

fig.tight_layout()


# ============================================================
# 17. Save figure
# ============================================================

fig.savefig(
    FIGURE_PNG,
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# 18. Final information
# ============================================================

print(
    "\nNumber of all representative models:",
    len(df),
)

print(
    "Number of Pareto non-dominated models:",
    len(pareto_df),
)

print(
    "\nSelected robust-compromise lambda_eq:",
    selected_lambda_eq,
)

print(
    "\nPareto CSV saved to:"
)

print(
    PARETO_CSV.resolve()
)

print(
    "\nPNG figure saved to:"
)

print(
    FIGURE_PNG.resolve()
)

print(
    "\nPDF figure saved to:"
)

print(
    FIGURE_PDF.resolve()
)


# In[1]:


# ============================================================
# SUPPLEMENTAL TEST DIAGNOSTICS — STANDALONE VERSION
#
# PURPOSE
# -------
# Add missing Test diagnostics AFTER the original model workflow
# has already been completed.
#
# IMPORTANT:
#     - NO model training
#     - NO hyperparameter search
#     - NO model loading
#     - NO new neural-network inference
#     - NO Test prediction rerun
#     - NO FactSage interpolation rerun
#
# This cell ONLY reads previously saved CSV files.
#
# Added metrics:
#     1) NRMSE_LocalSteady_SS
#     2) NRMSE_Transient_SS
#     3) Grad_MSE_Physical
#     4) Grad_RMSE_Physical
#
# Gradient metrics use ALL finite SS-stage Test probes.
# GCR epsilon filtering is NOT applied to Gradient MSE/RMSE.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Select model
# ============================================================

# Choose ONE:
#
# MODEL = "M3"
# MODEL = "M4"
# MODEL = "M5"

MODEL = "M4"


# ============================================================
# 2. Model-specific saved-result locations
# ============================================================

MODEL_CONFIG = {

    "M3": {
        "output_dir": Path(
            "MLP_TPE_results"
        ),
        "prefix": "mlp",
    },

    "M4": {
        "output_dir": Path(
            "MLP_Eq_Pareto_results"
        ),
        "prefix": "mlp_eq",
    },

    "M5": {
        "output_dir": Path(
            "MLP_Eq_Grad_Pareto_results"
        ),
        "prefix": "mlp_eq_grad",
    },
}


if MODEL not in MODEL_CONFIG:

    raise ValueError(
        f"Unsupported MODEL: {MODEL}. "
        "Use 'M3', 'M4', or 'M5'."
    )


config = MODEL_CONFIG[
    MODEL
]

OUTPUT_DIR = config[
    "output_dir"
]

FILE_PREFIX = config[
    "prefix"
]


# ============================================================
# 3. Required existing files
# ============================================================

final_metrics_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_final_metrics.csv"
)


test_predictions_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_test_predictions.csv"
)


gcr_probe_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_gcr_probe_details.csv"
)


required_files = [
    final_metrics_path,
    test_predictions_path,
    gcr_probe_path,
]


missing_files = [
    path
    for path in required_files
    if not path.exists()
]


if missing_files:

    print(
        "\nCurrent working directory:"
    )

    print(
        Path.cwd()
    )

    print(
        "\nExpected result directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nMissing files:"
    )

    for path in missing_files:

        print(
            "  ",
            path
        )

    raise FileNotFoundError(
        "\nThe saved result files could not be found.\n"
        "This does NOT mean that the model must be retrained.\n"
        "Check that this Notebook is being run from its original "
        "M3 / M4 / M5 folder."
    )


# ============================================================
# 4. Read existing saved results
# ============================================================

final_metrics_df = pd.read_csv(
    final_metrics_path
)


test_predictions_df = pd.read_csv(
    test_predictions_path
)


gcr_probe_details_df = pd.read_csv(
    gcr_probe_path
)


print(
    "\n======================================"
)

print(
    f"{MODEL} SAVED RESULTS LOADED"
)

print(
    "======================================"
)


print(
    "\nFinal metrics:"
)

print(
    final_metrics_path.resolve()
)


print(
    "\nTest predictions:"
)

print(
    test_predictions_path.resolve()
)


print(
    "\nGCR probe details:"
)

print(
    gcr_probe_path.resolve()
)


# ============================================================
# 5. Constants
# ============================================================

TARGETS = [
    "H2",
    "CO",
    "CO2",
    "CH4",
]


FACTSAGE_TARGETS = [
    "H2",
    "CO",
    "CO2",
]


# ============================================================
# 6. Validate final metrics
# ============================================================

required_metric_columns = [
    "Target",
    "RMSE",
    "NRMSE",
]


missing_metric_columns = [
    column
    for column in required_metric_columns
    if column
    not in final_metrics_df.columns
]


if missing_metric_columns:

    raise ValueError(
        "Missing columns in final metrics:\n"
        f"{missing_metric_columns}"
    )


# ============================================================
# 7. Recover the SAME NRMSE normalization ranges
#
# Original definition:
#
#     NRMSE = RMSE / target_range
#
# therefore:
#
#     target_range = RMSE / NRMSE
#
# This avoids reloading the original training dataset and
# guarantees the same normalization denominator as the
# already-published final metrics.
# ============================================================

target_range = {}


for target in TARGETS:

    row = final_metrics_df.loc[
        final_metrics_df[
            "Target"
        ].eq(
            target
        )
    ]


    if len(row) != 1:

        raise ValueError(
            f"Expected exactly one final-metrics row "
            f"for {target}, found {len(row)}."
        )


    rmse = float(
        row.iloc[0][
            "RMSE"
        ]
    )


    nrmse = float(
        row.iloc[0][
            "NRMSE"
        ]
    )


    if (
        not np.isfinite(rmse)
        or not np.isfinite(nrmse)
        or nrmse <= 0.0
    ):

        raise ValueError(
            f"Invalid RMSE / NRMSE for {target}."
        )


    target_range[
        target
    ] = (
        rmse
        / nrmse
    )


print(
    "\nRecovered NRMSE normalization ranges:"
)

for target in TARGETS:

    print(
        f"  {target}: "
        f"{target_range[target]:.10f}"
    )


# ============================================================
# 8. Robust boolean conversion
# ============================================================

def to_bool_array(
    series,
):

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.to_numpy(
            dtype=bool
        )


    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    ).to_numpy(
        dtype=bool
    )


# ============================================================
# 9. Validate Test prediction metadata
# ============================================================

required_prediction_columns = [
    "process_stage",
    "is_local_steady",
    "is_local_transient",
]


for target in TARGETS:

    required_prediction_columns.extend(
        [
            f"{target}_measured",
            f"{target}_predicted",
        ]
    )


missing_prediction_columns = [
    column
    for column
    in required_prediction_columns
    if column
    not in test_predictions_df.columns
]


if missing_prediction_columns:

    raise ValueError(
        "Missing columns in Test predictions:\n"
        f"{missing_prediction_columns}"
    )


# ============================================================
# 10. Define SS / local-steady / transient Test masks
# ============================================================

ss_mask = (
    test_predictions_df[
        "process_stage"
    ]
    .astype(str)
    .str.strip()
    .eq(
        "ss_stage"
    )
    .to_numpy(
        dtype=bool
    )
)


local_steady_flag = (
    to_bool_array(
        test_predictions_df[
            "is_local_steady"
        ]
    )
)


transient_flag = (
    to_bool_array(
        test_predictions_df[
            "is_local_transient"
        ]
    )
)


local_steady_ss_mask = (
    ss_mask
    &
    local_steady_flag
)


transient_ss_mask = (
    ss_mask
    &
    transient_flag
)


n_test = len(
    test_predictions_df
)


n_ss = int(
    ss_mask.sum()
)


n_local_steady = int(
    local_steady_ss_mask.sum()
)


n_transient = int(
    transient_ss_mask.sum()
)


print(
    "\n======================================"
)

print(
    f"{MODEL} TEST SUBSET COUNTS"
)

print(
    "======================================"
)


print(
    "All Test:",
    n_test
)


print(
    "SS-stage Test:",
    n_ss
)


print(
    "Local-steady SS Test:",
    n_local_steady
)


print(
    "Transient SS Test:",
    n_transient
)


if (
    n_local_steady
    + n_transient
    != n_ss
):

    print(
        "\nWARNING:"
        "\nLocal-steady + transient does not "
        "equal total SS-stage samples."
    )


# ============================================================
# 11. Helper: calculate NRMSE for a Test subset
# ============================================================

def calculate_subset_nrmse(
    measured,
    predicted,
    mask,
    normalization_range,
):

    measured = np.asarray(
        measured,
        dtype=float,
    )


    predicted = np.asarray(
        predicted,
        dtype=float,
    )


    mask = np.asarray(
        mask,
        dtype=bool,
    )


    n_subset = int(
        mask.sum()
    )


    if n_subset == 0:

        return np.nan


    errors = (
        predicted[
            mask
        ]
        -
        measured[
            mask
        ]
    )


    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


    nrmse = (
        rmse
        / float(
            normalization_range
        )
    )


    return float(
        nrmse
    )


# ============================================================
# 12. Calculate local-steady and transient NRMSE
# ============================================================

subset_records = []


for target in TARGETS:

    measured = (
        test_predictions_df[
            f"{target}_measured"
        ]
        .to_numpy(
            dtype=float
        )
    )


    predicted = (
        test_predictions_df[
            f"{target}_predicted"
        ]
        .to_numpy(
            dtype=float
        )
    )


    nrmse_local_steady = (
        calculate_subset_nrmse(
            measured=measured,
            predicted=predicted,
            mask=local_steady_ss_mask,
            normalization_range=(
                target_range[
                    target
                ]
            ),
        )
    )


    nrmse_transient = (
        calculate_subset_nrmse(
            measured=measured,
            predicted=predicted,
            mask=transient_ss_mask,
            normalization_range=(
                target_range[
                    target
                ]
            ),
        )
    )


    subset_records.append(
        {
            "Target": target,

            "NRMSE_LocalSteady_SS": (
                nrmse_local_steady
            ),

            "NRMSE_Transient_SS": (
                nrmse_transient
            ),
        }
    )


subset_metrics_df = pd.DataFrame(
    subset_records
)


# ============================================================
# 13. Validate saved gradient columns
# ============================================================

required_gradient_columns = []


for target in FACTSAGE_TARGETS:

    required_gradient_columns.extend(
        [
            f"{target}_model_gradient",
            f"{target}_FactSage_gradient",
        ]
    )


missing_gradient_columns = [
    column
    for column
    in required_gradient_columns
    if column
    not in gcr_probe_details_df.columns
]


if missing_gradient_columns:

    print(
        "\nAvailable columns in GCR probe file:"
    )

    print(
        list(
            gcr_probe_details_df.columns
        )
    )

    raise ValueError(
        "\nMissing saved gradient columns:\n"
        f"{missing_gradient_columns}"
    )


# ============================================================
# 14. Calculate Test Gradient MSE / RMSE
#
# IMPORTANT:
#
# Use ALL finite SS-stage gradient probes.
#
# Unlike GCR:
#     NO epsilon_g filtering is applied here.
#
# This matches the continuous gradient-residual concept used
# for M5 gradient-loss / Validation Gradient MSE.
# ============================================================

gradient_records = []

all_gradient_residuals = []

gradient_probe_complete_df = (
    gcr_probe_details_df
    .copy()
)


for target in FACTSAGE_TARGETS:

    model_gradient = (
        gcr_probe_details_df[
            f"{target}_model_gradient"
        ]
        .to_numpy(
            dtype=float
        )
    )


    factsage_gradient = (
        gcr_probe_details_df[
            f"{target}_FactSage_gradient"
        ]
        .to_numpy(
            dtype=float
        )
    )


    finite_mask = (
        np.isfinite(
            model_gradient
        )
        &
        np.isfinite(
            factsage_gradient
        )
    )


    n_valid = int(
        finite_mask.sum()
    )


    if n_valid == 0:

        grad_mse = np.nan
        grad_rmse = np.nan

    else:

        residual = (
            model_gradient[
                finite_mask
            ]
            -
            factsage_gradient[
                finite_mask
            ]
        )


        squared_error = (
            residual ** 2
        )


        grad_mse = float(
            np.mean(
                squared_error
            )
        )


        grad_rmse = float(
            np.sqrt(
                grad_mse
            )
        )


        all_gradient_residuals.extend(
            residual.tolist()
        )


    gradient_records.append(
        {
            "Target": target,

            "Grad_MSE_Physical": (
                grad_mse
            ),

            "Grad_RMSE_Physical": (
                grad_rmse
            ),

            "N_Gradient_Probes": (
                n_valid
            ),
        }
    )


    residual_full = np.full(
        len(
            gcr_probe_details_df
        ),
        np.nan,
        dtype=float,
    )


    squared_error_full = np.full(
        len(
            gcr_probe_details_df
        ),
        np.nan,
        dtype=float,
    )


    residual_full[
        finite_mask
    ] = (
        model_gradient[
            finite_mask
        ]
        -
        factsage_gradient[
            finite_mask
        ]
    )


    squared_error_full[
        finite_mask
    ] = (
        residual_full[
            finite_mask
        ]
        ** 2
    )


    gradient_probe_complete_df[
        f"{target}_gradient_residual"
    ] = residual_full


    gradient_probe_complete_df[
        f"{target}_gradient_squared_error"
    ] = squared_error_full


gradient_metrics_df = pd.DataFrame(
    gradient_records
)


# ============================================================
# 15. Global Gradient MSE / RMSE across H2 / CO / CO2
# ============================================================

all_gradient_residuals = np.asarray(
    all_gradient_residuals,
    dtype=float,
)


if len(
    all_gradient_residuals
) == 0:

    global_grad_mse = np.nan
    global_grad_rmse = np.nan

else:

    global_grad_mse = float(
        np.mean(
            all_gradient_residuals
            ** 2
        )
    )


    global_grad_rmse = float(
        np.sqrt(
            global_grad_mse
        )
    )


mean_target_grad_rmse = float(
    gradient_metrics_df[
        "Grad_RMSE_Physical"
    ].mean()
)


global_gradient_row = pd.DataFrame(
    [
        {
            "Target": (
                "Global_H2_CO_CO2"
            ),

            "Grad_MSE_Physical": (
                global_grad_mse
            ),

            "Grad_RMSE_Physical": (
                global_grad_rmse
            ),

            "N_Gradient_Probes": (
                int(
                    len(
                        all_gradient_residuals
                    )
                )
            ),
        }
    ]
)


gradient_metrics_complete_df = pd.concat(
    [
        gradient_metrics_df,
        global_gradient_row,
    ],
    ignore_index=True,
)


# ============================================================
# 16. Merge target-wise added metrics
# ============================================================

target_added_metrics_df = (
    subset_metrics_df
    .merge(
        gradient_metrics_df[
            [
                "Target",
                "Grad_MSE_Physical",
                "Grad_RMSE_Physical",
            ]
        ],
        on="Target",
        how="left",
        validate="one_to_one",
    )
)


# CH4 gradient metrics remain NaN by design.


# ============================================================
# 17. Build final complete metrics table
# ============================================================

final_metrics_complete_df = (
    final_metrics_df
    .merge(
        target_added_metrics_df,
        on="Target",
        how="left",
        validate="one_to_one",
    )
)


preferred_columns = [
    "Target",
    "R2",
    "RMSE",
    "MAE",
    "NRMSE",
    "NRMSE_SS",
    "NRMSE_LocalSteady_SS",
    "NRMSE_Transient_SS",
    "Eeq_SS",
    "TCR",
    "Grad_MSE_Physical",
    "Grad_RMSE_Physical",
    "GCR",
]


ordered_columns = [
    column
    for column in preferred_columns
    if column
    in final_metrics_complete_df.columns
]


other_columns = [
    column
    for column
    in final_metrics_complete_df.columns
    if column
    not in ordered_columns
]


final_metrics_complete_df = (
    final_metrics_complete_df[
        ordered_columns
        + other_columns
    ]
)


# ============================================================
# 18. Model-level summary
# ============================================================

model_summary_df = pd.DataFrame(
    [
        {
            "Model": MODEL,

            "N_Test": (
                n_test
            ),

            "N_SS_Test": (
                n_ss
            ),

            "N_LocalSteady_SS_Test": (
                n_local_steady
            ),

            "N_Transient_SS_Test": (
                n_transient
            ),

            "Mean_NRMSE_LocalSteady_SS": float(
                subset_metrics_df[
                    "NRMSE_LocalSteady_SS"
                ].mean()
            ),

            "Mean_NRMSE_Transient_SS": float(
                subset_metrics_df[
                    "NRMSE_Transient_SS"
                ].mean()
            ),

            "Test_Global_Grad_MSE_Physical": (
                global_grad_mse
            ),

            "Test_Global_Grad_RMSE_Physical": (
                global_grad_rmse
            ),

            "Test_Mean_Target_Grad_RMSE_Physical": (
                mean_target_grad_rmse
            ),

            "Gradient_MSE_GCR_Epsilon_Filter": (
                False
            ),
        }
    ]
)


# ============================================================
# 19. Print results
# ============================================================

print(
    "\n======================================"
)

print(
    f"{MODEL} COMPLETE TEST METRICS"
)

print(
    "======================================"
)


print(
    final_metrics_complete_df
    .to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    f"{MODEL} TEST GRADIENT METRICS"
)

print(
    "======================================"
)


print(
    gradient_metrics_complete_df
    .to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    f"{MODEL} MODEL-LEVEL SUMMARY"
)

print(
    "======================================"
)


print(
    model_summary_df
    .to_string(
        index=False
    )
)


# ============================================================
# 20. Save NEW files
#
# Original files are NOT overwritten.
# ============================================================

complete_metrics_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_final_metrics_complete.csv"
)


gradient_metrics_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_test_gradient_metrics.csv"
)


summary_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_additional_test_metrics_summary.csv"
)


gradient_probe_complete_path = (
    OUTPUT_DIR
    / f"{FILE_PREFIX}_gradient_test_probe_details.csv"
)


final_metrics_complete_df.to_csv(
    complete_metrics_path,
    index=False,
    encoding="utf-8-sig",
)


gradient_metrics_complete_df.to_csv(
    gradient_metrics_path,
    index=False,
    encoding="utf-8-sig",
)


model_summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


gradient_probe_complete_df.to_csv(
    gradient_probe_complete_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 21. Final confirmation
# ============================================================

print(
    "\n======================================"
)

print(
    f"{MODEL} SUPPLEMENT COMPLETED"
)

print(
    "======================================"
)


print(
    "\nNO training was performed."
)


print(
    "NO hyperparameter search was performed."
)


print(
    "NO model inference was performed."
)


print(
    "NO FactSage interpolation was rerun."
)


print(
    "\nSaved:"
)


print(
    complete_metrics_path.resolve()
)


print(
    gradient_metrics_path.resolve()
)


print(
    summary_path.resolve()
)


print(
    gradient_probe_complete_path.resolve()
)


# In[ ]:




