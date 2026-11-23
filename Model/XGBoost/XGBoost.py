#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# XGBoost + Optuna TPE
# Using ONE predefined fixed Train/Validation/Test split
#
# Final XGBoost metrics:
#     R2
#     RMSE
#     MAE
#     NRMSE
#     NRMSE_SS
#     Eeq_SS
#     TCR
#
# Important:
#     XGBoost DOES NOT generate a new split.
#     XGBoost DOES NOT recalculate D_thermal / stationarity.
#     XGBoost DOES NOT use physics_weight during training.
#     XGBoost DOES NOT calculate GCR.
#
#     All split and stationarity information is read from:
#         fixed_split_index.csv
#
#     The same split must be used by:
#         RF
#         XGBoost
#         MLP
#         Proposed model
# ============================================================

from pathlib import Path
import re
import json
import joblib
import optuna
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# 1. Basic settings
# ============================================================

RANDOM_STATE = 42
N_TRIALS = 100


# ------------------------------------------------------------
# Experimental data
# ------------------------------------------------------------

DATA_PATH = Path(
    "../20250606_oxygen enrich_whole process_"
    "linear interpolation.csv"
)


# ------------------------------------------------------------
# Fixed split generated independently before model training
#
# Current Notebook/script assumed to be inside:
#     PINN ver2/XGBoost/
#
# Fixed split folder:
#     PINN ver2/fixed_split_results/
# ------------------------------------------------------------

FIXED_SPLIT_PATH = Path(
    "../fixed_split_results/fixed_split_index.csv"
)


# ------------------------------------------------------------
# FactSage equilibrium data
# ------------------------------------------------------------

FACTSAGE_PATH = Path(
    "../Fig_FactSage_data.xlsx"
)


# ------------------------------------------------------------
# XGBoost results folder
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "XGBoost_TPE_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


optuna.logging.set_verbosity(
    optuna.logging.WARNING
)


# ============================================================
# 2. Features and targets
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


# FactSage does not contain CH4 in the present framework
FACTSAGE_TARGETS = [
    "H2",
    "CO",
    "CO2",
]


# ============================================================
# 3. Check input files
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


print("======================================")
print("Input files")
print("======================================")
print("Experimental data:", DATA_PATH.resolve())
print("Fixed split:", FIXED_SPLIT_PATH.resolve())
print("FactSage:", FACTSAGE_PATH.resolve())
print("Output folder:", OUTPUT_DIR.resolve())


# ============================================================
# 4. Load original experimental data
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
# 5. Load predefined fixed split
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
# 6. Verify fixed split matches original data
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
# 7. Merge fixed split information
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
# 8. Verify predefined split
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
# 9. Print fixed split summary
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
# 10. Representative temperature
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
# 11. Prepare Training / Validation / Test data
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
# 12. Training target ranges
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
# 13. Function for building multi-output XGBoost
# ============================================================

# One independent XGBRegressor is trained for each gas.
# All four targets use the same Optuna-proposed hyperparameters.

def build_multioutput_xgb(
    params,
):
    base_model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    model = MultiOutputRegressor(
        base_model,
        n_jobs=1,
    )

    return model


# ============================================================
# 14. TPE objective function
# ============================================================

# Optimization objective:
# Mean Validation NRMSE across H2 / CO / CO2 / CH4
# Lower = better
#
def objective(
    trial,
):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1200,
            step=50,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            10,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            12,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
            step=0.05,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.50,
            1.00,
            step=0.05,
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
            step=0.1,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            100.0,
            log=True,
        ),
    }


    model = build_multioutput_xgb(
        params
    )


    # Train on TRAIN only.
    model.fit(
        X_train,
        y_train,
    )


    val_pred = model.predict(
        X_val
    )


    nrmse_values = []


    for j, target in enumerate(
        TARGETS
    ):
        rmse = np.sqrt(
            mean_squared_error(
                y_val[target],
                val_pred[:, j],
            )
        )


        nrmse = (
            rmse
            / target_range[target]
        )


        nrmse_values.append(
            nrmse
        )


        trial.set_user_attr(
            f"{target}_RMSE",
            float(rmse),
        )


        trial.set_user_attr(
            f"{target}_NRMSE",
            float(nrmse),
        )


    return float(
        np.mean(nrmse_values)
    )


# ============================================================
# 15. Run TPE optimization
# ============================================================

sampler = optuna.samplers.TPESampler(
    seed=RANDOM_STATE
)


study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="XGBoost_TPE_Optimization",
)


print("\nStarting XGBoost TPE optimization...")


study.optimize(
    objective,
    n_trials=N_TRIALS,
    gc_after_trial=True,
    show_progress_bar=True,
)


best_score = study.best_value
best_params = study.best_params


print("\n======================================")
print("TPE optimization completed")
print("======================================")
print("\nBest validation Mean NRMSE:")
print(best_score)
print("\nBest XGBoost hyperparameters:")


for key, value in best_params.items():
    print(f"{key}: {value}")


# ============================================================
# 16. Save TPE results
# ============================================================

with open(
    OUTPUT_DIR / "xgb_best_parameters.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        best_params,
        file,
        indent=4,
    )


search_results = []


for trial in study.trials:
    if trial.value is None:
        continue


    result = {
        "Trial": trial.number + 1,
        "Mean_Val_NRMSE": trial.value,
    }


    result.update(
        trial.params
    )


    for target in TARGETS:
        result[
            f"{target}_RMSE"
        ] = trial.user_attrs.get(
            f"{target}_RMSE"
        )


        result[
            f"{target}_NRMSE"
        ] = trial.user_attrs.get(
            f"{target}_NRMSE"
        )


    search_results.append(
        result
    )


search_results_df = (
    pd.DataFrame(search_results)
    .sort_values("Mean_Val_NRMSE")
    .reset_index(drop=True)
)


print("\nTop 10 TPE combinations:")
print(
    search_results_df.head(10)
)


search_results_df.to_csv(
    OUTPUT_DIR / "xgb_tpe_search_results.csv",
    index=False,
)


# ============================================================
# 17. Rebuild best XGBoost on Training only
# ============================================================

best_xgb_validation = build_multioutput_xgb(
    best_params
)


best_xgb_validation.fit(
    X_train,
    y_train,
)


train_pred = best_xgb_validation.predict(
    X_train
)


val_pred = best_xgb_validation.predict(
    X_val
)


# ============================================================
# 18. Training and Validation diagnostics
# These are NOT final paper comparison metrics.
# ============================================================

validation_metrics = []


for j, target in enumerate(
    TARGETS
):
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


    validation_metrics.append(
        {
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

            "Train_NRMSE": (
                train_rmse
                / target_range[target]
            ),

            "Val_R2": r2_score(
                y_val[target],
                val_pred[:, j],
            ),

            "Val_RMSE": val_rmse,

            "Val_MAE": mean_absolute_error(
                y_val[target],
                val_pred[:, j],
            ),

            "Val_NRMSE": (
                val_rmse
                / target_range[target]
            ),
        }
    )


validation_metrics_df = pd.DataFrame(
    validation_metrics
)


print("\nBest XGBoost — Training / Validation:")
print(validation_metrics_df)


validation_metrics_df.to_csv(
    OUTPUT_DIR / "xgb_best_validation_metrics.csv",
    index=False,
)


# ============================================================
# 19. Combine Training + Validation
# ============================================================

X_train_val = pd.concat(
    [
        X_train,
        X_val,
    ],
    axis=0,
)


y_train_val = pd.concat(
    [
        y_train,
        y_val,
    ],
    axis=0,
)


print("\nFinal XGBoost training data:")
print("X_train_val:", X_train_val.shape)
print("y_train_val:", y_train_val.shape)


# ============================================================
# 20. Train final XGBoost
# ============================================================

xgb_final = build_multioutput_xgb(
    best_params
)


xgb_final.fit(
    X_train_val,
    y_train_val,
)


# ============================================================
# 21. Final Test prediction
# ============================================================

y_test_pred = xgb_final.predict(
    X_test
)


# ============================================================
# 22. Overall Test metrics
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
# 23. Extract Test metadata
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
# 24. Define SS-stage Test mask
#
# NRMSE_SS uses ALL Test samples inside:
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
# 25. Calculate NRMSE_SS
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
# 26. Read FactSage equilibrium data
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
# 27. Verify FactSage table
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
# 28. FactSage interpolation
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
# 29. Generate FactSage equilibrium values
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
# 30. Calculate Eeq_SS
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
# 31. Calculate TCR
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
# 32. Combine FINAL XGBoost metrics
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
)


# CH4:
# Eeq_SS = NaN
# TCR    = NaN
# because CH4 is not included in the current FactSage
# equilibrium constraint.

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
    ]
]


print("\n======================================")
print("FINAL XGBOOST TEST METRICS")
print("======================================")
print(final_metrics_df)


# ============================================================
# 33. Save final XGBoost metrics
# ============================================================

final_metrics_df.to_csv(
    OUTPUT_DIR / "xgb_final_metrics.csv",
    index=False,
)


# ============================================================
# 34. Save Test predictions
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
    OUTPUT_DIR / "xgb_test_predictions.csv",
    index=False,
)


# ============================================================
# 35. Save final XGBoost model
# ============================================================

joblib.dump(
    xgb_final,
    OUTPUT_DIR / "xgb_final_model.pkl",
)


# ============================================================
# 36. XGBoost feature importance
#
# MultiOutputRegressor contains four XGBRegressor models:
# estimator 0 -> H2
# estimator 1 -> CO
# estimator 2 -> CO2
# estimator 3 -> CH4
# ============================================================

feature_importance_data = {
    "Feature": FEATURES,
}


for target, estimator in zip(
    TARGETS,
    xgb_final.estimators_,
):
    feature_importance_data[
        f"{target}_Importance"
    ] = estimator.feature_importances_


feature_importance_df = pd.DataFrame(
    feature_importance_data
)


importance_columns = [
    f"{target}_Importance"
    for target in TARGETS
]


feature_importance_df[
    "Mean_Importance"
] = (
    feature_importance_df[
        importance_columns
    ]
    .mean(axis=1)
)


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Mean_Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\nXGBoost Feature Importance:")
print(feature_importance_df)


feature_importance_df.to_csv(
    OUTPUT_DIR / "xgb_feature_importance.csv",
    index=False,
)


# ============================================================
# 37. Save FactSage long table
# ============================================================

factsage_long_df.to_csv(
    OUTPUT_DIR
    / "factsage_equilibrium_H2_CO_CO2_long.csv",
    index=False,
)


# ============================================================
# 38. Save Test subset summary
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
    OUTPUT_DIR / "xgb_test_subset_summary.csv",
    index=False,
)


# ============================================================
# 39. Final summary
# ============================================================

print("\n======================================")
print("XGBoost + TPE WORKFLOW COMPLETED")
print("======================================")


print("\nFixed split:")
print(split_counts)


print("\nBest Mean Validation NRMSE:")
print(best_score)


print("\nBest hyperparameters:")
print(best_params)


print("\nTest subsets:")
print(test_subset_summary_df)


print("\nFinal metrics:")
print(final_metrics_df)


print("\nResults folder:")
print(OUTPUT_DIR)


print("\nSaved files:")


saved_files = [
    "xgb_best_parameters.json",
    "xgb_tpe_search_results.csv",
    "xgb_best_validation_metrics.csv",
    "xgb_final_metrics.csv",
    "xgb_test_predictions.csv",
    "xgb_test_subset_summary.csv",
    "xgb_final_model.pkl",
    "xgb_feature_importance.csv",
    "factsage_equilibrium_H2_CO_CO2_long.csv",
]


for filename in saved_files:
    print(
        OUTPUT_DIR / filename
    )


# In[ ]:




