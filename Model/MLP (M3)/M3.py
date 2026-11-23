#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# MLP + Optuna TPE
# Using ONE predefined fixed Train/Validation/Test split
#
# Final MLP metrics:
#     R2
#     RMSE
#     MAE
#     NRMSE
#     NRMSE_SS
#     Eeq_SS
#     TCR
#     GCR
#
# Important:
#     MLP DOES NOT generate a new split.
#     MLP DOES NOT recalculate D_thermal / stationarity.
#     MLP DOES NOT use physics_weight during training.
#     Baseline MLP DOES NOT use FactSage, TCR, Eeq_SS, or GCR
#     in the loss function or hyperparameter-search objective.
#
#     All split and stationarity information is read from:
#         fixed_split_index.csv
#
#     The same split must be used by:
#         RF
#         XGBoost
#         MLP
#         Proposed model
#
# GCR definition used here:
#     - SS-stage Test samples only
#     - H2 / CO / CO2 only
#     - all 11 temperature inputs are perturbed together
#     - Delta T = 5 degC
#     - Trep = Tmax for FactSage
#     - central finite difference
#     - epsilon_g,j = 0.01 * max_i |g_FS,ij|
#     - only probes with |g_FS,ij| > epsilon_g,j are counted
#     - consistency if g_model,ij * g_FS,ij > 0
# ============================================================

from pathlib import Path
import re
import json
import random
import joblib
import optuna
import numpy as np
import pandas as pd

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
N_TRIALS = 100

MAX_EPOCHS = 1500
EARLY_STOPPING_PATIENCE = 120
EARLY_STOPPING_MIN_DELTA = 1e-6

# GCR settings fixed by the adopted definition.
GCR_DELTA_T = 5.0
GCR_EPS_FRACTION = 0.01


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
#     PINN ver2/MLP/
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
# MLP results folder
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "MLP_TPE_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


optuna.logging.set_verbosity(
    optuna.logging.WARNING
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


# FactSage does not contain CH4 in the present framework.
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


print("\n======================================")
print("Input files")
print("======================================")
print("Experimental data:", DATA_PATH.resolve())
print("Fixed split:", FIXED_SPLIT_PATH.resolve())
print("FactSage:", FACTSAGE_PATH.resolve())
print("Output folder:", OUTPUT_DIR.resolve())


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
# 14. Standardization for MLP hyperparameter optimization
#
# IMPORTANT:
#     During TPE and early stopping, scalers are fitted on TRAIN only.
#     Validation and Test do not contribute to these scalers.
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


# ============================================================
# 15. MLP definition
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
# 16. MLP helper functions
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

    return model.to(
        DEVICE
    )



def make_train_loader(
    X_scaled,
    y_scaled,
    batch_size,
    seed,
):
    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_scaled,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    return loader



def predict_scaled(
    model,
    X_scaled,
):
    model.eval()

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction = model(
            X_tensor
        ).cpu().numpy()

    return prediction



def predict_physical(
    model,
    X_physical,
    x_scaler,
    y_scaler,
):
    if isinstance(
        X_physical,
        pd.DataFrame,
    ):
        X_array = X_physical[
            FEATURES
        ].to_numpy(dtype=float)
    else:
        X_array = np.asarray(
            X_physical,
            dtype=float,
        )

    X_scaled = x_scaler.transform(
        X_array
    )

    prediction_scaled = predict_scaled(
        model,
        X_scaled,
    )

    prediction_physical = y_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction_physical



def calculate_mean_nrmse(
    y_true_physical,
    y_pred_physical,
):
    nrmse_values = []

    for j, target in enumerate(
        TARGETS
    ):
        rmse = np.sqrt(
            mean_squared_error(
                y_true_physical[:, j],
                y_pred_physical[:, j],
            )
        )

        nrmse = (
            rmse
            / float(target_range[target])
        )

        nrmse_values.append(
            nrmse
        )

    return float(
        np.mean(nrmse_values)
    )



def train_with_fixed_validation(
    params,
    X_train_scaled_local,
    y_train_scaled_local,
    X_val_scaled_local,
    y_val_physical_local,
    y_scaler_local,
    seed=RANDOM_STATE,
):
    """
    Train on TRAIN only and use the predefined VALIDATION set
    for early stopping.

    The early-stopping criterion is Mean Validation NRMSE in
    physical units across H2 / CO / CO2 / CH4.
    """

    set_global_seed(
        seed
    )

    model = build_mlp_from_params(
        params
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    criterion = nn.MSELoss()

    train_loader = make_train_loader(
        X_train_scaled_local,
        y_train_scaled_local,
        batch_size=int(params["batch_size"]),
        seed=seed,
    )

    best_state = None
    best_epoch = None
    best_val_mean_nrmse = np.inf
    epochs_without_improvement = 0

    history_records = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            prediction = model(
                X_batch
            )

            loss = criterion(
                prediction,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_n = len(X_batch)

            epoch_loss_sum += (
                float(loss.item())
                * batch_n
            )

            epoch_sample_count += batch_n

        train_loss = (
            epoch_loss_sum
            / epoch_sample_count
        )

        val_prediction_scaled = predict_scaled(
            model,
            X_val_scaled_local,
        )

        val_prediction_physical = y_scaler_local.inverse_transform(
            val_prediction_scaled
        )

        val_mean_nrmse = calculate_mean_nrmse(
            y_true_physical=y_val_physical_local,
            y_pred_physical=val_prediction_physical,
        )

        history_records.append(
            {
                "Epoch": epoch,
                "Train_MSE_scaled": train_loss,
                "Val_Mean_NRMSE": val_mean_nrmse,
            }
        )

        if (
            val_mean_nrmse
            < best_val_mean_nrmse
            - EARLY_STOPPING_MIN_DELTA
        ):
            best_val_mean_nrmse = val_mean_nrmse
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "MLP early stopping failed to save a best model state."
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )

    history_df = pd.DataFrame(
        history_records
    )

    return (
        model,
        int(best_epoch),
        float(best_val_mean_nrmse),
        history_df,
    )



def train_fixed_epochs(
    params,
    X_scaled,
    y_scaled,
    n_epochs,
    seed=RANDOM_STATE,
):
    """
    Final training on TRAIN + VALIDATION for exactly n_epochs.
    No Test information is used.
    """

    set_global_seed(
        seed
    )

    model = build_mlp_from_params(
        params
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    criterion = nn.MSELoss()

    train_loader = make_train_loader(
        X_scaled,
        y_scaled,
        batch_size=int(params["batch_size"]),
        seed=seed,
    )

    history_records = []

    for epoch in range(
        1,
        int(n_epochs) + 1,
    ):
        model.train()

        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            prediction = model(
                X_batch
            )

            loss = criterion(
                prediction,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_n = len(X_batch)

            epoch_loss_sum += (
                float(loss.item())
                * batch_n
            )

            epoch_sample_count += batch_n

        train_loss = (
            epoch_loss_sum
            / epoch_sample_count
        )

        history_records.append(
            {
                "Epoch": epoch,
                "Train_MSE_scaled": train_loss,
            }
        )

    history_df = pd.DataFrame(
        history_records
    )

    return model, history_df


# ============================================================
# 17. TPE objective function
# ============================================================

# Optimization objective:
# Mean Validation NRMSE across H2 / CO / CO2 / CH4
# Lower = better
#
# IMPORTANT:
# Eeq_SS / TCR / GCR are NOT used to select the baseline MLP.
# This prevents thermodynamic information from leaking into the
# baseline model-selection process.
#

def objective(
    trial,
):
    params = {
        "n_hidden_layers": trial.suggest_int(
            "n_hidden_layers",
            1,
            4,
        ),

        "hidden_units": trial.suggest_categorical(
            "hidden_units",
            [
                16,
                32,
                64,
                128,
                256,
            ],
        ),

        "activation": trial.suggest_categorical(
            "activation",
            [
                "relu",
                "tanh",
                "silu",
            ],
        ),

        "dropout": trial.suggest_float(
            "dropout",
            0.0,
            0.30,
            step=0.05,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            5e-3,
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-8,
            1e-3,
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [
                16,
                32,
                64,
            ],
        ),
    }

    (
        model,
        best_epoch,
        best_val_mean_nrmse,
        _,
    ) = train_with_fixed_validation(
        params=params,
        X_train_scaled_local=X_train_scaled,
        y_train_scaled_local=y_train_scaled,
        X_val_scaled_local=X_val_scaled,
        y_val_physical_local=y_val[TARGETS].to_numpy(dtype=float),
        y_scaler_local=y_scaler_search,
        seed=RANDOM_STATE,
    )

    val_pred = predict_physical(
        model=model,
        X_physical=X_val,
        x_scaler=x_scaler_search,
        y_scaler=y_scaler_search,
    )

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

        trial.set_user_attr(
            f"{target}_RMSE",
            float(rmse),
        )

        trial.set_user_attr(
            f"{target}_NRMSE",
            float(nrmse),
        )

    trial.set_user_attr(
        "best_epoch",
        int(best_epoch),
    )

    return float(
        best_val_mean_nrmse
    )


# ============================================================
# 18. Run TPE optimization
# ============================================================

sampler = optuna.samplers.TPESampler(
    seed=RANDOM_STATE
)


study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="MLP_TPE_Optimization",
)


print("\nStarting MLP TPE optimization...")


study.optimize(
    objective,
    n_trials=N_TRIALS,
    gc_after_trial=True,
    show_progress_bar=True,
)


best_score = study.best_value
best_params = study.best_params
best_epoch_from_search = int(
    study.best_trial.user_attrs[
        "best_epoch"
    ]
)


print("\n======================================")
print("TPE optimization completed")
print("======================================")
print("\nBest validation Mean NRMSE:")
print(best_score)
print("\nBest MLP hyperparameters:")


for key, value in best_params.items():
    print(f"{key}: {value}")


print("Best epoch from predefined validation:")
print(best_epoch_from_search)


# ============================================================
# 19. Save TPE results
# ============================================================

best_parameter_payload = dict(
    best_params
)

best_parameter_payload[
    "best_epoch"
] = best_epoch_from_search

best_parameter_payload[
    "max_epochs"
] = MAX_EPOCHS

best_parameter_payload[
    "early_stopping_patience"
] = EARLY_STOPPING_PATIENCE

best_parameter_payload[
    "early_stopping_min_delta"
] = EARLY_STOPPING_MIN_DELTA


with open(
    OUTPUT_DIR / "mlp_best_parameters.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        best_parameter_payload,
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
        "Best_Epoch": trial.user_attrs.get(
            "best_epoch"
        ),
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
    OUTPUT_DIR / "mlp_tpe_search_results.csv",
    index=False,
)


# ============================================================
# 20. Rebuild best MLP on Training only with fixed Validation
# ============================================================

(
    best_mlp_validation,
    validation_best_epoch,
    validation_best_score,
    validation_history_df,
) = train_with_fixed_validation(
    params=best_params,
    X_train_scaled_local=X_train_scaled,
    y_train_scaled_local=y_train_scaled,
    X_val_scaled_local=X_val_scaled,
    y_val_physical_local=y_val[TARGETS].to_numpy(dtype=float),
    y_scaler_local=y_scaler_search,
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


validation_history_df.to_csv(
    OUTPUT_DIR / "mlp_best_validation_history.csv",
    index=False,
)


# ============================================================
# 21. Training and Validation diagnostics
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


print("\nBest MLP — Training / Validation:")
print(validation_metrics_df)


print("\nBest validation epoch used for final retraining:")
print(validation_best_epoch)


print("\nBest validation Mean NRMSE after rebuild:")
print(validation_best_score)


validation_metrics_df.to_csv(
    OUTPUT_DIR / "mlp_best_validation_metrics.csv",
    index=False,
)


# ============================================================
# 22. Combine Training + Validation
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


print("\nFinal MLP training data:")
print("X_train_val:", X_train_val.shape)
print("y_train_val:", y_train_val.shape)


# ============================================================
# 23. Refit scalers on Training + Validation
#
# The hyperparameters and epoch count have already been selected
# using the predefined Validation set. Test remains untouched.
# ============================================================

x_scaler_final = StandardScaler()
y_scaler_final = StandardScaler()


X_train_val_scaled = x_scaler_final.fit_transform(
    X_train_val[FEATURES].to_numpy(dtype=float)
)


y_train_val_scaled = y_scaler_final.fit_transform(
    y_train_val[TARGETS].to_numpy(dtype=float)
)


# ============================================================
# 24. Train final MLP on Training + Validation
#
# We use the best epoch selected by the winning Optuna trial on
# the predefined Validation set. No early stopping is performed
# here because Validation has now been merged into final training
# data.
# ============================================================

if validation_best_epoch != best_epoch_from_search:
    print(
        "\nWarning: rebuilt validation run selected epoch "
        f"{validation_best_epoch}, while the winning Optuna trial "
        f"stored epoch {best_epoch_from_search}. "
        "The final model will use the Optuna-selected epoch."
    )

final_epochs = int(
    best_epoch_from_search
)

mlp_final, final_training_history_df = train_fixed_epochs(
    params=best_params,
    X_scaled=X_train_val_scaled,
    y_scaled=y_train_val_scaled,
    n_epochs=final_epochs,
    seed=RANDOM_STATE,
)


final_training_history_df.to_csv(
    OUTPUT_DIR / "mlp_final_training_history.csv",
    index=False,
)


# ============================================================
# 25. Final Test prediction
# ============================================================

y_test_pred = predict_physical(
    model=mlp_final,
    X_physical=X_test,
    x_scaler=x_scaler_final,
    y_scaler=y_scaler_final,
)


# ============================================================
# 26. Overall Test metrics
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
# 27. Extract Test metadata
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
# 28. Define SS-stage Test mask
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
# 29. Calculate NRMSE_SS
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
# 30. Read FactSage equilibrium data
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
# 31. Verify FactSage table
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
# 32. FactSage interpolation
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
# 33. Generate FactSage equilibrium values
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
# 34. Calculate Eeq_SS
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
# 35. Calculate TCR
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
# 36. Calculate GCR
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
# 36.1 Build physical-space +5 / -5 temperature perturbations
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
    model=mlp_final,
    X_physical=X_ss_plus,
    x_scaler=x_scaler_final,
    y_scaler=y_scaler_final,
)


model_prediction_minus = predict_physical(
    model=mlp_final,
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
# 36.2 FactSage +5 / -5 central finite-difference gradients
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
# 36.3 Probe-level GCR details
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
# 37. Combine FINAL MLP metrics
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
print("FINAL MLP TEST METRICS")
print("======================================")
print(final_metrics_df)


print("\nMean GCR across H2 / CO / CO2:")
print(mean_gcr)


# ============================================================
# 38. Save final MLP metrics
# ============================================================

final_metrics_df.to_csv(
    OUTPUT_DIR / "mlp_final_metrics.csv",
    index=False,
)


gcr_summary_df.to_csv(
    OUTPUT_DIR / "mlp_gcr_summary.csv",
    index=False,
)


gcr_probe_details_df.to_csv(
    OUTPUT_DIR / "mlp_gcr_probe_details.csv",
    index=False,
)


# ============================================================
# 39. Save Test predictions
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
    OUTPUT_DIR / "mlp_test_predictions.csv",
    index=False,
)


# ============================================================
# 40. Save final MLP model and scalers
# ============================================================

model_checkpoint = {
    "model_state_dict": {
        key: value.detach().cpu()
        for key, value in mlp_final.state_dict().items()
    },
    "features": FEATURES,
    "targets": TARGETS,
    "best_params": best_params,
    "final_epochs": int(final_epochs),
    "random_state": RANDOM_STATE,
    "gcr_delta_T": GCR_DELTA_T,
    "gcr_eps_fraction": GCR_EPS_FRACTION,
}


torch.save(
    model_checkpoint,
    OUTPUT_DIR / "mlp_final_model.pt",
)


joblib.dump(
    x_scaler_final,
    OUTPUT_DIR / "mlp_x_scaler.pkl",
)


joblib.dump(
    y_scaler_final,
    OUTPUT_DIR / "mlp_y_scaler.pkl",
)


# ============================================================
# 41. Save FactSage long table
# ============================================================

factsage_long_df.to_csv(
    OUTPUT_DIR
    / "factsage_equilibrium_H2_CO_CO2_long.csv",
    index=False,
)


# ============================================================
# 42. Save Test subset summary
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
    OUTPUT_DIR / "mlp_test_subset_summary.csv",
    index=False,
)


# ============================================================
# 43. Save compact run configuration
# ============================================================

run_configuration = {
    "random_state": RANDOM_STATE,
    "device": str(DEVICE),
    "n_trials": N_TRIALS,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    "best_validation_mean_nrmse": float(best_score),
    "best_validation_epoch": int(best_epoch_from_search),
    "final_training_samples": int(len(X_train_val)),
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
    OUTPUT_DIR / "mlp_run_configuration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_configuration,
        file,
        indent=4,
    )


# ============================================================
# 44. Final summary
# ============================================================

print("\n======================================")
print("MLP + TPE + GCR WORKFLOW COMPLETED")
print("======================================")


print("\nFixed split:")
print(split_counts)


print("\nBest Mean Validation NRMSE:")
print(best_score)


print("\nBest hyperparameters:")
print(best_params)


print("\nBest validation epoch / final epochs:")
print(final_epochs)


print("\nTest subsets:")
print(test_subset_summary_df)


print("\nFinal metrics:")
print(final_metrics_df)


print("\nGCR summary:")
print(gcr_summary_df)


print("\nMean GCR:")
print(mean_gcr)


print("\nResults folder:")
print(OUTPUT_DIR)


print("\nSaved files:")


saved_files = [
    "mlp_best_parameters.json",
    "mlp_tpe_search_results.csv",
    "mlp_best_validation_history.csv",
    "mlp_best_validation_metrics.csv",
    "mlp_final_training_history.csv",
    "mlp_final_metrics.csv",
    "mlp_test_predictions.csv",
    "mlp_test_subset_summary.csv",
    "mlp_gcr_summary.csv",
    "mlp_gcr_probe_details.csv",
    "mlp_final_model.pt",
    "mlp_x_scaler.pkl",
    "mlp_y_scaler.pkl",
    "mlp_run_configuration.json",
    "factsage_equilibrium_H2_CO_CO2_long.csv",
]


for filename in saved_files:
    print(
        OUTPUT_DIR / filename
    )


# In[2]:


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

MODEL = "M3"


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




