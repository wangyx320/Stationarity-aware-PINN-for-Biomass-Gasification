#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# M6 Proposed
# MLP-Eq-Grad + Stationarity Weighting
#
# Part 1:
#   - Basic settings
#   - Load M5 parameters
#   - Load experimental data
#   - Load fixed split
#   - Build stationarity weights
#
# ============================================================


from pathlib import Path
import json
import random
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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

MAX_EPOCHS = 1500


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility (same seed settings as M5)
# ============================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve reproducibility; exact determinism may depend on backend.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_STATE)


print("======================================")
print("PyTorch device")
print("======================================")
print(DEVICE)



# ============================================================
# 2. Fixed M5 parameters
# ============================================================


# ------------------------------------------------------------
# M5 result folder
# ------------------------------------------------------------

M5_DIR = Path(
    "../M5 MLP-Eq-Grad/MLP_Eq_Grad_Pareto_results"
)


M5_PARAMETER_PATH = (
    M5_DIR
    /
    "mlp_eq_grad_best_parameters.json"
)


if not M5_PARAMETER_PATH.exists():

    raise FileNotFoundError(
        f"""
Cannot find M5 parameter file:

{M5_PARAMETER_PATH}

Please confirm M5 final selection file.
"""
    )



with open(
    M5_PARAMETER_PATH,
    "r",
    encoding="utf-8"
) as f:

    M5_PARAMS = json.load(f)



# ============================================================
# Fixed M5 physics coefficients
# ============================================================

FIXED_LAMBDA_EQ = 0.707945784384138

FIXED_LAMBDA_GRAD = 0.1412537544622755



print("\n======================================")
print("M5 fixed parameters")
print("======================================")


print(
    "lambda_eq:",
    FIXED_LAMBDA_EQ
)


print(
    "lambda_grad:",
    FIXED_LAMBDA_GRAD
)



# ------------------------------------------------------------
# MLP hyperparameters inherited from M5
# ------------------------------------------------------------


MLP_KEYS = [
    "n_hidden_layers",
    "hidden_units",
    "activation",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
]


BASE_MLP_PARAMS = {

    key:
    M5_PARAMS[key]

    for key in MLP_KEYS

}



print("\nMLP parameters:")

for k,v in BASE_MLP_PARAMS.items():

    print(
        f"{k}: {v}"
    )





# ============================================================
# 3. Experimental data
# ============================================================


DATA_PATH = Path(
    "../20250606_oxygen enrich_whole process_linear interpolation.csv"
)



if not DATA_PATH.exists():

    raise FileNotFoundError(
        DATA_PATH
    )



df = pd.read_csv(
    DATA_PATH
).copy()



# preserve original index

df["original_row"] = np.arange(
    len(df)
)



print("\n======================================")
print("Experimental data")
print("======================================")

print(
    df.shape
)



# ============================================================
# 4. Feature and target definition
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



FACTSAGE_TARGETS = [

    "H2",
    "CO",
    "CO2",

]



# ============================================================
# 5. Load fixed split
# ============================================================


FIXED_SPLIT_PATH = Path(
    "../fixed_split_results/fixed_split_index.csv"
)



if not FIXED_SPLIT_PATH.exists():

    raise FileNotFoundError(
        FIXED_SPLIT_PATH
    )



split_df = pd.read_csv(
    FIXED_SPLIT_PATH
)



print("\n======================================")
print("Fixed split")
print("======================================")


print(
    split_df[
        "split"
    ].value_counts()
)





# ============================================================
# 6. Merge fixed split information
# ============================================================


merge_columns = [

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
            merge_columns
        ],

        on="original_row",

        how="left",

        validate="one_to_one"

    )

    .sort_values(
        "original_row"
    )

    .reset_index(
        drop=True
    )

)



# ============================================================
# 7. Representative temperature
# ============================================================


# Trep = Tmax

df["Trep"] = (

    df[TEMP_COLS]

    .max(axis=1)

)



df["Tmax_sensor"] = (

    df[TEMP_COLS]

    .idxmax(axis=1)

)





# ============================================================
# 8. Construct M6 stationarity weighting
# ============================================================


# ------------------------------------------------------------
# Definition:
#
# wi = ISS * Si
#
# Si = 1/(1+(Di/Dcrit)^p)
#
# Di = D_thermal from fixed_split_results
#
# ------------------------------------------------------------


D_CRIT = 1.0

P_POWER = 2



# ============================================================
# Fix NaN D_thermal
# ============================================================


D_values = (
    df["D_thermal"]
    .fillna(
        np.inf
    )
)


stationarity_factor = (

    1.0

    /

    (

        1.0

        +

        (

            D_values

            /

            D_CRIT

        )

        ** P_POWER

    )

)



# steady-state indicator

SS_indicator = (

    df["process_stage"]

    .eq(
        "ss_stage"
    )

    .astype(float)

)



# final physics weight

df["M6_physics_weight"] = (

    SS_indicator

    *

    stationarity_factor

)



print("\n======================================")
print("M6 physics weight")
print("======================================")


print(
    df[
        [
            "process_stage",
            "D_thermal",
            "M6_physics_weight"
        ]
    ]
    .head()
)



print("\nWeight statistics:")


print(
    df[
        "M6_physics_weight"
    ]
    .describe()
)





# ============================================================
# 9. Split index
# ============================================================


train_idx = (

    df.index[
        df["split"]
        ==
        "train"
    ]

    .to_numpy()

)



val_idx = (

    df.index[
        df["split"]
        ==
        "validation"
    ]

    .to_numpy()

)



test_idx = (

    df.index[
        df["split"]
        ==
        "test"
    ]

    .to_numpy()

)



print("\n======================================")
print("Dataset split")
print("======================================")


print(
    "Train:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)

print(
    "Test:",
    len(test_idx)
)



# ============================================================
# END OF PART 1
# ============================================================


# In[2]:


# ============================================================
# M6 Proposed
#
# Part 2:
#   - FactSage parser
#   - Eq / Gradient targets
#   - Scaling
#   - Weighted physics loss
#   - MLP definition
#
# ============================================================


import re



# ============================================================
# 10. FactSage equilibrium data
# ============================================================


FACTSAGE_PATH = Path(
    "../Fig_FactSage_data.xlsx"
)



if not FACTSAGE_PATH.exists():

    raise FileNotFoundError(
        FACTSAGE_PATH
    )



factsage_raw_df = pd.read_excel(
    FACTSAGE_PATH,
    sheet_name=0,
    header=None
)



# ------------------------------------------------------------
# normalize gas names
# ------------------------------------------------------------


def normalize_factsage_gas_name(value):

    name = (

        str(value)

        .strip()

        .replace("₂","2")

        .replace("₄","4")

        .replace(" ","")

        .upper()

    )


    mapping = {

        "H2":"H2",

        "CO":"CO",

        "CO2":"CO2",

    }


    return mapping.get(name)





# ------------------------------------------------------------
# Parse FactSage table
# ------------------------------------------------------------


factsage_records = []

row_index = 0



while row_index < len(factsage_raw_df):


    first_cell = factsage_raw_df.iat[
        row_index,
        0
    ]


    oxygen_match = re.match(

        r"^\s*(\d+(?:\.\d+)?)\s*%\s*O2\s*$",

        str(first_cell),

        flags=re.IGNORECASE

    )



    if oxygen_match is None:

        row_index += 1

        continue



    oxygen_value = float(
        oxygen_match.group(1)
    )



    temperatures = pd.to_numeric(

        factsage_raw_df.iloc[
            row_index,
            1:
        ],

        errors="coerce"

    )


    valid_temperature = temperatures.notna()


    T_grid = (

        temperatures

        .loc[valid_temperature]

        .to_numpy(dtype=float)

    )



    gas_curves = {}



    gas_row = row_index + 1



    while gas_row < len(factsage_raw_df):


        gas_label = factsage_raw_df.iat[
            gas_row,
            0
        ]


        if pd.isna(gas_label):

            break



        next_o2 = re.match(

            r"^\s*(\d+(?:\.\d+)?)\s*%\s*O2\s*$",

            str(gas_label),

            flags=re.IGNORECASE

        )


        if next_o2 is not None:

            break



        gas_name = normalize_factsage_gas_name(
            gas_label
        )



        if gas_name is not None:


            values = pd.to_numeric(

                factsage_raw_df.iloc[
                    gas_row,
                    1:
                ],

                errors="coerce"

            )

            values = (

                values

                .loc[valid_temperature]

                .to_numpy(dtype=float)

            )


            gas_curves[gas_name] = values



        gas_row += 1



    for idx,T in enumerate(T_grid):


        record = {

            "Oxygen concentration":
            oxygen_value,

            "Temperature":
            float(T),

        }


        for target in FACTSAGE_TARGETS:


            record[
                f"{target}_eq"
            ] = float(
                gas_curves[target][idx]
            )


        factsage_records.append(
            record
        )



    row_index = gas_row





factsage_long_df = pd.DataFrame(
    factsage_records
)



factsage_long_df = (

    factsage_long_df

    .sort_values(
        [
            "Oxygen concentration",
            "Temperature"
        ]
    )

    .reset_index(
        drop=True
    )

)



print("\n======================================")
print("FactSage summary")
print("======================================")


print(
    factsage_long_df.groupby(
        "Oxygen concentration"
    )
    ["Temperature"]
    .agg(
        [
            "count",
            "min",
            "max"
        ]
    )
)





# ============================================================
# 11. FactSage interpolation
# ============================================================


def interpolate_factsage_values(
        oxygen_values,
        temperature_values,
        target
):


    output = np.zeros(
        len(temperature_values)
    )


    oxygen_values = np.asarray(
        oxygen_values
    )


    temperature_values = np.asarray(
        temperature_values
    )



    for o2 in np.unique(
        oxygen_values
    ):


        mask = np.isclose(
            oxygen_values,
            o2
        )


        curve = (

            factsage_long_df

            .loc[
                np.isclose(
                    factsage_long_df[
                        "Oxygen concentration"
                    ],
                    o2
                )
            ]

            .sort_values(
                "Temperature"
            )

        )



        output[mask] = np.interp(

            temperature_values[mask],

            curve[
                "Temperature"
            ],

            curve[
                f"{target}_eq"
            ]

        )



    return output





# ============================================================
# 12. Build Eq and Gradient targets
# ============================================================



def build_physics_targets(
        metadata,
        delta_T=5.0
):


    n = len(metadata)



    eq_target = np.zeros(
        (n,3)
    )


    grad_target = np.zeros(
        (n,3)
    )


    physics_mask = (

        metadata[
            "process_stage"
        ]

        .eq(
            "ss_stage"
        )

        .to_numpy()

    )



    if physics_mask.any():


        oxygen = (

            metadata

            .loc[
                physics_mask,
                "Oxygen concentration"
            ]

            .to_numpy()

        )



        temperature = (

            metadata

            .loc[
                physics_mask,
                "Trep"
            ]

            .to_numpy()

        )



        for k,target in enumerate(
            FACTSAGE_TARGETS
        ):


            eq_target[
                physics_mask,
                k
            ] = interpolate_factsage_values(

                oxygen,

                temperature,

                target

            )



            plus = interpolate_factsage_values(

                oxygen,

                temperature + delta_T,

                target

            )


            minus = interpolate_factsage_values(

                oxygen,

                temperature - delta_T,

                target

            )



            grad_target[
                physics_mask,
                k
            ] = (

                plus-minus

            )/(2*delta_T)



    return (
        eq_target,
        grad_target,
        physics_mask
    )





(
    eq_train_physical,
    grad_train_physical,
    physics_train_mask
)=build_physics_targets(
    df.loc[
        train_idx
    ]
)



(
    eq_val_physical,
    grad_val_physical,
    physics_val_mask
)=build_physics_targets(
    df.loc[
        val_idx
    ]
)



print("\nPhysics samples:")

print(
    physics_train_mask.sum()
)





# ============================================================
# 13. Scaling
# ============================================================


X_train = df.loc[
    train_idx,
    FEATURES
]


y_train = df.loc[
    train_idx,
    TARGETS
]


X_val = df.loc[
    val_idx,
    FEATURES
]


y_val = df.loc[
    val_idx,
    TARGETS
]



X_test = df.loc[
    test_idx,
    FEATURES
]


y_test = df.loc[
    test_idx,
    TARGETS
]




x_scaler = StandardScaler()


y_scaler = StandardScaler()



X_train_scaled = x_scaler.fit_transform(
    X_train
)


X_val_scaled = x_scaler.transform(
    X_val
)


X_test_scaled = x_scaler.transform(
    X_test
)




y_train_scaled = y_scaler.fit_transform(
    y_train
)



y_val_scaled = y_scaler.transform(
    y_val
)



target_range = (

    y_train.max()

    -

    y_train.min()

).to_numpy(dtype=float)

# The original TRAIN-only range is fixed for validation and ALL test metrics.
# Do not recalculate after merging TRAIN and VALIDATION.
target_range_final = target_range.copy()

if (not np.isfinite(target_range_final).all()) or np.any(target_range_final <= 0):
    raise ValueError("Invalid TRAIN target range for NRMSE normalization")





# ============================================================
# 14. Weighted MSE
# ============================================================



def weighted_mse(prediction, target, weight):
    """Absolute sample-weighted MSE on H2, CO, CO2 (SS samples only).

    prediction, target: tensors of shape [N_SS_in_batch, 3]
    weight: tensor of shape [N_SS_in_batch]

    M5 equivalence: all weights = 1 -> ordinary nn.MSELoss().
    Weights reduce the absolute physics-loss contribution; they are NOT
    normalized by their sum. Lambda_eq / lambda_grad stay fixed.
    """
    weight = weight.reshape(-1, 1)
    squared_error = (prediction - target) ** 2
    return (weight * squared_error).mean()



# ============================================================
# 15. MLP definition
# ============================================================


def get_activation(name):


    if name=="relu":

        return nn.ReLU()


    if name=="tanh":

        return nn.Tanh()


    if name=="silu":

        return nn.SiLU()


    raise ValueError(name)





class MultiOutputMLP(nn.Module):


    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units,
        n_hidden_layers,
        activation,
        dropout
    ):


        super().__init__()



        layers=[]


        current=input_dim



        for _ in range(
            n_hidden_layers
        ):


            layers.append(

                nn.Linear(
                    current,
                    hidden_units
                )

            )


            layers.append(
                get_activation(
                    activation
                )
            )


            if dropout>0:

                layers.append(
                    nn.Dropout(
                        dropout
                    )
                )


            current=hidden_units



        layers.append(

            nn.Linear(
                current,
                output_dim
            )

        )



        self.network=nn.Sequential(
            *layers
        )



    def forward(
        self,x
    ):

        return self.network(x)





print("\n======================================")
print("M6 Part 2 completed")
print("======================================")


# In[3]:


# ============================================================
# M6 Proposed
#
# Part 3:
#   - DataLoader
#   - Gradient calculation
#   - M6 training
#   - Validation epoch selection
#
# ============================================================


# ============================================================
# 16. Prepare tensors
# ============================================================


# training physics weight

train_weights = (

    df.loc[
        train_idx,
        "M6_physics_weight"
    ]

    .to_numpy()

)



val_weights = (

    df.loc[
        val_idx,
        "M6_physics_weight"
    ]

    .to_numpy()

)



# ------------------------------------------------------------
# Scale equilibrium target
# ------------------------------------------------------------

eq_train_scaled = np.zeros_like(
    eq_train_physical
)



eq_train_scaled[
    physics_train_mask
] = (

    eq_train_physical[
        physics_train_mask
    ]

    -

    y_scaler.mean_[:3]

) / y_scaler.scale_[:3]





# gradient target remains physical

grad_train_target = np.nan_to_num(
    grad_train_physical
)





# ============================================================
# 17. DataLoader
# ============================================================


from torch.utils.data import (
    TensorDataset,
    DataLoader
)



def create_loader(
        X,
        y,
        eq,
        grad,
        physics_mask,
        weights,
        batch_size,
        shuffle=True,
        seed=RANDOM_STATE
):


    dataset = TensorDataset(

        torch.tensor(
            X,
            dtype=torch.float32
        ),

        torch.tensor(
            y,
            dtype=torch.float32
        ),

        torch.tensor(
            eq,
            dtype=torch.float32
        ),

        torch.tensor(
            grad,
            dtype=torch.float32
        ),

        torch.tensor(
            physics_mask,
            dtype=torch.bool
        ),

        torch.tensor(
            weights,
            dtype=torch.float32
        )

    )


    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        generator=generator,

        drop_last=False

    )





train_loader = create_loader(

    X_train_scaled,

    y_train_scaled,

    eq_train_scaled,

    grad_train_target,

    physics_train_mask,

    train_weights,

    BASE_MLP_PARAMS["batch_size"]

)





# ============================================================
# 18. Model builder
# ============================================================


def build_model():


    model = MultiOutputMLP(

        input_dim=len(FEATURES),

        output_dim=len(TARGETS),

        hidden_units=int(
            BASE_MLP_PARAMS["hidden_units"]
        ),

        n_hidden_layers=int(
            BASE_MLP_PARAMS["n_hidden_layers"]
        ),

        activation=
            BASE_MLP_PARAMS["activation"],

        dropout=float(
            BASE_MLP_PARAMS["dropout"]
        )

    )


    return model.to(
        DEVICE
    )





# ============================================================
# 19. Temperature gradient calculation
# ============================================================


GRAD_DELTA_T = 5.0



TEMP_POSITIONS = [

    FEATURES.index(c)

    for c in TEMP_COLS

]





def temperature_shift_vector(
        scaler,
        delta_T
):


    shift = np.zeros(
        len(FEATURES)
    )



    for p in TEMP_POSITIONS:


        shift[p] = (

            delta_T

            /

            scaler.scale_[p]

        )


    return shift





def get_gradient_shift(
        scaler
):


    return torch.tensor(

        temperature_shift_vector(

            scaler,

            GRAD_DELTA_T

        ),

        dtype=torch.float32,

        device=DEVICE

    ).reshape(
        1,-1
    )





def model_gradient_physical(
        model,
        X_scaled,
        gradient_shift,
        y_scaler_used
):


    pred_plus = model(

        X_scaled + gradient_shift

    )


    pred_minus = model(

        X_scaled - gradient_shift

    )



    grad_scaled = (

        pred_plus

        -

        pred_minus

    ) / (

        2*GRAD_DELTA_T

    )



    y_scale = torch.tensor(

        y_scaler_used.scale_,

        dtype=torch.float32,

        device=DEVICE

    )



    grad_physical = (

        grad_scaled

        *

        y_scale

    )



    return grad_physical





# ============================================================
# 20. Validation NRMSE
# ============================================================


def calculate_val_nrmse(
        model
):


    model.eval()


    with torch.no_grad():


        pred = model(

            torch.tensor(

                X_val_scaled,

                dtype=torch.float32,

                device=DEVICE

            )

        )



    pred = pred.cpu().numpy()



    pred_physical = (

        y_scaler

        .inverse_transform(
            pred
        )

    )



    true = y_val.to_numpy()



    values=[]



    for j in range(
        len(TARGETS)
    ):


        rmse = np.sqrt(

            mean_squared_error(

                true[:,j],

                pred_physical[:,j]

            )

        )



        values.append(

            rmse

            /

            target_range[j]

        )



    return float(
        np.mean(values)
    )





# ============================================================
# 21. M6 training function
# ============================================================


def train_M6(
        epochs=MAX_EPOCHS
):


    set_global_seed(RANDOM_STATE)

    model = build_model()



    # gradient shift based on training scaler

    gradient_shift_train = get_gradient_shift(

        x_scaler

    )



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=float(

            BASE_MLP_PARAMS[
                "learning_rate"
            ]

        ),

        weight_decay=float(

            BASE_MLP_PARAMS[
                "weight_decay"
            ]

        )

    )



    history=[]



    criterion = nn.MSELoss()



    for epoch in range(

        1,

        epochs+1

    ):



        model.train()



        epoch_loss=[]



        for (

            Xb,

            yb,

            eqb,

            gradb,

            maskb,

            wb

        ) in train_loader:



            Xb = Xb.to(DEVICE)

            yb = yb.to(DEVICE)

            eqb = eqb.to(DEVICE)

            gradb = gradb.to(DEVICE)

            maskb = maskb.to(DEVICE)

            wb = wb.to(DEVICE)



            optimizer.zero_grad()



            pred = model(
                Xb
            )



            # -----------------------------
            # data loss
            # -----------------------------

            loss_data = criterion(

                pred,

                yb

            )



            loss_eq = torch.zeros(

                (),

                device=DEVICE

            )


            loss_grad = torch.zeros(

                (),

                device=DEVICE

            )



            if maskb.sum() > 0:



                pred_eq = (

                    pred[

                        maskb,

                        :3

                    ]

                )



                eq_target = (

                    eqb[

                        maskb

                    ]

                )



                weights = (

                    wb[

                        maskb

                    ]

                )



                # weighted equilibrium loss

                loss_eq = weighted_mse(

                    pred_eq,

                    eq_target,

                    weights

                )



                # weighted gradient loss

                grad_pred = (

                    model_gradient_physical(

                        model,

                        Xb[maskb],

                        gradient_shift_train,

                        y_scaler

                    )

                    [:,:3]

                )



                loss_grad = weighted_mse(

                    grad_pred,

                    gradb[maskb],

                    weights

                )





            loss = (

                loss_data

                +

                FIXED_LAMBDA_EQ

                *

                loss_eq

                +

                FIXED_LAMBDA_GRAD

                *

                loss_grad

            )



            loss.backward()



            optimizer.step()



            epoch_loss.append(

                float(
                    loss.item()
                )

            )





        val_nrmse = calculate_val_nrmse(
            model
        )



        history.append(

            {

                "Epoch":

                epoch,


                "Train_Loss":

                np.mean(epoch_loss),


                "Val_Mean_NRMSE":

                val_nrmse

            }

        )



        if epoch % 100 == 0:


            print(

                f"Epoch {epoch}, "

                f"Loss={np.mean(epoch_loss):.6f}, "

                f"Val NRMSE={val_nrmse:.6f}"

            )





    history = pd.DataFrame(
        history
    )



    return model, history





# ============================================================
# 22. Run validation training
# ============================================================


m6_model_search, m6_history = train_M6()



best_epoch = int(

    m6_history.loc[

        m6_history[
            "Val_Mean_NRMSE"
        ]

        .idxmin(),

        "Epoch"

    ]

)



print("\n======================================")
print("M6 selected epoch")
print("======================================")


print(
    best_epoch
)


# ============================================================
# END OF PART 3
# ============================================================


# In[4]:


# ============================================================
# M6 Proposed
#
# Part 4:
#   - Final retraining with Train + Validation
#   - Test inference
#   - Prediction metrics
#   - Save results
#
# ============================================================


from pathlib import Path
import json
import joblib



# ============================================================
# 23. Output folder
# ============================================================


OUTPUT_DIR = Path(
    "./MLP_Eq_Grad_Stationarity_results"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# save epoch search history

m6_history.to_csv(

    OUTPUT_DIR /
    "m6_training_history.csv",

    index=False

)



# ============================================================
# 24. Combine Train + Validation
# ============================================================


final_train_idx = np.concatenate(
    [
        train_idx,
        val_idx
    ]
)



X_final = df.loc[
    final_train_idx,
    FEATURES
]


y_final = df.loc[
    final_train_idx,
    TARGETS
]



X_test = df.loc[
    test_idx,
    FEATURES
]


y_test = df.loc[
    test_idx,
    TARGETS
]





# ============================================================
# 25. Final scalers
# ============================================================


x_scaler_final = StandardScaler()

y_scaler_final = StandardScaler()



X_final_scaled = (

    x_scaler_final

    .fit_transform(
        X_final
    )

)



X_test_scaled_final = (

    x_scaler_final

    .transform(
        X_test
    )

)



y_final_scaled = (

    y_scaler_final

    .fit_transform(
        y_final
    )

)






# ============================================================
# 26. Final physics targets
# ============================================================



(
    eq_final,
    grad_final,
    physics_final_mask

)=build_physics_targets(

    df.loc[
        final_train_idx
    ]

)





eq_final_scaled = np.zeros_like(
    eq_final
)



eq_final_scaled[
    physics_final_mask
] = (

    eq_final[
        physics_final_mask
    ]

    -

    y_scaler_final.mean_[:3]

) / y_scaler_final.scale_[:3]



grad_final = np.nan_to_num(
    grad_final
)



final_weights = (

    df.loc[
        final_train_idx,
        "M6_physics_weight"
    ]

    .to_numpy()

)





# ============================================================
# 27. Final DataLoader
# ============================================================


final_loader = create_loader(

    X_final_scaled,

    y_final_scaled,

    eq_final_scaled,

    grad_final,

    physics_final_mask,

    final_weights,

    BASE_MLP_PARAMS["batch_size"]

)





# ============================================================
# 28. Final model
# ============================================================


set_global_seed(RANDOM_STATE)

final_model = build_model()



gradient_shift_final = get_gradient_shift(

    x_scaler_final

)



optimizer = torch.optim.Adam(

    final_model.parameters(),

    lr=float(
        BASE_MLP_PARAMS[
            "learning_rate"
        ]
    ),

    weight_decay=float(
        BASE_MLP_PARAMS[
            "weight_decay"
        ]
    )

)



criterion = nn.MSELoss()



print("\n======================================")
print("Final M6 retraining")
print("======================================")


print(
    "Epoch:",
    best_epoch
)





# ============================================================
# 29. Final training
# ============================================================


for epoch in range(

    1,

    best_epoch + 1

):


    final_model.train()


    epoch_losses=[]



    for (

        Xb,

        yb,

        eqb,

        gradb,

        maskb,

        wb

    ) in final_loader:


        Xb=Xb.to(DEVICE)

        yb=yb.to(DEVICE)

        eqb=eqb.to(DEVICE)

        gradb=gradb.to(DEVICE)

        maskb=maskb.to(DEVICE)

        wb=wb.to(DEVICE)



        optimizer.zero_grad()



        pred = final_model(
            Xb
        )



        loss_data = criterion(

            pred,

            yb

        )



        loss_eq=torch.zeros(

            (),

            device=DEVICE

        )


        loss_grad=torch.zeros(

            (),

            device=DEVICE

        )



        if maskb.sum()>0:



            weights = wb[maskb]



            loss_eq = weighted_mse(

                pred[
                    maskb,
                    :3
                ],

                eqb[
                    maskb
                ],

                weights

            )



            grad_pred=(

                model_gradient_physical(

                    final_model,

                    Xb[maskb],

                    gradient_shift_final,

                    y_scaler_final

                )

                [:,:3]

            )



            loss_grad = weighted_mse(

                grad_pred,

                gradb[
                    maskb
                ],

                weights

            )



        loss=(

            loss_data

            +

            FIXED_LAMBDA_EQ
            *
            loss_eq

            +

            FIXED_LAMBDA_GRAD
            *
            loss_grad

        )



        loss.backward()


        optimizer.step()



        epoch_losses.append(

            loss.item()

        )



    if epoch % 100 == 0:


        print(

            f"Epoch {epoch}, "

            f"Loss={np.mean(epoch_losses):.6f}"

        )





print(
    "Final retraining completed"
)





# ============================================================
# 30. Save model
# ============================================================


torch.save(

    final_model.state_dict(),

    OUTPUT_DIR /
    "m6_final_model.pt"

)



joblib.dump(

    x_scaler_final,

    OUTPUT_DIR /
    "m6_x_scaler.pkl"

)


joblib.dump(

    y_scaler_final,

    OUTPUT_DIR /
    "m6_y_scaler.pkl"

)





# ============================================================
# 31. Test inference
# ============================================================


final_model.eval()



with torch.no_grad():


    pred_scaled = final_model(

        torch.tensor(

            X_test_scaled_final,

            dtype=torch.float32,

            device=DEVICE

        )

    )



pred_scaled = (

    pred_scaled

    .cpu()

    .numpy()

)



pred_test = (

    y_scaler_final

    .inverse_transform(

        pred_scaled

    )

)



true_test = y_test.to_numpy()





# ============================================================
# 32. Save prediction
# ============================================================


prediction_df = pd.DataFrame()



for i,target in enumerate(TARGETS):


    prediction_df[

        f"{target}_true"

    ] = true_test[:,i]


    prediction_df[

        f"{target}_pred"

    ] = pred_test[:,i]



prediction_df.to_csv(

    OUTPUT_DIR /
    "m6_test_predictions.csv",

    index=False

)





# ============================================================
# 33. Basic metrics
# ============================================================


metrics=[]



for i,target in enumerate(TARGETS):


    rmse=np.sqrt(

        mean_squared_error(

            true_test[:,i],

            pred_test[:,i]

        )

    )


    metrics.append(

        {


        "Target":

        target,


        "R2":

        r2_score(

            true_test[:,i],

            pred_test[:,i]

        ),


        "RMSE":

        rmse,


        "MAE":

        mean_absolute_error(

            true_test[:,i],

            pred_test[:,i]

        ),


        "NRMSE":

        rmse /

        target_range_final[i]


        }

    )



metrics_df=pd.DataFrame(
    metrics
)





# ============================================================
# 34. Stage-based NRMSE
# ============================================================


test_meta=df.loc[
    test_idx
].copy()



stage_masks={


"NRMSE_SS":

test_meta[
    "process_stage"
]
==
"ss_stage",



"NRMSE_LocalSteady_SS":

test_meta[
    "is_local_steady"
]
==
True,



"NRMSE_Transient_SS":

test_meta[
    "is_local_transient"
]
==
True

}





for metric_name,mask in stage_masks.items():


    idx=np.where(mask)[0]



    for i,target in enumerate(TARGETS):


        if len(idx)>0:


            rmse=np.sqrt(

                mean_squared_error(

                    true_test[idx,i],

                    pred_test[idx,i]

                )

            )


            value=(

                rmse /

                target_range_final[i]

            )

        else:

            value=np.nan



        metrics_df.loc[

            metrics_df.Target==target,

            metric_name

        ]=value





# ============================================================
# 35. Save metrics
# ============================================================


metrics_df.to_csv(

    OUTPUT_DIR /

    "m6_final_metrics.csv",

    index=False

)



metrics_df.to_csv(

    OUTPUT_DIR /

    "m6_final_metrics_complete.csv",

    index=False

)



print("\n======================================")
print("M6 BASIC TEST METRICS")
print("======================================")


print(
    metrics_df
)



print("\n======================================")
print("M6 Part 4 completed")
print("======================================")


# In[5]:


# ============================================================
# M6 Supplement Evaluation FINAL
#
# Methodology identical to M5
#
# Calculate:
#   NRMSE_SS
#   NRMSE_LocalSteady_SS
#   NRMSE_Transient_SS
#   Eeq_SS
#   TCR
#   Gradient RMSE
#   GCR
#
# ============================================================


# ============================================================
# 36. Test metadata
# ============================================================


test_meta = df.loc[
    test_idx
].copy()



ss_mask = (

    test_meta["process_stage"]

    ==

    "ss_stage"

)



local_steady_mask = (

    ss_mask

    &

    test_meta[
        "is_local_steady"
    ]

)



transient_mask = (

    ss_mask

    &

    test_meta[
        "is_local_transient"
    ]

)



ss_indices = np.where(
    ss_mask
)[0]


local_steady_indices = np.where(
    local_steady_mask
)[0]


transient_indices = np.where(
    transient_mask
)[0]



print("\n======================================")
print("M6 TEST SUBSET COUNTS")
print("======================================")


print(
    "All Test:",
    len(test_idx)
)


print(
    "SS-stage Test:",
    len(ss_indices)
)


print(
    "Local-steady SS Test:",
    len(local_steady_indices)
)


print(
    "Transient SS Test:",
    len(transient_indices)
)





# ============================================================
# 37. Fixed normalization range from original TRAIN only
# ============================================================

# Same immutable range as used in validation and final test metrics.
# Do not reconstruct it from RMSE / NRMSE or shadow `target_range`.
target_range_by_gas = {
    gas: float(target_range_final[i])
    for i, gas in enumerate(TARGETS)
}

print("\nTRAIN-only NRMSE normalization ranges:")
for gas, value in target_range_by_gas.items():
    print(gas, ":", value)



# ============================================================
# 38. Subset NRMSE
#
# Same definition as M5
#
# ============================================================


def subset_nrmse(
        y_true,
        y_pred,
        indices,
        target
):


    if len(indices)==0:

        return np.nan



    rmse = np.sqrt(

        mean_squared_error(

            y_true[
                indices
            ],

            y_pred[
                indices
            ]

        )

    )



    return (

        rmse

        /

        target_range_by_gas[target]

    )





for i,target in enumerate(TARGETS):


    metrics_df.loc[

        metrics_df.Target==target,

        "NRMSE_SS"

    ] = subset_nrmse(

        y_test.to_numpy()[:,i],

        pred_test[:,i],

        ss_indices,

        target

    )



    metrics_df.loc[

        metrics_df.Target==target,

        "NRMSE_LocalSteady_SS"

    ] = subset_nrmse(

        y_test.to_numpy()[:,i],

        pred_test[:,i],

        local_steady_indices,

        target

    )



    metrics_df.loc[

        metrics_df.Target==target,

        "NRMSE_Transient_SS"

    ] = subset_nrmse(

        y_test.to_numpy()[:,i],

        pred_test[:,i],

        transient_indices,

        target

    )





# ============================================================
# 39. FactSage equilibrium and gradient target
# ============================================================


(
    eq_test,

    grad_test,

    physics_test_mask

)=build_physics_targets(

    test_meta

)





# ============================================================
# 40. Eeq_SS and TCR
#
# Same methodology as M5
#
# ============================================================


eq_metrics = {}



y_test_array = y_test.to_numpy()



for i,target in enumerate(
    FACTSAGE_TARGETS
):


    target_idx = TARGETS.index(
        target
    )



    measured_ss = (

        y_test_array[
            ss_indices,
            target_idx
        ]

    )


    predicted_ss = (

        pred_test[
            ss_indices,
            target_idx
        ]

    )


    eq_ss = (

        eq_test[
            ss_indices,
            i
        ]

    )



    # ----------------------------------------
    # Eeq_SS
    #
    # normalized equilibrium RMSE
    # ----------------------------------------


    rmse_eq = np.sqrt(

        np.mean(

            (

                predicted_ss

                -

                eq_ss

            )**2

        )

    )


    Eeq_SS = (

        rmse_eq

        /

        target_range_by_gas[target]

    )



    # ----------------------------------------
    # TCR
    # ----------------------------------------


    model_error = np.abs(

        predicted_ss

        -

        eq_ss

    )



    experiment_error = np.abs(

        measured_ss

        -

        eq_ss

    )


    TCR = np.mean(

        model_error

        <=

        experiment_error

    )



    eq_metrics[target]={

        "Eeq_SS":Eeq_SS,

        "TCR":TCR

    }





# ============================================================
# 41. Gradient prediction
# ============================================================


final_model.eval()



gradient_shift_test = get_gradient_shift(

    x_scaler_final

)



X_test_tensor = torch.tensor(

    X_test_scaled_final,

    dtype=torch.float32,

    device=DEVICE

)



grad_pred_test = (

    model_gradient_physical(

        final_model,

        X_test_tensor,

        gradient_shift_test,

        y_scaler_final

    )

    .detach()

    .cpu()

    .numpy()

)





# ============================================================
# 42. Gradient RMSE and GCR
#
# Same methodology as M5
#
# ============================================================


gradient_metrics=[]



for i,target in enumerate(
    FACTSAGE_TARGETS
):


    pred_grad = (

        grad_pred_test[
            ss_indices,
            i
        ]

    )


    fs_grad = (

        grad_test[
            ss_indices,
            i
        ]

    )



    # remove invalid values

    valid = (

        np.isfinite(pred_grad)

        &

        np.isfinite(fs_grad)

    )


    pred_grad = pred_grad[valid]

    fs_grad = fs_grad[valid]



    # ----------------------------------------
    # Gradient RMSE
    # ----------------------------------------


    grad_rmse = np.sqrt(

        np.mean(

            (

                pred_grad

                -

                fs_grad

            )**2

        )

    )



    # ----------------------------------------
    # GCR
    # ----------------------------------------


    epsilon_g = (

        0.01

        *

        np.max(

            np.abs(fs_grad)

        )

    )



    gcr_valid = (

        np.abs(fs_grad)

        >

        epsilon_g

    )



    if gcr_valid.sum()>0:


        GCR = np.mean(

            (

                pred_grad[gcr_valid]

                *

                fs_grad[gcr_valid]

            )

            >

            0

        )

    else:


        GCR=np.nan





    gradient_metrics.append(

        {

        "Target":
        target,


        "Grad_RMSE_Physical":
        grad_rmse,


        "GCR":
        GCR,


        "N_Gradient_Probes":
        len(pred_grad)

        }

    )



gradient_df = pd.DataFrame(
    gradient_metrics
)





# ============================================================
# 43. Merge final metrics
# ============================================================


complete_metrics = metrics_df.copy()



for target,values in eq_metrics.items():


    complete_metrics.loc[

        complete_metrics.Target==target,

        "Eeq_SS"

    ] = values["Eeq_SS"]



    complete_metrics.loc[

        complete_metrics.Target==target,

        "TCR"

    ] = values["TCR"]





for _,row in gradient_df.iterrows():


    target=row["Target"]


    complete_metrics.loc[

        complete_metrics.Target==target,

        "Grad_RMSE_Physical"

    ] = row[
        "Grad_RMSE_Physical"
    ]



    complete_metrics.loc[

        complete_metrics.Target==target,

        "GCR"

    ] = row[
        "GCR"
    ]





# CH4 no thermodynamic constraint


complete_metrics.loc[

    complete_metrics.Target=="CH4",

    [

        "Eeq_SS",

        "TCR",

        "Grad_RMSE_Physical",

        "GCR"

    ]

]=np.nan





# ============================================================
# 44. Save
# ============================================================


complete_metrics.to_csv(

    OUTPUT_DIR /

    "m6_final_metrics_complete.csv",

    index=False

)



gradient_df.to_csv(

    OUTPUT_DIR /

    "m6_test_gradient_metrics.csv",

    index=False

)



print("\n======================================")
print("M6 COMPLETE TEST METRICS")
print("======================================")


print(
    complete_metrics
)



print("\n======================================")
print("M6 GRADIENT METRICS")
print("======================================")


print(
    gradient_df
)



print("\nSaved:")


print(
    OUTPUT_DIR /
    "m6_final_metrics_complete.csv"
)


print(
    OUTPUT_DIR /
    "m6_test_gradient_metrics.csv"
)


# In[ ]:




