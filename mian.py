import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize, LabelEncoder
import copy
import matplotlib.pyplot as plt
import random
import logging

# --- Configuration & Setup ---
CONFIG = {
    "seed": 42,
    # --- Data Configuration ---
    "dataframe_path": "dummy_data.csv",  # Path to your input CSV DataFrame
    "target_column_name": "target",            # Name of the target variable column
    "numerical_col_names": ["num_1", "num_2"],                 # List of numerical column names
    "categorical_col_names_embed": ["cat_embed_1", "cat_embed_2"],         # List of categorical columns for embedding
    "onehot_col_names": ["onehot_1", "onehot_2"],                    # List of columns already one-hot encoded (0/1)
    # --- Model & Training General ---
    "num_classes": 3,                          # Number of target classes (if known, otherwise inferred for classification)
    "batch_size": 64,
    "epochs_optuna": 2, # Reduced for dummy data
    "epochs_final": 3,  # Reduced for dummy data
    "early_stopper_patience": 2, # Reduced for dummy data
    "early_stopper_min_delta": 0.001,
    "optuna_trials": 1, # Set to 0 to skip Optuna and use default_hyperparams
    "default_hyperparams": { # Used if optuna_trials is 0
        "embedding_dim": 8,
        "cross_layers": 1,
        "cross_rank": 8,
        "n_deep_layers": 1,
        "deep_layer_1_units": 32,
        "deep_layer_2_units": 16,
        "lr": 0.001,
    },
    # --- SHAP Configuration ---
    "shap_samples": 20, # Samples for background and explanation. Ensure val_dataset is larger.
    # --- Paths ---
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path_template": "best_model_trial_{}.pt",
    "final_model_path": "final_best_model.pt",
    "shap_plot_save_template": "shap_summary_class_{}.png"
}

# --- Reproducibility ---
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
set_seed(CONFIG["seed"])

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- DataFrame Preprocessing ---
def preprocess_dataframe(df, numerical_cols, categorical_cols_embed, onehot_cols, target_col):
    """
    Preprocesses the DataFrame:
    - Validates columns
    - Label encodes categorical features for embedding
    - Determines cardinalities for categorical features
    - Extracts features and target
    """
    df_processed = df.copy()

    # Validate column names
    all_feature_cols = numerical_cols + categorical_cols_embed + onehot_cols
    for col_list in [numerical_cols, categorical_cols_embed, onehot_cols, [target_col]]:
        for col in col_list:
            if col not in df_processed.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Check for overlap in feature column lists
    if len(set(numerical_cols) & set(categorical_cols_embed)) > 0 or \
       len(set(numerical_cols) & set(onehot_cols)) > 0 or \
       len(set(categorical_cols_embed) & set(onehot_cols)) > 0:
        raise ValueError("Feature column lists (numerical, categorical_embed, onehot) must be disjoint.")

    # Label encode target if it's not already numerical
    if df_processed[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_processed[target_col]):
        target_encoder = LabelEncoder()
        df_processed[target_col] = target_encoder.fit_transform(df_processed[target_col])
        logging.info(f"Target column '{target_col}' label encoded. Classes: {target_encoder.classes_}")
        # Update num_classes if not set or different
        CONFIG['num_classes'] = len(target_encoder.classes_)


    # Label encode categorical features for embedding and get cardinalities
    cat_label_encoders = {}
    cat_cardinalities = []
    for col in categorical_cols_embed:
        encoder = LabelEncoder()
        df_processed[col] = encoder.fit_transform(df_processed[col].astype(str)) # astype(str) to handle mixed types or NaNs
        cat_label_encoders[col] = encoder
        cat_cardinalities.append(len(encoder.classes_))
        logging.info(f"Categorical column '{col}' label encoded. Cardinality: {len(encoder.classes_)}")

    # Ensure one-hot columns are numeric (0/1)
    for col in onehot_cols:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            try: # Attempt conversion if not numeric, e.g. boolean
                df_processed[col] = df_processed[col].astype(float)
            except ValueError:
                raise ValueError(f"One-hot column '{col}' must be numeric or convertible to numeric (0/1). Found dtype: {df_processed[col].dtype}")
        # Optionally, check if values are indeed 0 or 1
        # if not df_processed[col].isin([0, 1]).all():
        #     logging.warning(f"One-hot column '{col}' contains values other than 0 or 1.")


    num_numerical_features = len(numerical_cols)
    num_categorical_features_embed = len(categorical_cols_embed)
    onehot_features_size = len(onehot_cols)

    # Update CONFIG with derived values (important for model and SHAP)
    CONFIG['derived_num_numerical'] = num_numerical_features
    CONFIG['derived_num_categorical_embed'] = num_categorical_features_embed
    CONFIG['derived_cat_cardinalities'] = cat_cardinalities
    CONFIG['derived_onehot_size'] = onehot_features_size

    # Store all feature names in the order they will be concatenated for the model
    # This order is: numerical, categorical_embed (indices), onehot
    CONFIG['ordered_feature_names_for_shap'] = numerical_cols + categorical_cols_embed + onehot_cols


    return df_processed, cat_label_encoders


# --- Custom Dataset for Pandas DataFrame ---
class PandasDataset(Dataset):
    def __init__(self, dataframe, numerical_cols, categorical_cols_embed, onehot_cols, target_col):
        self.dataframe = dataframe
        self.numerical_cols = numerical_cols
        self.categorical_cols_embed = categorical_cols_embed
        self.onehot_cols = onehot_cols
        self.target_col = target_col

        self.numerical_data = torch.tensor(self.dataframe[self.numerical_cols].values, dtype=torch.float32) if self.numerical_cols else torch.empty((len(self.dataframe), 0), dtype=torch.float32)
        self.categorical_embed_data = torch.tensor(self.dataframe[self.categorical_cols_embed].values, dtype=torch.long) if self.categorical_cols_embed else torch.empty((len(self.dataframe), 0), dtype=torch.long)
        self.onehot_data = torch.tensor(self.dataframe[self.onehot_cols].values, dtype=torch.float32) if self.onehot_cols else torch.empty((len(self.dataframe), 0), dtype=torch.float32)
        self.target_data = torch.tensor(self.dataframe[self.target_col].values, dtype=torch.long)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return (
            self.numerical_data[idx],
            self.categorical_embed_data[idx],
            self.onehot_data[idx],
            self.target_data[idx]
        )

# --- Focal Loss (Unchanged) ---
class FocalLossMulticlass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            if num_classes is None:
                self.alpha = None
                logging.warning("FocalLoss alpha is None and num_classes is not provided. Alpha will be treated as 1.0 for all classes.")
            else:
                self.alpha = torch.ones(num_classes)
        else:
            if not isinstance(alpha, torch.Tensor):
                raise TypeError("alpha must be a torch.Tensor or None.")
            if num_classes is not None and alpha.shape[0] != num_classes:
                raise ValueError(f"alpha shape {alpha.shape} does not match num_classes {num_classes}.")
            self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_components = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            current_alpha = self.alpha
            if current_alpha.device != logits.device:
                current_alpha = current_alpha.to(logits.device)
            if current_alpha.shape[0] == 1 and logits.shape[1] > 1 and current_alpha.numel() == 1:
                 at = current_alpha.expand_as(targets)
            elif current_alpha.shape[0] == logits.shape[1]:
                 at = current_alpha[targets]
            else:
                logging.warning(f"FocalLoss alpha shape {current_alpha.shape} is not compatible with logits classes {logits.shape[1]}. Applying uniform alpha weighting.")
                at = 1.0
            focal_loss_components = at * focal_loss_components
        if self.reduction == 'mean':
            return focal_loss_components.mean()
        elif self.reduction == 'sum':
            return focal_loss_components.sum()
        else:
            return focal_loss_components

# --- DCNv2 Model (Adjusted for dynamic inputs) ---
class CrossLayerV2(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))
    def forward(self, x0, xl):
        return x0 * self.V(self.U(xl)) + self.bias + xl

class DCNv2Multiclass(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim,
                 cross_layers, cross_rank, deep_layers_dims, onehot_size, num_classes):
        super().__init__()

        self.num_numerical = num_numerical
        self.num_categorical_embed = len(cat_cardinalities)
        self.onehot_size = onehot_size

        self.embeddings = nn.ModuleList()
        if self.num_categorical_embed > 0:
            self.embeddings.extend([
                nn.Embedding(card, embedding_dim) for card in cat_cardinalities
            ])

        # Calculate total input dimension to the dense layers of DCN
        # This is after embeddings for categorical features
        current_input_dim = self.num_numerical
        if self.num_categorical_embed > 0:
            current_input_dim += self.num_categorical_embed * embedding_dim
        current_input_dim += self.onehot_size

        self.cross_net = nn.ModuleList([
            CrossLayerV2(current_input_dim, cross_rank) for _ in range(cross_layers)
        ])

        deep_network_layers = []
        input_to_deep_layer = current_input_dim # Deep net also sees the full concatenated input
        for layer_dim in deep_layers_dims:
            deep_network_layers.append(nn.Linear(input_to_deep_layer, layer_dim))
            deep_network_layers.append(nn.ReLU())
            input_to_deep_layer = layer_dim
        self.deep_net = nn.Sequential(*deep_network_layers)

        output_layer_input_dim = current_input_dim # From cross net
        if deep_layers_dims: # If there are deep layers
            output_layer_input_dim += deep_layers_dims[-1] # Add output from deep net

        self.output_layer = nn.Linear(output_layer_input_dim, num_classes)

    def forward(self, numerical, categorical_indices, categorical_onehot):
        # numerical: (batch, num_numerical)
        # categorical_indices: (batch, num_categorical_embed)
        # categorical_onehot: (batch, onehot_size)

        all_inputs = []
        if self.num_numerical > 0:
            all_inputs.append(numerical)

        if self.num_categorical_embed > 0:
            cat_embeds_list = []
            for i, emb_layer in enumerate(self.embeddings):
                cat_embeds_list.append(emb_layer(categorical_indices[:, i]))
            all_inputs.append(torch.cat(cat_embeds_list, dim=1))

        if self.onehot_size > 0:
            all_inputs.append(categorical_onehot)

        if not all_inputs:
             raise ValueError("Model received no input features. Check feature column configurations.")

        x = torch.cat(all_inputs, dim=1)

        x0 = x
        xl_cross = x
        for layer in self.cross_net:
            xl_cross = layer(x0, xl_cross)

        combined_output_parts = [xl_cross]
        if hasattr(self, 'deep_net') and len(list(self.deep_net.children())) > 0 :
            xl_deep = self.deep_net(x)
            combined_output_parts.append(xl_deep)

        combined = torch.cat(combined_output_parts, dim=1)
        return self.output_layer(combined)

# --- Training Utilities (Largely Unchanged, but check PRAUC binarization) ---
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, model_path='best_model.pt'):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0
        self.best_score = None; self.best_model_state_dict = None; self.model_path = model_path
    def should_stop(self, current_score, model):
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score; self.counter = 0
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(self.best_model_state_dict, self.model_path)
            logging.info(f"EarlyStopper: New best score: {self.best_score:.4f}. Model saved to {self.model_path}")
        else:
            self.counter += 1
            logging.info(f"EarlyStopper: Score did not improve. Counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience

def train_model_with_prauc(train_loader, val_loader, model, optimizer, criterion, device,
                           num_classes_train, epochs, model_save_path): # Renamed num_classes to num_classes_train
    model.to(device)
    early_stopper = EarlyStopper(patience=CONFIG["early_stopper_patience"],
                                 min_delta=CONFIG["early_stopper_min_delta"],
                                 model_path=model_save_path)

    for epoch in range(epochs):
        model.train(); epoch_loss = 0
