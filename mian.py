import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import optuna
import shap
from sklearn.metrics import average_precision_score
import copy
import os
import matplotlib.pyplot as plt

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, num_numerical=5, num_categorical=3, onehot_size=10, num_classes=3):
        self.num_samples = num_samples
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.onehot_size = onehot_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        numerical = torch.randn(self.num_numerical)
        categorical_emb = torch.randint(0, 10, (self.num_categorical,))
        categorical_onehot = torch.randint(0, 2, (self.onehot_size,), dtype=torch.float)
        target = torch.randint(0, self.num_classes, (1,)).item()
        return numerical, categorical_emb, categorical_onehot, target

# Focal Loss
class FocalLossMulticlass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha if alpha is not None else torch.ones(1)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha[targets]
        else:
            at = 1.0
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# CrossLayerV2 and DCNv2
class CrossLayerV2(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        return x0 * self.V(self.U(xl)) + self.bias + xl

class DCNv2Multiclass(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim, cross_layers, cross_rank, deep_layers, onehot_size, num_classes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card + 1, embedding_dim) for card in cat_cardinalities])
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim + onehot_size
        self.cross_net = nn.ModuleList([CrossLayerV2(input_dim, cross_rank) for _ in range(cross_layers)])
        deep = []
        dims = [input_dim] + deep_layers
        for i in range(len(deep_layers)):
            deep += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.deep_net = nn.Sequential(*deep)
        self.output_layer = nn.Linear(input_dim + deep_layers[-1], num_classes)

    def forward(self, numerical, categorical_emb, categorical_onehot):
        cat_embeds = [emb(categorical_emb[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        x = torch.cat([numerical, cat_embeds, categorical_onehot], dim=1)
        x0, xl = x, x
        for layer in self.cross_net:
            xl = layer(x0, xl)
        deep_x = self.deep_net(x)
        combined = torch.cat([xl, deep_x], dim=1)
        return self.output_layer(combined)

# Training
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model = None

    def should_stop(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

def train_model_with_prauc(train_loader, val_loader, model, optimizer, criterion, device, num_classes, epochs=20, model_path='best_model.pt'):
    model.to(device)
    early_stopper = EarlyStopper(patience=3)
    for epoch in range(epochs):
        model.train()
        for numerical, categorical_emb, categorical_onehot, target in train_loader:
            numerical, categorical_emb, categorical_onehot, target = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(numerical, categorical_emb, categorical_onehot)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for numerical, categorical_emb, categorical_onehot, target in val_loader:
                numerical, categorical_emb, categorical_onehot = numerical.to(device), categorical_emb.to(device), categorical_onehot.to(device)
                logits = model(numerical, categorical_emb, categorical_onehot)
                probs = F.softmax(logits, dim=1)
                all_preds.append(probs.cpu())
                all_targets.append(target.cpu())

        y_true = torch.cat(all_targets).numpy()
        y_scores = torch.cat(all_preds).numpy()
        prauc = average_precision_score(y_true=y_true, y_score=y_scores, average='macro')
        print(f"Epoch {epoch+1}, PRAUC: {prauc:.4f}")

        if early_stopper.should_stop(prauc, model):
            print("Early stopping triggered. Saving and restoring best model.")
            torch.save(early_stopper.best_model, model_path)
            break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    train_dataset = DummyDataset(num_samples=512, num_classes=num_classes)
    val_dataset = DummyDataset(num_samples=128, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    numerical_feature_names = ["age", "account_balance", "credit_score", "savings_rate", "income_growth"]
    categorical_feature_names = ["education_level", "employment_type", "marital_status"]
    onehot_feature_names = [f"region_{i}" for i in range(10)]

    def objective(trial):
        model = DCNv2Multiclass(
            num_numerical=5,
            cat_cardinalities=[10, 10, 10],
            embedding_dim=trial.suggest_categorical("embedding_dim", [4, 8]),
            cross_layers=trial.suggest_int("cross_layers", 1, 3),
            cross_rank=trial.suggest_int("cross_rank", 4, 16),
            deep_layers=[trial.suggest_int("deep_layer_1", 32, 128, step=32), trial.suggest_int("deep_layer_2", 16, 64, step=16)],
            onehot_size=10,
            num_classes=num_classes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
        criterion = FocalLossMulticlass(gamma=2.0, alpha=torch.tensor([1.0] * num_classes))
        train_model_with_prauc(train_loader, val_loader, model, optimizer, criterion, device, num_classes, epochs=10)
        return average_precision_score(
            torch.cat([target for _, _, _, target in val_loader]).numpy(),
            torch.cat([F.softmax(model(*[x.to(device) for x in batch[:3]]), dim=1).detach().cpu() for batch in val_loader]).numpy(),
            average='macro'
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    final_model = DCNv2Multiclass(
        num_numerical=5,
        cat_cardinalities=[10, 10, 10],
        embedding_dim=best_params["embedding_dim"],
        cross_layers=best_params["cross_layers"],
        cross_rank=best_params["cross_rank"],
        deep_layers=[best_params["deep_layer_1"], best_params["deep_layer_2"]],
        onehot_size=10,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])
    criterion = FocalLossMulticlass(gamma=2.0, alpha=torch.tensor([1.0] * num_classes))
    train_model_with_prauc(train_loader, val_loader, final_model, optimizer, criterion, device, num_classes, epochs=10)

    numerical, categorical_emb, categorical_onehot, _ = next(iter(val_loader))
    numerical = numerical.to(device)
    categorical_emb = categorical_emb.to(device)
    categorical_onehot = categorical_onehot.to(device)

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        return F.softmax(final_model(
            x_tensor[:, :5],
            x_tensor[:, 5:8].long(),
            x_tensor[:, 8:]
        ), dim=1).detach().cpu().numpy()

    X_combined = torch.cat([numerical, categorical_emb.float(), categorical_onehot], dim=1).cpu().numpy()
    explainer = shap.Explainer(model_wrapper, X_combined)
    shap_values = explainer(X_combined[:100])

    values = shap_values.values
    mean_abs_shap_per_class = np.mean(np.abs(values), axis=0).T

    for class_idx in range(num_classes):
        grouped_names = []
        grouped_values = []

        for i, name in enumerate(numerical_feature_names):
            grouped_names.append(name)
            grouped_values.append(mean_abs_shap_per_class[class_idx][i])

        for j, name in enumerate(categorical_feature_names):
            start = len(numerical_feature_names) + j * best_params["embedding_dim"]
            end = start + best_params["embedding_dim"]
            grouped_names.append(name)
            grouped_values.append(np.mean(mean_abs_shap_per_class[class_idx][start:end]))

        start = len(numerical_feature_names) + len(categorical_feature_names) * best_params["embedding_dim"]
        grouped_names.append("region")
        grouped_values.append(np.mean(mean_abs_shap_per_class[class_idx][start:]))

        pairs = list(zip(grouped_names, grouped_values))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top_k = 15
        top_names, top_vals = zip(*pairs[:top_k])
        plt.figure(figsize=(10, 6))
        plt.barh(top_names[::-1], top_vals[::-1])
        plt.title(f"Grouped SHAP Feature Importance — Class {class_idx}")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.show()
