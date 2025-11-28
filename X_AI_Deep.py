import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Settings
np.int = int
torch.manual_seed(42)
np.random.seed(42)

# Better display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -------------------------------------------------
# 1) Loading and preparing data
# -------------------------------------------------
df = pd.read_excel("Data/combined_data_Deaths.xlsx")
target = "Deaths"

X = df.drop(columns=[target, "County_Name", "County", "State"], errors="ignore")
y = df[target]

# Selecting numeric features and filling missing values
X = X.select_dtypes(include=[np.number]).fillna(X.mean())
feature_names = X.columns.tolist()

print(f"Number of samples: {len(X)}")
print(f"Number of features: {len(feature_names)}")

# -------------------------------------------------
# 2) Normalizing data
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")

# Converting to tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -------------------------------------------------
# 3) Building PyTorch model (Improved MLP)
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = MLP(X_train.shape[1], hidden_dims=[128, 64, 32], dropout=0.3)

# -------------------------------------------------
# 4) Training model with Loss monitoring
# -------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

epochs = 100
batch_size = 32
train_losses = []
val_losses = []

print("\n" + "="*50)
print("Starting model training...")
print("="*50)

# Best model
best_val_loss = float('inf')
best_model_state = None

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Evaluation on validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t).item()
    model.train()
    
    avg_train_loss = epoch_loss / (len(X_train_t) // batch_size)
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    # Saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Loading best model
model.load_state_dict(best_model_state)
print("\nModel training completed. Best model loaded.")

# -------------------------------------------------
# 5) Loss plot
# -------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Model Learning Curve', fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Deep/pytorch_training_loss.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 6) Model evaluation
# -------------------------------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).numpy().flatten()

y_test_np = y_test.values

r2 = r2_score(y_test_np, y_pred)
mae = mean_absolute_error(y_test_np, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))

print("\n" + "="*50)
print("Model Evaluation Metrics:")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test_np.min(), y_test_np.max()], 
         [y_test_np.min(), y_test_np.max()], 
         'r--', linewidth=2, label='Ideal Line')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title(f'Comparison of Actual and Predicted Values (R² = {r2:.4f})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Deep/pytorch_prediction_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 7) SHAP GradientExplainer
# -------------------------------------------------
print("\n" + "="*50)
print("Calculating SHAP values...")
print("="*50)

# Selecting background data (representative sample from training data)
background_size = min(200, len(X_train_t))
background_data = X_train_t[np.random.choice(len(X_train_t), background_size, replace=False)]

# Selecting test data for analysis
test_sample_size = min(500, len(X_test_t))
X_test_sample = X_test_t[:test_sample_size]
X_test_sample_np = X_test[:test_sample_size]

# Creating explainer
explainer = shap.GradientExplainer(model, background_data)

# Calculating SHAP values
shap_values = explainer.shap_values(X_test_sample)

# If shap_values is a list (common in PyTorch), select the first output
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# shap_values should now be 2D (samples × features)
shap_values = np.array(shap_values)

# Squeeze if model has extra output dimensions
shap_values = shap_values.squeeze()

print(f"SHAP values shape: {shap_values.shape}")

# -------------------------------------------------
# 8) SHAP plots
# -------------------------------------------------

# Summary Plot (Beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample_np, 
                  feature_names=feature_names, 
                  max_display=20, show=False)
plt.title("Distribution of Feature Impacts on Prediction (PyTorch)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_summary_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()

# Bar Plot - Feature importance
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample_np, 
                  feature_names=feature_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("Absolute Feature Importance (PyTorch)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 9) Detailed feature importance table
# -------------------------------------------------
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_|SHAP|': np.abs(shap_values).mean(axis=0),
    'Mean_SHAP': shap_values.mean(axis=0),
    'Std_SHAP': shap_values.std(axis=0),
    'Max_SHAP': shap_values.max(axis=0),
    'Min_SHAP': shap_values.min(axis=0)
}).sort_values('Mean_|SHAP|', ascending=False)

print("\n" + "="*50)
print("Top 20 Important Features:")
print("="*50)
print(feature_importance.head(20).to_string(index=False))

# Saving table
feature_importance.to_excel("Deep/shap_pytorch_feature_importance_detailed.xlsx", index=False)
feature_importance.to_csv("Deep/shap_pytorch_feature_importance.csv", index=False)
print("\nFeature importance table saved.")

# -------------------------------------------------
# 10) Dependence Plots for top 6 features
# -------------------------------------------------
top_features = feature_importance.head(6)['Feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    feature_idx = feature_names.index(feature)
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test_sample_np,
        feature_names=feature_names,
        ax=axes[idx],
        show=False
    )
    axes[idx].set_title(f"Impact of {feature}", fontsize=12)

plt.tight_layout()
plt.savefig("Deep/shap_pytorch_dependence_plots.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 11) Force Plot for specific samples
# -------------------------------------------------
print("\nAnalyzing specific samples...")

# Calculating errors
errors = np.abs(y_test_np[:test_sample_size] - y_pred[:test_sample_size])
worst_idx = errors.argmax()
best_idx = errors.argmin()

print(f"\nWorst Prediction:")
print(f"  Actual Value: {y_test_np[worst_idx]:.2f}")
print(f"  Prediction: {y_pred[worst_idx]:.2f}")
print(f"  Error: {errors[worst_idx]:.2f}")

print(f"\nBest Prediction:")
print(f"  Actual Value: {y_test_np[best_idx]:.2f}")
print(f"  Prediction: {y_pred[best_idx]:.2f}")
print(f"  Error: {errors[best_idx]:.2f}")

# Manually calculating expected_value
model.eval()
with torch.no_grad():
    background_preds = model(background_data)
    expected_value = background_preds.mean().item()

# Force plot for worst prediction
plt.figure(figsize=(20, 3))
shap.force_plot(
    expected_value,
    shap_values[worst_idx],
    X_test_sample_np[worst_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title("SHAP Analysis for Worst Prediction (PyTorch)", fontsize=14)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_worst_prediction.png", dpi=300, bbox_inches='tight')
plt.show()

# Force plot for best prediction
plt.figure(figsize=(20, 3))
shap.force_plot(
    expected_value,
    shap_values[best_idx],
    X_test_sample_np[best_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title("SHAP Analysis for Best Prediction (PyTorch)", fontsize=14)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_best_prediction.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 12) Distribution of SHAP values for important features
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx in range(min(4, len(top_features))):
    feature = top_features[idx]
    feature_idx = feature_names.index(feature)
    
    axes[idx].hist(shap_values[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('SHAP Value', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'SHAP Distribution: {feature}', fontsize=12)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Deep/shap_pytorch_value_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 13) Comparison of feature importance (Top 15)
# -------------------------------------------------
top_15 = feature_importance.head(15)

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
bars = plt.barh(range(len(top_15)), top_15['Mean_|SHAP|'], color=colors)
plt.yticks(range(len(top_15)), top_15['Feature'])
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.title('Top 15 Important Features (PyTorch Model)', fontsize=14, pad=15)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_top15_features.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 14) SHAP values correlation matrix
# -------------------------------------------------
# Calculating correlation between SHAP values of important features
top_10_indices = [feature_names.index(f) for f in top_features[:10] if f in feature_names]
shap_top10 = shap_values[:, top_10_indices]

corr_matrix = np.corrcoef(shap_top10.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            xticklabels=[top_features[i] for i in range(min(10, len(top_features)))],
            yticklabels=[top_features[i] for i in range(min(10, len(top_features)))],
            annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation of SHAP Values for Important Features', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("Deep/shap_pytorch_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------
# 15) Final summary
# -------------------------------------------------
print("\n" + "="*50)
print("SHAP Analysis Summary (PyTorch):")
print("="*50)
print(f"✓ Model Architecture: MLP with {len(model.network)} layers")
print(f"✓ Number of features: {len(feature_names)}")
print(f"✓ Model Accuracy (R²): {r2:.4f}")
print(f"✓ RMSE: {rmse:.4f}")
print(f"✓ Number of analyzed samples: {test_sample_size}")
print(f"\nTop 5 Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Mean_|SHAP|']:.4f}")

print("\n✓ All charts, tables, and model saved.")
print("="*50)

# Saving model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'feature_names': feature_names,
    'performance': {'r2': r2, 'mae': mae, 'rmse': rmse}
}, 'Deep/pytorch_model_complete.pth')

print("\n✓ Full model saved in 'pytorch_model_complete.pth'.")