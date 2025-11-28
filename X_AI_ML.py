import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Settings to fix np.int issue
np.int = int

# Better display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -----------------------------
# Loading and preparing data
# -----------------------------
df = pd.read_excel("Data/combined_data_Deaths.xlsx")
target = "Deaths"

# Separating features
X = df.drop(columns=[target, "County_Name", "County", "State"], errors="ignore")
y = df[target]

# Selecting only numeric columns and filling missing values
X = X.select_dtypes(include=[np.number]).fillna(X.mean())

# Saving feature names
feature_names = X.columns.tolist()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")
print(f"Number of features: {len(feature_names)}")

# -----------------------------
# Training Random Forest model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("Model Evaluation Metrics:")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# -----------------------------
# Calculating SHAP Values
# -----------------------------
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# -----------------------------
# 1. Summary Plot (Beeswarm) - Feature importance and impact
# -----------------------------
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.title("Distribution of Feature Impacts on Mortality Prediction", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("ML/shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 2. Bar Plot - Mean absolute feature importance
# -----------------------------
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
plt.title("Absolute Feature Importance (Mean |SHAP value|)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("ML/shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 3. Feature importance table with detailed statistics
# -----------------------------
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

# Saving full table
feature_importance.to_excel("ML/shap_feature_importance_detailed.xlsx", index=False)
print("\nFull table saved in file 'ML/shap_feature_importance_detailed.xlsx'.")

# -----------------------------
# 4. Dependence Plots - Impact of top 6 features
# -----------------------------
top_features = feature_importance.head(6)['Feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        ax=axes[idx],
        show=False
    )
    axes[idx].set_title(f"Impact of {feature}", fontsize=12)

plt.tight_layout()
plt.savefig("ML/shap_dependence_plots.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 5. Force Plot for specific samples
# -----------------------------
print("\nSaving Force Plot for the first test sample...")
shap.force_plot(
    explainer.expected_value,
    shap_values[0,:],
    X_test.iloc[0,:],
    matplotlib=True,
    show=False
)
plt.savefig("ML/shap_force_plot_sample.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 6. Analysis of interactions between features (optional)
# -----------------------------
print("\nCalculating feature interactions (may be time-consuming)...")
# For speed, applied only on a subset of data
sample_size = min(500, len(X_test))
X_sample = X_test.iloc[:sample_size]

shap_interaction_values = explainer.shap_interaction_values(X_sample)

# Displaying interaction between two important features
if len(top_features) >= 2:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        (top_features[0], top_features[1]),
        shap_interaction_values,
        X_sample,
        show=False
    )
    plt.title(f"Interaction between {top_features[0]} and {top_features[1]}", fontsize=14)
    plt.tight_layout()
    plt.savefig("ML/shap_interaction_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# 7. Analysis of specific samples (high and low)
# -----------------------------
errors = np.abs(y_test.values - y_pred)
worst_pred_idx = errors.argmax()
best_pred_idx = errors.argmin()

print("\n" + "="*50)
print("Analysis of Specific Samples:")
print("="*50)

print(f"\nWorst Prediction (High Error):")
print(f"  Actual Value: {y_test.iloc[worst_pred_idx]:.2f}")
print(f"  Prediction: {y_pred[worst_pred_idx]:.2f}")
print(f"  Error: {errors[worst_pred_idx]:.2f}")

print(f"\nBest Prediction (Low Error):")
print(f"  Actual Value: {y_test.iloc[best_pred_idx]:.2f}")
print(f"  Prediction: {y_pred[best_pred_idx]:.2f}")
print(f"  Error: {errors[best_pred_idx]:.2f}")

# Force plot for worst prediction
plt.figure(figsize=(20, 3))
shap.force_plot(
    explainer.expected_value,
    shap_values[worst_pred_idx,:],
    X_test.iloc[worst_pred_idx,:],
    matplotlib=True,
    show=False
)
plt.title("SHAP Analysis for Worst Prediction", fontsize=14)
plt.savefig("ML/shap_worst_prediction.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 8. Distribution of SHAP values for important features
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx in range(min(4, len(top_features))):
    feature = top_features[idx]
    feature_idx = feature_names.index(feature)
    
    axes[idx].hist(shap_values[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('SHAP Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of SHAP values: {feature}')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ML/shap_value_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 9. Final summary
# -----------------------------
print("\n" + "="*50)
print("SHAP Analysis Summary:")
print("="*50)
print(f"✓ Total number of features: {len(feature_names)}")
print(f"✓ Model Accuracy (R²): {r2:.4f}")
print(f"✓ Mean Absolute Error: {mae:.4f}")
print(f"\nTop 5 Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Mean_|SHAP|']:.4f}")

print("\n✓ All charts and tables have been saved.")
print("="*50)


importance = np.mean(np.abs(shap_values), axis=0)
importance_sorted = sorted(
    zip(X.columns, importance), key=lambda x: x[1], reverse=True
)

pd.DataFrame(importance_sorted, columns=["Feature", "Importance"]).to_csv(
    "ML/shap_feature_importance.csv", index=False
)

print("File saved.")