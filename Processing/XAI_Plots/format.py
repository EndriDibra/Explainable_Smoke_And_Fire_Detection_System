import pandas as pd
import matplotlib.pyplot as plt

# Your model ranking data
data = {
    "Model": [
        "DecisionTree", "LightGBM", "RandomForest", "XGBoost", "HistGradientBoosting",
        "CatBoost", "AdaBoost", "ExtraTree", "GradientBoosting", "KNN",
        "KerasNN", "SVC", "MLPClassifier", "LogisticRegression", "RidgeClassifier",
        "GaussianNB", "Dummy"
    ],
    "Accuracy": [
        1.0, 1.0, 1.0, 0.99992, 0.99992,
        0.99992, 0.99976, 0.999601, 0.999601, 0.998004,
        0.996886, 0.968066, 0.964713, 0.895098, 0.870749,
        0.768322, 0.714594
    ],
    "Precision": [
        1.0, 1.0, 1.0, 1.0, 0.999888,
        1.0, 0.999777, 0.999553, 0.999665, 0.998436,
        0.995993, 0.964073, 0.95566, 0.908964, 0.857032,
        0.765051, 0.714594
    ],
    "Recall": [
        1.0, 1.0, 1.0, 0.999888, 1.0,
        0.999888, 0.999888, 0.999888, 0.999777, 0.998771,
        0.999665, 0.992291, 0.996872, 0.948162, 0.98313,
        0.97531, 1.0
    ],
    "F1 Score": [
        1.0, 1.0, 1.0, 0.9999, 0.9999,
        0.9999, 0.9998, 0.9997, 0.9997, 0.9986,
        0.9978, 0.9780, 0.9758, 0.9281, 0.9158,
        0.8575, 0.8335
    ],
    "AUC Score": [
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.999999, 0.999385, 1.0, 0.999292,
        0.999934, 0.996984, 0.993913, 0.963708, None,
        0.93613, 0.5
    ],
    "Training Time": [
        0.282650, 0.511582, 6.328965, 0.340062, 1.894807,
        13.376978, 4.880908, 0.029572, 16.584621, 2.337749,
        63.780287, 150.555977, 4.943271, 0.417725, 0.053782,
        0.073078, 0.015080
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Round numeric columns for display
numeric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score", "Training Time"]
df[numeric_cols] = df[numeric_cols].round(3)

# Plot table
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')  # Hide axes

# Create the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['lightblue']*len(df.columns)
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)  # Adjust table size

# Save and show
plt.tight_layout()
plt.savefig("model_ranking_table.png", dpi=300, bbox_inches='tight')
plt.show()