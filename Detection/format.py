import pandas as pd
import matplotlib.pyplot as plt

# Your data directly as a dictionary
data = {
    "Model": ["YOLOv5n", "YOLOv8n", "YOLOv10n", "YOLOv11n", "YOLOv12n"],
    "mAP@0.50": [0.8267, 0.8320, 0.7906, 0.8293, 0.8306],
    "mAP@0.50-0.95": [0.5422, 0.5489, 0.5156, 0.5474, 0.5544],
    "Inference Speed (FPS)": ["8.5 - 10.5", "7.5 - 9.5", "6.9 - 9.5", "7.0 - 8.9", "6.2 - 7.3"],
    "Parameters (M)": [2.6, 3.2, 2.3, 2.6, 2.6]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute average FPS from the "min - max" ranges
df["FPS_avg"] = df["Inference Speed (FPS)"].apply(
    lambda x: sum(map(float, x.replace(" ", "").split("-"))) / 2
)

# Round values
df[["mAP@0.50", "mAP@0.50-0.95", "FPS_avg"]] = \
    df[["mAP@0.50", "mAP@0.50-0.95", "FPS_avg"]].round(4)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

x = range(len(df))
width = 0.25

bars1 = ax.bar([i - width for i in x], df["mAP@0.50"], width=width, label="mAP@0.50")
bars2 = ax.bar(x, df["mAP@0.50-0.95"], width=width, label="mAP@0.50–0.95")
bars3 = ax.bar([i + width for i in x], df["FPS_avg"], width=width, label="FPS (avg)")

# Add exact values above the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha='center', va='bottom', fontsize=9
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels(df["Model"])
ax.set_ylabel("Metric Value")
ax.set_title("YOLO Model Performance Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("yolo_model_performance_plot_values.png", dpi=300, bbox_inches='tight')
plt.show()