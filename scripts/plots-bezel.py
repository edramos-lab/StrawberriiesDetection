import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# Sample data
data = [
    {"Optimizer": "AdamW", "Batch Size": 32, "Learning Rate": 0.01, "Accuracy": 0.84},
    {"Optimizer": "AdamW", "Batch Size": 32, "Learning Rate": 0.001, "Accuracy": 0.72},
    {"Optimizer": "AdamW", "Batch Size": 64, "Learning Rate": 0.01, "Accuracy": 0.73},
    {"Optimizer": "AdamW", "Batch Size": 64, "Learning Rate": 0.001, "Accuracy": 0.85},
    {"Optimizer": "Adam", "Batch Size": 32, "Learning Rate": 0.01, "Accuracy": 0.73},
    {"Optimizer": "Adam", "Batch Size": 32, "Learning Rate": 0.001, "Accuracy": 0.82},
    {"Optimizer": "Adam", "Batch Size": 64, "Learning Rate": 0.01, "Accuracy": 0.75},
    {"Optimizer": "Adam", "Batch Size": 64, "Learning Rate": 0.001, "Accuracy": 0.89},
    {"Optimizer": "SGD", "Batch Size": 32, "Learning Rate": 0.01, "Accuracy": 0.69},
    {"Optimizer": "SGD", "Batch Size": 32, "Learning Rate": 0.001, "Accuracy": 0.77},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Normalize the values for plotting
for col in df.columns:
    if df[col].dtype != 'object':  # Skip categorical columns
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Categorical encoding for the Optimizer column
df['Optimizer'] = df['Optimizer'].astype('category').cat.codes

# Prepare sinusoidal-style parallel coordinates
fig, ax = plt.subplots(figsize=(10, 6))
columns = df.columns
x_positions = np.arange(len(columns))

# Draw each row as a set of sinusoidal curves
for _, row in df.iterrows():
    x = x_positions
    y = row.values

    # Create sinusoidal-like curves
    verts = [(x[0], y[0])]
    for i in range(1, len(x)):
        mid_x = (x[i-1] + x[i]) / 2
        mid_y = (y[i-1] + y[i]) / 2 + 0.1  # Add a curve effect
        verts.extend([(mid_x, mid_y), (x[i], y[i])])
    path = Path(verts, [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1))
    patch = PathPatch(path, facecolor='none', edgecolor='blue', lw=2, alpha=0.7)
    ax.add_patch(patch)

# Set the axes and labels
ax.set_xlim(-0.5, len(columns) - 0.5)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks(x_positions)
ax.set_xticklabels(columns, rotation=45, ha='right')
ax.set_ylabel("Normalized Values")
ax.set_title("Sinusoidal-Style Parallel Coordinates")

plt.tight_layout()
plt.show()
