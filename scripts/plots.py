import plotly.express as px
import pandas as pd

# Sample hyperparameter data
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
    {"Optimizer": "SGD", "Batch Size": 64, "Learning Rate": 0.01, "Accuracy": 0.74},
    {"Optimizer": "SGD", "Batch Size": 64, "Learning Rate": 0.001, "Accuracy": 0.78},
    {"Optimizer": "RAdam", "Batch Size": 32, "Learning Rate": 0.01, "Accuracy": 0.81},
    {"Optimizer": "RAdam", "Batch Size": 32, "Learning Rate": 0.001, "Accuracy": 0.86},
    {"Optimizer": "RAdam", "Batch Size": 64, "Learning Rate": 0.01, "Accuracy": 0.88},
    {"Optimizer": "RAdam", "Batch Size": 64, "Learning Rate": 0.001, "Accuracy": 0.91},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Convert categorical columns to numeric for plotting
df['Optimizer'] = df['Optimizer'].astype('category').cat.codes

# Create the parallel coordinates plot
fig = px.parallel_coordinates(
    df,
    dimensions=["Optimizer", "Batch Size", "Learning Rate", "Accuracy"],
    color="Accuracy",
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={
        "Optimizer": "Optimizer (Encoded)",
        "Batch Size": "Batch Size",
        "Learning Rate": "Learning Rate",
        "Accuracy": "Accuracy",
    },
)

# Show the plot
fig.show()
