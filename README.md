# 🧠 MNIST Handwritten Digit Classification

This project implements **Deep Neural Networks (DNNs)** to classify handwritten digits from the **MNIST dataset** using TensorFlow/Keras. Three different models with varying architectures, activation functions, optimizers, and hyperparameters are trained and evaluated.

## 📌 Project Overview
- **Dataset**: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Goal**: Classify digits (0-9) using deep learning models.
- **Models**: Three different architectures were tested with variations in:
  - Number of layers & neurons
  - Activation functions
  - Optimizers & learning rates
  - Batch sizes & epochs
  - Early stopping for regularization

## 🚀 Installation & Setup
To run this project locally, install the required dependencies:

```bash
pip install tensorflow numpy pandas matplotlib jupyter

Clone this repository and navigate to the project folder:

```bash
git clone https://github.com/your-username/mnist-dnn.git
cd mnist-dnn

Launch Jupyter Notebook:

```bash
jupyter notebook

## 📊 Model Architectures & Results
| Model   | Hidden Layers | Neurons per Layer   | Activation | Optimizer           | Batch Size | Test Accuracy (%) |
|---------|--------------|---------------------|------------|----------------------|------------|-------------------|
| Model 1 | 1            | [64]                | ReLU       | SGD (lr=0.01)       | 32         | **96.11**         |
| Model 2 | 2            | [128, 64]           | Tanh       | Adam (lr=0.001)     | 64         | **97.32**         |
| Model 3 | 3            | [256, 128, 64]      | ELU        | RMSprop (lr=0.001)  | 128        | **97.70**         |


## 🔑 Key Observations
More layers & neurons → Higher accuracy 🚀
ReLU (Model 1) underperformed compared to Tanh & ELU
Adam & RMSprop optimizers performed better than SGD
Larger batch sizes (128 in Model 3) didn't hurt accuracy
Early stopping prevented overfitting effectively

## 📌 How to Run the Models
You can train all three models using the mnist_dnn.ipynb Jupyter Notebook. Below is a snippet for Model 3, the best-performing model:
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

```python
# Model 3: 3 Hidden Layers, ELU Activation
model3 = models.Sequential([
    layers.Dense(256, activation='elu', input_shape=(784,)),
    layers.Dense(128, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(10, activation='softmax')
])

# Compile Model
model3.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping
early_stop3 = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
history3 = model3.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=20,
    callbacks=[early_stop3],
    verbose=1
)

# Evaluate on Test Set
_, acc = model3.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

## 📈 Performance Visualization
The following code generates a bar chart comparing model performance:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Model Results
results = {
    "Model": ["Model 1", "Model 2", "Model 3"],
    "Test Accuracy (%)": [96.11, 97.32, 97.70]
}

df_results = pd.DataFrame(results)

# Plot Performance
plt.figure(figsize=(8,5))
plt.bar(df_results["Model"], df_results["Test Accuracy (%)"], color=['blue', 'green', 'red'])
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("Comparison of MNIST Model Performance")
plt.ylim(95, 98)
plt.show()
