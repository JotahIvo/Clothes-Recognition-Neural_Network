# Fashion MNIST Image Classifier

A neural network implemented from scratch in Python (using only NumPy) to classify images of clothing items from the Fashion MNIST dataset.

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)   
4. [Data Preparation](#data-preparation)  
5. [Usage](#usage)  
   - [Download Dataset](#download-dataset)  
   - [Prepare Data](#prepare-data)  
   - [Train Model](#train-model)  
   - [Evaluate Model](#evaluate-model)  
   - [Inference](#inference)  
6. [Contributing](#contributing)   

## Features

- **Data download & extraction**: script to fetch and unzip the Fashion MNIST images.  
- **Data preprocessing**: load images into NumPy arrays, normalize pixel values, one-hot encode labels.  
- **Custom neural network**: feedforward architecture built from scratch (`Model` class, layers, activations, optimizer, loss).  
- **Accuracy metrics**: `Accuracy_Categorical` and `Accuracy_Regression` classes for tracking performance.  
- **Model persistence**: save and load trained models to/from disk.  
- **Inference example**: example script showing how to make predictions and map labels to human-readable names.  

## Prerequisites

- Python 3.7 or higher  
- [NumPy](https://numpy.org/)  

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/JotahIvo/Clothes-Recognition-Neural_Network.git
   cd Clothes-Recognition-Neural_Network
   ```

2. **(Optional) Create & activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Download Dataset

Run the download script to fetch and extract the images:

```bash
python data_preprocessing/download_data.py
```

This will:

- Download `fashion_mnist_images.zip` from `https://nnfs.io/datasets/fashion_mnist_images.zip`.  
- Unzip its contents into `fashion_mnist_images/`.

### Prepare Data

The module `data_processing/data_preparation.py` reads the extracted image files and produces four NumPy arrays:

- `X_train`, `y_train`  
- `X_test`,  `y_test`  

It also normalizes pixel values (0–255 → 0.0–1.0) and one-hot encodes classification labels where appropriate.

## Usage

### Train Model

Use the `Model` class in `Neural_Network_Classes/nn_model.py` to build, compile, and train your network:

```python
from Neural_Network_Classes.nn_model import Model
from data_processing.data_preparation import X_train, y_train, X_test, y_test

# Initialize & compile
model = Model()
model.compile()

# Train for 10 epochs with a batch size of 64
model.train(X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64)

# Save the trained model to disk
model.save('fashion_mnist.model')
```

### Evaluate Model

The `Neural_Network_Classes/accuracy.py` file provides tools to measure performance:

```python
from metrics.accuracy import Accuracy_Categorical

acc = Accuracy_Categorical()
acc.new_pass()

# After each batch or epoch:
batch_acc = acc.calculate(predictions, y_true)
print(f'Batch accuracy: {batch_acc:.2%}')

# To get overall (accumulated) accuracy:
epoch_acc = acc.calculate_accumulated()
print(f'Epoch accuracy: {epoch_acc:.2%}')
```

### Inference

An example of loading the saved model and predicting labels on test images is provided in `predicting_in_test_data.py`:

```bash
python examples/predict.py
```

This script will:

1. Load `fashion_mnist.model`.  
2. Predict class confidences on the first 5 test samples.  
3. Convert confidences to label indices.  
4. Map indices to human-readable labels:

   ```python
   fashion_mnist_labels = {
       0: 'T-shirt/top', 1: 'Trouser',   2: 'Pullover',
       3: 'Dress',       4: 'Coat',      5: 'Sandal',
       6: 'Shirt',       7: 'Sneaker',   8: 'Bag',
       9: 'Ankle boot'
   }
   ```

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and open a pull request.
