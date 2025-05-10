import os
import cv2
import numpy as np 
import nnfs

nnfs.init()

def load_mnist_dataset(dataset, path):
    dataset_path = os.path.join(path, dataset)
    labels = sorted(os.listdir(dataset_path), key=lambda x: int(x))

    X, y = [], []

    for label in labels:
        label_folder = os.path.join(dataset_path, label)
        for file in sorted(os.listdir(label_folder)):
            file_path = os.path.join(label_folder, file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue  
            X.append(image)
            y.append(int(label))

    X = np.array(X)
    y = np.array(y, dtype='uint8')
    return X, y

def create_data_mnist(path):
    X, y       = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test',  path)
    return X, y, X_test, y_test

# Data preprocessing
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Data Shuffling
keys = np.arange(X.shape[0])
np.random.shuffle(keys)

X = X[keys]
y = y[keys]
