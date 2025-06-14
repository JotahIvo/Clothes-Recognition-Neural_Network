import cv2
import numpy as np
from Neural_Network_Classes.nn_model import Model


fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

image_data = cv2.imread('images/pants.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5


model = Model.load('fashion_mnist.model')

confidences = model.predict(image_data)

predictions = model.output_layer_activation.predictions(confidences)

prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
