from Neural_Network_Classes.nn_model import Model
from data_processing.data_preparation import X_test, y_test


model = Model.load('fashion_mnist.model')

confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)

print(y_test[:5])

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

for prediction in predictions:
    print(fashion_mnist_labels[prediction])
