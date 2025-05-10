from Neural_Network_Classes.nn_model import Model
from Neural_Network_Classes.layers import Layer_Dense
from Neural_Network_Classes.activation_functions import Activation_ReLU, Activation_Softmax
from Neural_Network_Classes.loss_functions import Loss_CategoricalCrossentropy
from Neural_Network_Classes.optimizers import Optimizer_Adam
from Neural_Network_Classes.accuracy_functions import Accuracy_Categorical
from data_processing.data_preparation import X, y, X_test, y_test


model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

model.save('fashion_mnist.model')
