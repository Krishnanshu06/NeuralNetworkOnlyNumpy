from dataset.data import *
from model import *

# print(y_train.shape)

layer1 = Layer(784 , 50)
activation1 = ReLU()
layer2 = Layer(50 , 10)
loss_activation = CombinedLossSoftmax()

# optimizer = Optimizer_GD(decay=1e-4 , momentum=0.3)
optimizer = Optimizer_Adam(learningRate= 0.01 , decay= 1e-5)




def data_generator(X, y, batch_size):
    n_samples = X.shape[0]
    while True:  
        indices = np.random.permutation(n_samples) 
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]  

batch_size = 128
gen = data_generator(x_train, y_train, batch_size)


for epoch in range(10001):

    batch_X, batch_y = next(gen)
     
    layer1.passForward(batch_X)
    activation1.passForward(layer1.LayerOutput)
    layer2.passForward(activation1.output)
    loss = loss_activation.passForward(layer2.LayerOutput , batch_y)

    predictions = np.argmax(loss_activation.SoftmaxOutput , axis = 1)

    accuracy = np.mean(predictions == batch_y)

    if epoch % 1 == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')
        
    if loss <= 0.07:
        np.save(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer1Weights.npy' , layer1.weights)
        np.save(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer2Weights.npy' , layer2.weights)
        np.save(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer1biases.npy' , layer1.biases)
        np.save(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer2biases.npy' , layer2.biases)

        break
        


    loss_activation.passBackward(loss_activation.SoftmaxOutput , batch_y)
    layer2.passBackward(loss_activation.dLoss_dInputs)
    activation1.passBackward(layer2.dLoss_dInputs)
    layer1.passBackward(activation1.dLoss_dInputs)

    optimizer.preUpdateParams()
    optimizer.updateParams(layer1)
    optimizer.updateParams(layer2)
    optimizer.endUpdate()

print("testing")

layer1.passForward(x_test)
activation1.passForward(layer1.LayerOutput)
layer2.passForward(activation1.output)
loss = loss_activation.passForward(layer2.LayerOutput , y_test)

predictions = np.argmax(loss_activation.SoftmaxOutput , axis = 1)

accuracy = np.mean(predictions == y_test)

print(f'acc: {accuracy:.3f}, 'f'loss: {loss:.3f}')
        




