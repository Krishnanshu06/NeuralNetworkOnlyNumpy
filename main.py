from dataset.data import *
from model import *

layer1 = Layer(784 , 50)
activation1 = ReLU()
layer2 = Layer(50 , 10)
loss_activation = CombinedLossSoftmax()
optimizer = Optimizer_GD(decay=1e-4 , momentum=0.3)

layer1.weights = np.load(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer1Weights.npy')
layer1.biases = np.load(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer1biases.npy')
layer2.weights = np.load(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer2Weights.npy')
layer2.biases = np.load(r'Ai\MachineLearning\DigitRecognition\ModelParams\layer2biases.npy')


def CheckDigit(inputData):

    tempVal = np.array([[5]])    # temp val 5, the nn requires a (1,1) dimensional array to work but this has no contribution in the output
    layer1.passForward(inputData)
    activation1.passForward(layer1.LayerOutput)
    layer2.passForward(activation1.output)
    loss = loss_activation.passForward(layer2.LayerOutput , tempVal)


    predictions = np.argmax(loss_activation.SoftmaxOutput , axis = 1)
    print(loss_activation.SoftmaxOutput)
    print(predictions)
    return(predictions[0])

# data = np.load('input.npy')

# CheckDigit(data)
