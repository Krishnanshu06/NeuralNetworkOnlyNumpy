import numpy as np




class Layer:
    def __init__(self , noInputs , noNeurons):   #Generate Random Weights and biases for intializing.
        
        self.weights = 0.01 * np.random.randn(noInputs,noNeurons)
        self.biases = np.zeros((1,noNeurons))
    
    def passForward(self, inputs):     #Forward pass -- Output is the dot prod of inputs and weights + the biases

        self.LayerInput = inputs
        self.LayerOutput = np.dot(inputs,self.weights) + self.biases

    def passBackward(self , dLoss_dOut):                                            #1
        
        self.dLoss_dWeights = np.dot((self.LayerInput).T , dLoss_dOut)
        self.dLoss_dBias = np.sum(dLoss_dOut , axis = 0 , keepdims= True)

        self.dLoss_dInputs = np.dot(dLoss_dOut , (self.weights).T)                  #2

class ReLU:

    def passForward(self , inputs):
        
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def passBackward(self , dLoss_dOut):  #dLoss_dOut is the pd of the loss wrt output of the ReLU

        self.dLoss_dInputs = dLoss_dOut.copy()
        self.dLoss_dInputs[self.inputs <= 0] = 0   #derivative of relu for the negetive terms is 0

class Softmax:

    def passForward(self , inputs):
        
        exp_inputs = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        probablities = exp_inputs / np.sum(exp_inputs , axis=1 , keepdims=True)

        self.output = probablities

    # dont define the Backward pass now as its derivative is very complex

class Loss:

    def calculate(self, predictions , actualOutputs):
        
        sampleLoss = self.passForward(predictions , actualOutputs) # loss of the enlire sample seperately in a array

        finalLoss = np.mean(sampleLoss) # average of the losses for one batch
        return finalLoss

class CategoricalCrossEntropy_Loss(Loss):

    def passForward(self , y_pred , y_true):
        
        nSamples = len(y_pred)   # no of samples in a batch
 
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1 - 1e-7)  # to avoid the log fn to give inf value at log(0)

        # if catogorial lables (as we have in out nnfs dataset)
        if len(y_true.shape) == 1: #check if 1 dimension or 2 
            correct_guesses = y_pred_clipped [ range(nSamples) ,  y_true ]

        #if one Hot encoded
        elif len(y_true.shape) == 2:
            correct_guesses = np.sum(y_pred_clipped * y_true , axis=1)

        Loss = -np.log(correct_guesses)
        return Loss

    def passBackward(self, dLoss_dOut, y_true):

        # Number of samples
        samples = len(dLoss_dOut)

        # Number of labels in every sample
        labels = len(dLoss_dOut[0])

        # If labels are catagorical, turn them into one-hot encoded
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient 
        self.dinputs = -y_true / dLoss_dOut

        # Normalize gradient
        self.dinputs = self.dinputs / samples
 
class CombinedLossSoftmax:

    def __init__(self):
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy_Loss()

    def passForward(self , inputs , y_true):

        self.softmax.passForward(inputs)
        self.SoftmaxOutput = self.softmax.output

        return self.loss.calculate(self.SoftmaxOutput,y_true)

    def passBackward(self , dLoss_dOut , y_true):
        
        nSamples = len(dLoss_dOut)
        
        #convert to discrete values if one hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dLoss_dInputs = dLoss_dOut.copy()
        self.dLoss_dInputs[range(nSamples), y_true] -= 1    # the pd of the combined is just y_pred - y_true{which is just 0 or 1}
        self.dLoss_dInputs = self.dLoss_dInputs / nSamples

class Optimizer_GD:

    def __init__(self , learningRate = 1. , decay = 0. , momentum =0.):
        self.learningRate = learningRate
        self.CurrentLR = learningRate
        self.decay = decay
        self.momentum = momentum
        self.nIterations = 0
    
    def preUpdateParams(self):
        if self.decay:
            self.CurrentLR = self.learningRate / (1. + self.decay * self.nIterations)

    def updateParams(self , layer):   #takes the layer to update
        

        if self.momentum:

            if not hasattr(layer , 'weightCache'):
                layer.weightCache = np.zeros_like(layer.weights)
                layer.biasCache = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weightCache - self.CurrentLR * layer.dLoss_dWeights
            bias_updates = self.momentum * layer.biasCache - self.CurrentLR * layer.dLoss_dBias

            layer.weightCache = weight_updates
            layer.biasCache = bias_updates

        else:

            weight_updates = -self.CurrentLR * layer.dLoss_dWeights
            bias_updates =   -self.CurrentLR * layer.dLoss_dBias


        layer.weights += weight_updates
        layer.biases += bias_updates

    def endUpdate(self):
        self.nIterations += 1

class Optimizer_AdaGrad:

    def __init__(self , learningRate = 1. , decay = 0. , epsilon =1e-7):
        self.learningRate = learningRate
        self.CurrentLR = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.nIterations = 0
    
    def preUpdateParams(self):
        if self.decay:
            self.CurrentLR = self.learningRate / (1. + self.decay * self.nIterations)

    def updateParams(self , layer):   #takes the layer to update
        

        if not hasattr(layer , 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)


        layer.weightCache += layer.dLoss_dWeights**2
        layer.biasCache += layer.dLoss_dBias**2

        weight_updates = - (self.CurrentLR * layer.dLoss_dWeights) / (np.sqrt(layer.weightCache) + self.epsilon)
        bias_updates =  - (self.CurrentLR * layer.dLoss_dBias) / (np.sqrt(layer.biasCache) + self.epsilon)
       
        layer.weights += weight_updates
        layer.biases += bias_updates

    def endUpdate(self):
        self.nIterations += 1

class Optimizer_RMSprop:

    def __init__(self , learningRate = 1. , decay = 0. , epsilon =1e-7 , rho = 0.9):
        self.learningRate = learningRate
        self.CurrentLR = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.nIterations = 0
        self.beta2 = rho
    
    def preUpdateParams(self):
        if self.decay:
            self.CurrentLR = self.learningRate / (1. + self.decay * self.nIterations)

    def updateParams(self , layer):   #takes the layer to update
        
        if not hasattr(layer , 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)


        layer.weightCache = self.beta2 * layer.weightCache + (1- self.beta2) * layer.dLoss_dWeights**2
        layer.biasCache = self.beta2 * layer.biasCache + (1- self.beta2) * layer.dLoss_dBias**2

        weight_updates = - (self.CurrentLR * layer.dLoss_dWeights) / (np.sqrt(layer.weightCache) + self.epsilon)
        bias_updates =  - (self.CurrentLR * layer.dLoss_dBias) / (np.sqrt(layer.biasCache) + self.epsilon)
       
        layer.weights += weight_updates
        layer.biases += bias_updates

    def endUpdate(self):
        self.nIterations += 1

class Optimizer_Adam:

    def __init__(self , learningRate = 0.001 , decay = 0. , epsilon =1e-7 , beta1 = 0.9 , beta2 = 0.999):
        self.learningRate = learningRate
        self.CurrentLR = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.nIterations = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def preUpdateParams(self):
        if self.decay:
            self.CurrentLR = self.learningRate / (1. + self.decay * self.nIterations)

    def updateParams(self , layer):   #takes the layer to update
        
        if not hasattr(layer , 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)


        layer.weightCache = self.beta2 * layer.weightCache + (1- self.beta2) * layer.dLoss_dWeights**2
        layer.biasCache = self.beta2 * layer.biasCache + (1- self.beta2) * layer.dLoss_dBias**2

        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dLoss_dWeights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 - self.beta1) * layer.dLoss_dBias

        weightMomentumsCorrected = layer.weightMomentums / (1- self.beta1 ** (self.nIterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1- self.beta1 ** (self.nIterations + 1))
        weightCacheCorrected = layer.weightCache / (1- self.beta2 ** (self.nIterations + 1))
        biasCacheCorrected = layer.biasCache / (1- self.beta2 ** (self.nIterations + 1))



        weight_updates = - (self.CurrentLR * weightMomentumsCorrected) / (np.sqrt(weightCacheCorrected) + self.epsilon)
        bias_updates =  - (self.CurrentLR * biasMomentumsCorrected) / (np.sqrt(biasCacheCorrected) + self.epsilon)
       
        layer.weights += weight_updates
        layer.biases += bias_updates

    def endUpdate(self):
        self.nIterations += 1
    
    
