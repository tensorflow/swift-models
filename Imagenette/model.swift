import TensorFlow

struct BasicCNNModel: Layer {
    var conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 32), padding: .same, activation: relu)
    var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 32), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout1 = Dropout<Float>(probability: 0.25)
    var conv2a = Conv2D<Float>(filterShape: (3, 3, 32, 64), padding: .same, activation: relu)
    var conv2b = Conv2D<Float>(filterShape: (3, 3, 64, 64), activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout2 = Dropout<Float>(probability: 0.25)
    var conv3a = Conv2D<Float>(filterShape: (3, 3, 64, 128), padding: .same, activation: relu)
    var conv3b = Conv2D<Float>(filterShape: (3, 3, 128, 128), activation: relu)
    var pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout3 = Dropout<Float>(probability: 0.25)
    var conv4a = Conv2D<Float>(filterShape: (3, 3, 128, 256), padding: .same, activation: relu)
    var conv4b = Conv2D<Float>(filterShape: (3, 3, 256, 256), activation: relu)
    var pool4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout4 = Dropout<Float>(probability: 0.25)
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 9216, outputSize: 512, activation: relu)
    var dropoutFinal = Dropout<Float>(probability: 0.5)
    var classifier = Dense<Float>(inputSize: 512, outputSize: 10, activation: identity)

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let layer1 = input.sequenced(through: conv1a, conv1b, pool1, dropout1)
        let layer2 = layer1.sequenced(through: conv2a, conv2b, pool2, dropout2)
        let layer3 = layer2.sequenced(through: conv3a, conv3b, pool3, dropout3)
        let layer4 = layer3.sequenced(through: conv4a, conv4b, pool4, dropout4)
        return layer4.sequenced(through: flatten, dense1, dropoutFinal, classifier)
    }
}
