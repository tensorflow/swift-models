import TensorFlow

// Ported from pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
public struct PyTorchModel : Layer {
    var conv1: Conv2D<Float>
    var pool: MaxPool2D<Float>
    var conv2: Conv2D<Float>
    var flatten: Flatten<Float>
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    var dense3: Dense<Float>
    public init() {
        conv1 = Conv2D(filterShape: (5, 5, 3, 6), activation: relu)
        pool = MaxPool2D(poolSize: (2, 2), strides: (2, 2))
        conv2 = Conv2D(filterShape: (5, 5, 6, 16), activation: relu)
        flatten = Flatten()
        dense1 = Dense(inputSize: 16 * 5 * 5, outputSize: 120, activation: relu)
        dense2 = Dense(inputSize: 120, outputSize: 84, activation: relu)
        dense3 = Dense(inputSize: 84, outputSize: 10, activation: identity)
    }
    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let convolved = input.sequenced(in: context, through: conv1, pool, conv2, pool)
        return convolved.sequenced(in: context, through: flatten, dense1, dense2, dense3)
    }
}

// Ported from github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
public struct KerasModel : Layer {
    var conv1a: Conv2D<Float>
    var conv1b: Conv2D<Float>
    var pool1: MaxPool2D<Float>
    var dropout1: Dropout<Float>
    var conv2a: Conv2D<Float>
    var conv2b: Conv2D<Float>
    var pool2: MaxPool2D<Float>
    var dropout2: Dropout<Float>
    var flatten: Flatten<Float>
    var dense1: Dense<Float>
    var dropout3: Dropout<Float>
    var dense2: Dense<Float>
    public init() {
        conv1a = Conv2D(filterShape: (3, 3, 3, 32), padding: .same, activation: relu)
        conv1b = Conv2D(filterShape: (3, 3, 32, 32),activation: relu)
        pool1 = MaxPool2D(poolSize: (2, 2), strides: (2, 2))
        dropout1 = Dropout(probability: 0.25)
        conv2a = Conv2D(filterShape: (3, 3, 32, 64), padding: .same, activation: relu)
        conv2b = Conv2D(filterShape: (3, 3, 64, 64), activation: relu)
        pool2 = MaxPool2D(poolSize: (2, 2), strides: (2, 2))
        dropout2 = Dropout(probability: 0.25)
        flatten = Flatten()
        dense1 = Dense(inputSize: 64 * 6 * 6, outputSize: 512, activation: relu)
        dropout3 = Dropout(probability: 0.5)
        dense2 = Dense(inputSize: 512, outputSize: 10, activation: identity)
    }
    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let conv1 = input.sequenced(in: context, through: conv1a, conv1b, pool1, dropout1)
        let conv2 = conv1.sequenced(in: context, through: conv2a, conv2b, pool2, dropout2)
        return conv2.sequenced(in: context, through: flatten, dense1, dropout3, dense2)
    }
}
