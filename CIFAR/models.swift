import TensorFlow

// Ported from pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
public struct PyTorchModel: Layer {
    var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 6), activation: relu)
    var pool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 16 * 5 * 5, outputSize: 120, activation: relu)
    var dense2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    var dense3 = Dense<Float>(inputSize: 84, outputSize: 10, activation: identity)

    @differentiable
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = input
        tmp = conv1.applied(to: tmp, in: context)
        tmp = pool.applied(to: tmp, in: context)
        tmp = conv2.applied(to: tmp, in: context)
        tmp = pool.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        tmp = dense1.applied(to: tmp, in: context)
        tmp = dense2.applied(to: tmp, in: context)
        tmp = dense3.applied(to: tmp, in: context)
        return tmp
        // blocked by TF-340
        // let convolved = input.sequenced(in: context, through: conv1, pool, conv2, pool)
        // return convolved.sequenced(in: context, through: flatten, dense1, dense2, dense3)
    }
}

// Ported from github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
public struct KerasModel: Layer {
    var conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 32), padding: .same, activation: relu)
    var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 32), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout1 = Dropout<Float>(probability: 0.25)
    var conv2a = Conv2D<Float>(filterShape: (3, 3, 32, 64), padding: .same, activation: relu)
    var conv2b = Conv2D<Float>(filterShape: (3, 3, 64, 64), activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var dropout2 = Dropout<Float>(probability: 0.25)
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 64 * 6 * 6, outputSize: 512, activation: relu)
    var dropout3 = Dropout<Float>(probability: 0.5)
    var dense2 = Dense<Float>(inputSize: 512, outputSize: 10, activation: identity)

    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = input
        tmp = conv1a.applied(to: tmp, in: context)
        tmp = conv1b.applied(to: tmp, in: context)
        tmp = pool1.applied(to: tmp, in: context)
        tmp = dropout1.applied(to: tmp, in: context)
        tmp = conv2a.applied(to: tmp, in: context)
        tmp = conv2b.applied(to: tmp, in: context)
        tmp = pool2.applied(to: tmp, in: context)
        tmp = dropout2.applied(to: tmp, in: context)
        tmp = flatten.applied(to: tmp, in: context)
        tmp = dense1.applied(to: tmp, in: context)
        tmp = dropout3.applied(to: tmp, in: context)
        tmp = dense2.applied(to: tmp, in: context)
        return tmp
        // blocked by TF-340
        // let conv1 = input.sequenced(in: context, through: conv1a, conv1b, pool1, dropout1)
        // let conv2 = conv1.sequenced(in: context, through: conv2a, conv2b, pool2, dropout2)
        // return conv2.sequenced(in: context, through: flatten, dense1, dropout3, dense2)
    }
}
