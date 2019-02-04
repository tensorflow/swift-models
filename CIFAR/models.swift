import TensorFlow

// Ported from pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
public struct PyTorchModel : Layer {
    var conv1: Conv2D<Float>
    var pool: MaxPool2D<Float>
    var conv2: Conv2D<Float>
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    var dense3: Dense<Float>
    public init() {
        conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 6), padding: .valid)
        pool = MaxPool2D<Float>(
            poolSize: (2, 2), strides: (2, 2), padding: .valid)
        conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), padding: .valid)
        dense1 = Dense<Float>(
            inputSize: 16 * 5 * 5, outputSize: 120, activation: relu)
        dense2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
        dense3 = Dense<Float>(inputSize: 84, outputSize: 10, activation: { $0 })
    }
    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        var tmp = input
        tmp = pool.applied(to: relu(conv1.applied(to: tmp)))
        tmp = pool.applied(to: relu(conv2.applied(to: tmp)))
        let batchSize = tmp.shape[0]
        tmp = tmp.reshaped(
            toShape: Tensor<Int32>([batchSize, Int32(16 * 5 * 5)]))
        tmp = dense1.applied(to: tmp)
        tmp = dense2.applied(to: tmp)
        tmp = dense3.applied(to: tmp)
        return tmp
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
    var dense1: Dense<Float>
    var dropout3: Dropout<Float>
    var dense2: Dense<Float>
    public init(learningPhaseIndicator: LearningPhaseIndicator) {
        conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 32), padding: .same)
        conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 32), padding: .valid)
        pool1 = MaxPool2D<Float>(
            poolSize: (2, 2), strides: (2, 2), padding: .valid)
        dropout1 = Dropout<Float>(
            probability: 0.25, learningPhaseIndicator: learningPhaseIndicator)
        conv2a = Conv2D<Float>(filterShape: (3, 3, 32, 64), padding: .same)
        conv2b = Conv2D<Float>(filterShape: (3, 3, 64, 64), padding: .same)
        pool2 = MaxPool2D<Float>(
            poolSize: (2, 2), strides: (2, 2), padding: .valid)
        dropout2 = Dropout<Float>(
            probability: 0.25, learningPhaseIndicator: learningPhaseIndicator)
        dense1 = Dense<Float>(
            inputSize: 64 * 7 * 7, outputSize: 512, activation: relu)
        dropout3 = Dropout<Float>(
            probability: 0.5, learningPhaseIndicator: learningPhaseIndicator)
        dense2 = Dense<Float>(
            inputSize: 512, outputSize: 10, activation: { $0 })
    }
    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        var tmp = input
        tmp = relu(conv1b.applied(to: relu(conv1a.applied(to: tmp))))
        tmp = dropout1.applied(to: pool1.applied(to: tmp))
        tmp = relu(conv2b.applied(to: relu(conv2a.applied(to: tmp))))
        tmp = dropout2.applied(to: pool2.applied(to: tmp))
        let batchSize = tmp.shape[0]
        tmp = tmp.reshaped(
            toShape: Tensor<Int32>([batchSize, Int32(64 * 7 * 7)]))
        tmp = dropout3.applied(to: dense1.applied(to: tmp))
        tmp = dense2.applied(to: tmp)
        return tmp
    }
}

public typealias CIFARModel = PyTorchModel

@differentiable(wrt: model)
public func loss(model: CIFARModel, images: Tensor<Float>, labels: Tensor<Int32>)
    -> Tensor<Float> {
    let logits = model.applied(to: images)
    let oneHotLabels = Tensor<Float>(
        oneHotAtIndices: labels, depth: logits.shape[1])
    return softmaxCrossEntropy(logits: logits, labels: oneHotLabels)
}
