import TensorFlow
import ResNet

func loss(model: ResNet50, images: Tensor<Float>, labels: Tensor<Int32>) {
    let logits = model.applied(to: images)
    let oneHotLabels = Tensor<Float>(
        oneHotAtIndices: labels, depth: logits.shape.last)
    return softmaxCrossEntropy(logits: logits, labels: oneHotLabels)
}

let batchSize: Int32 = 128
let classCount = 1000

let fakeImageBatch = Tensor<Float>(zeros: [batchSize, 224, 224, 3])
let fakeLabelBatch = Tensor<Int32>(zeros: [batchSize])

let modeRef = ModeRef()
let resnet = ResNet50(classCount: classCount, modeRef: modeRef)
let optimizer = SGD<ResNet50, Float>(learningRate: 0.1, momentum: 0.9)

for iteration in 0..<10 {
    let gradients = gradient(at: resnet) {
      loss(model: resnet, images, fakeImageBatch, labels: fakeLabelBatch)
    }
    optimizer.update(resnet.allDifferentiableVariables, along: gradients)
}
