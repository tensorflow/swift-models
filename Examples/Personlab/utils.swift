import TensorFlow

func h(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0, 0]) \(tensor[0, -1, 1, 1]), \(tensor[0, 1, -1, 0]), \(tensor[0, -1, -1, -1])]")
}


struct Config {
    let inputSize = (height:241, width: 289)
    let testImagePath = "/Users/joaqo/swift-models/pose.jpg"
}


