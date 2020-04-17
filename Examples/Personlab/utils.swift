import TensorFlow
import ModelSupport


struct Config {
    let inputSize = (height:241, width: 289)
    let testImagePath = "/Users/joaqo/swift-models/pose.jpg"
}


extension CheckpointReader {
    func load(from name: String) -> Tensor<Float> {
        return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
    }
}


/// Usef for copying model from Tensorflow1.5
func hash(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0, 0]) \(tensor[0, -1, 1, 1]), \(tensor[0, 1, -1, 0]), \(tensor[0, -1, -1, -1])]")
}
