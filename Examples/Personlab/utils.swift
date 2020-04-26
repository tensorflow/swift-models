import TensorFlow
import ModelSupport


struct Config {
    // Image
    let inputImageSize = (height:241, width: 289)
    let testImagePath = "/Users/joaqo/swift-models/pose.jpg"

    // Decoder
    let outputStride = 16
    let personScoreThreshold: Float = 0.15
    let keypointScoreThreshold: Float = 0.15
    let nmsRadius = 20
    let keypointLocalMaximumRadius = 1
    
}


extension CheckpointReader {
    func load(from name: String) -> Tensor<Float> {
        return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
    }
}


/// Usef for copying model from Python Tensorflow 1.5 version
func hash(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0, 0]) \(tensor[0, -1, 1, 1]), \(tensor[0, 1, -1, 0]), \(tensor[0, -1, -1, -1])]")
}
