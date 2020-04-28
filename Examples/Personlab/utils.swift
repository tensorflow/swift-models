import TensorFlow
import ModelSupport


struct Config {
    // Image
    let inputImageSize = (height:241, width: 289)
    let testImagePath = "/Users/joaqo/swift-models/pose.jpg"

    // Decoder
    let outputStride = 16
    let poseScoreThreshold: Float = 0.15
    let keypointScoreThreshold: Float = 0.1
    let nmsRadius: Float = 20.0
    let keypointLocalMaximumRadius = 1  // TODO: what? they use 1 in implementation but not in paper
    
}


extension CheckpointReader {
    func load(from name: String) -> Tensor<Float> {
        return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
    }
}


/// Usef for copying model from Python Tensorflow 1.5 version
func hash(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0]) \(tensor[0, -1, 1]), \(tensor[0, 1, 0]), \(tensor[0, -1, -1])]")
}
