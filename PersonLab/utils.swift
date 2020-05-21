import TensorFlow
import ModelSupport
import SwiftCV


public struct Config {
    let checkpointPath: String
    let printProfilingData: Bool
    let inputImageSize = (height: 241, width: 289)

    // Decoder
    let outputStride = 16
    let poseScoreThreshold: Float = 0.15
    let keypointScoreThreshold: Float = 0.1
    let nmsRadius: Float = 20.0
    let keypointLocalMaximumRadius = 1

    // Drawing
    let color = Scalar(val1: 0, val2: 255, val3: 255, val4: 1)
    let lineWidth: Int32 = 2
}


extension CheckpointReader {
    func load(from name: String) -> Tensor<Float> {
        return Tensor(self.loadTensor(named: "MobilenetV1/\(name)"))
    }
}


func draw(_ pose: Pose, on image: Mat, color: Scalar, lineWidth: Int32) {
  var pose = pose
  pose.rescale(to: (height: image.rows, width: image.cols))

  func recursivellyDrawNextKeypoint(after previousKeypoint: Keypoint, into image: Mat) {
    for (nextKeypointIndex, direction) in getNextKeypointIndexAndDirection(previousKeypoint.index) {
      if direction == .fwd {
        if let nextKeypoint = pose.getKeypoint(nextKeypointIndex) {
          let point1 = Point(x: Int32(previousKeypoint.x), y: Int32(previousKeypoint.y))
          let point2 = Point(x: Int32(nextKeypoint.x), y: Int32(nextKeypoint.y))
          line(img: image, pt1: point1, pt2: point2, color: color, thickness: lineWidth)
          recursivellyDrawNextKeypoint(after: nextKeypoint, into: image)
        }
      }
    }
  }

  recursivellyDrawNextKeypoint(after: pose.getKeypoint(.nose)!, into: image)
}


/// Used as an ad-hoc "hash" for tensor checking when copying the backbone from
/// our Python Tensorflow 1.5 version
func hash(_ tensor: Tensor<Float>) {
    print("[\(tensor.flattened().sum()), \(tensor[0, 0, 0]) \(tensor[0, -1, 1]), \(tensor[0, 1, 0]), \(tensor[0, -1, -1])]")
}


/// Wrapper for Tensor which allows several order of magnitude faster subscript access,
/// as it avoids unnecesary GPU->CPU copies on each access.
struct CPUTensor<T: TensorFlowScalar> {
  private var flattenedTensor: [T]
  var shape: TensorShape

  init(_ tensor: Tensor<T>) {
    self.flattenedTensor = tensor.scalars
    self.shape = tensor.shape
  }

  subscript(indexes: Int...) -> T {
    var oneDimensionalIndex = 0
    for i in 1..<shape.count {
      oneDimensionalIndex += indexes[i - 1] * shape[i...].reduce(1, *)
    }
    // Last dimension doesn't have multipliers.
    oneDimensionalIndex += indexes.last!
    return flattenedTensor[oneDimensionalIndex]
  }
}
