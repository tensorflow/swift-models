import TensorFlow

struct Keypoint {
  let y: Float
  let x: Float
  let index: KeypointIndex
  let score: Float

  init(heatmapY: Int, heatmapX: Int, index: Int, score: Float, offsets: Tensor<Float>) {
    self.y = Float(heatmapY) * Float(config.outputStride) + offsets[heatmapY, heatmapX, index].scalarized()
    self.x = Float(heatmapX) * Float(config.outputStride) + offsets[heatmapY, heatmapX, index + KeypointIndex.allCases.count].scalarized()
    self.index = KeypointIndex(rawValue: index)!
    self.score = score
  }

  init(y: Float, x: Float, index: KeypointIndex, score: Float) {
    self.y = y
    self.x = x
    self.index = index
    self.score = score
  }
}

enum KeypointIndex: Int, CaseIterable {
  case nose = 0
  case leftEye
  case rightEye
  case leftEar
  case rightEar
  case leftShoulder
  case rightShoulder
  case leftElbow
  case rightElbow
  case leftWrist
  case rightWrist
  case leftHip
  case rightHip
  case leftKnee
  case rightKnee
  case leftAnkle
  case rightAnkle
}

// TODO: Add default and remove 4 nil cases
func getSucceedingtKeypointIndex(_ keypointId: KeypointIndex) -> [KeypointIndex?] {
  switch keypointId {
  case .nose: return [.leftShoulder, .rightShoulder]
  case .leftEye: return [.nose]
  case .rightEye: return [.nose]
  case .leftEar: return [.leftEye]
  case .rightEar: return [.rightEye]
  case .leftShoulder: return [.leftHip, .leftElbow]
  case .rightShoulder: return [.rightHip, .rightElbow]
  case .leftElbow: return [.leftWrist]
  case .rightElbow: return [.rightWrist]
  case .leftWrist: return [nil]
  case .rightWrist: return [nil]
  case .leftHip: return [.leftKnee]
  case .rightHip: return [.rightKnee]
  case .leftKnee: return [.leftAnkle]
  case .rightKnee: return [.rightAnkle]
  case .leftAnkle: return [nil]
  case .rightAnkle: return [nil]
  }
}
  
func getPrecedingKeypointIndex(_ keypointId: KeypointIndex) -> [KeypointIndex?] {
  switch keypointId {
  case .nose: return [.leftEye, .rightEye]
  case .leftEye: return [.leftEar]
  case .rightEye: return [.rightEar]
  case .leftEar: return [nil]
  case .rightEar: return [nil]
  case .leftShoulder: return [.nose]
  case .rightShoulder: return [.nose]
  case .leftElbow: return [.leftShoulder]
  case .rightElbow: return [.rightShoulder]
  case .leftWrist: return [.leftElbow]
  case .rightWrist: return [.rightElbow]
  case .leftHip: return [.leftShoulder]
  case .rightHip: return [.rightShoulder]
  case .leftKnee: return [.leftHip]
  case .rightKnee: return [.rightHip]
  case .leftAnkle: return [.leftKnee]
  case .rightAnkle: return [.rightKnee]
  }
}

// NOTE: Excelent type inference swift holy shit!
/// Maps a pair of keypoint indexes to the appropiate index to be used
/// to get the displacement in the displacement tensors.
let keypointPairToDisplacementIndexMap: [Set<KeypointIndex>: Int] = [
  Set([.nose, .leftEye]): 0,
  Set([.leftEye, .leftEar]): 1,
  Set([.nose, .rightEye]): 2,
  Set([.rightEye, .rightEar]): 3,
  Set([.nose, .leftShoulder]): 4,
  Set([.leftShoulder, .leftElbow]): 5,
  Set([.leftElbow, .leftWrist]): 6,
  Set([.leftShoulder, .leftHip]): 7,
  Set([.leftHip, .leftKnee]): 8,
  Set([.leftKnee, .leftAnkle]): 9,
  Set([.nose, .rightShoulder]): 10,
  Set([.rightShoulder, .rightElbow]): 11,
  Set([.rightElbow, .rightWrist]): 12,
  Set([.rightShoulder, .rightHip]): 13,
  Set([.rightHip, .rightKnee]): 14,
  Set([.rightKnee, .rightAnkle]): 15
]

// let previousJointsMap: [[KeypointId?]] = [
//   [.leftEye, .rightEye],  // nose
//   [.leftEar],  //  leftEye
//   [.rightEar],  //rightEye
//   [nil],  // leftEar
//   [nil],  // rightEar
//   [.nose],  // leftShoulder
//   [.nose],  // rightShoulder
//   [.leftShoulder],  // leftElbow
//   [.rightShoulder],  // rightElbow
//   [.leftElbow],  // leftWrist
//   [.rightElbow],  // rightWrist
//   [.leftShoulder],  // leftHip
//   [.rightShoulder],  // rightHip
//   [.leftHip],  // leftKnee
//   [.rightHip],  // rightKnee
//   [.leftKnee],  // leftAngle
//   [.rightKnee]  // rightAnkle
// ]

// let nextJointsMap: [[KeypointId?]] = [
//   [.leftShoulder, .rightShoulder],  // nose
//   [.nose],  //  leftEye
//   [.nose],  //rightEye
//   [.leftEye],  // leftEar
//   [.rightEye],  // rightEar
//   [.leftHip, .leftElbow],  // leftShoulder
//   [.rightHip, .rightElbow],  // rightShoulder
//   [.leftWrist],  // leftElbow
//   [.rightWrist],  // rightElbow
//   [nil],  // leftWrist
//   [nil],  // rightWrist
//   [.leftKnee],  // leftHip
//   [.rightKnee],  // rightHip
//   [.leftAnkle],  // leftKnee
//   [.rightAnkle],  // rightKnee
//   [nil],  // leftAngle
//   [nil]  // rightAnkle
// ]

// // TODO: Check performance difference vs just using an Array with enums with
// // associated values.
// // Using a struct should be faster as we are stack allocated, but still,
// // I want to measure the difference.
// struct pose {
//   let nose: Keypoint
//   let lefteye: Keypoint
//   let righeye: Keypoint
//   let leftear: Keypoint
//   let rightear: Keypoint
//   let leftshoulder: Keypoint
//   let rightshoulder: Keypoint
//   let leftelbow: Keypoint
//   let rightelbow: Keypoint
//   let leftwrist: Keypoint
//   let rightwrist: Keypoint
//   let lefthip: Keypoint
//   let righthip: Keypoint
//   let leftknee: Keypoint
//   let rightknee: Keypoint
//   let leftankle: Keypoint
//   let rightankle: Keypoint
// }


/// The final object the model returns.
/// Implementation just wraps a list, but doing this defines intent more clearly 
/// than just using a random list of keypoints through the code base.
struct Pose  {
  var pose: [Keypoint?] = Array(repeating: nil, count: KeypointIndex.allCases.count)

  mutating func add(_ keypoint: Keypoint) {
    pose[keypoint.index.rawValue] = keypoint
  }

  func getKeypoint(with index: KeypointIndex) -> Keypoint? {
    return pose[index.rawValue]
  }
}
