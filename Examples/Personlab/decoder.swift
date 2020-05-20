import TensorFlow
import Foundation

struct PoseDecoder {
  let heatmap: CPUTensor<Float>
  let offsets: CPUTensor<Float>
  let displacementsFwd: CPUTensor<Float>
  let displacementsBwd: CPUTensor<Float>
  let config: Config

  init(for results: PersonlabHeadsResults, with config: Config) {
    // Remove [0] indexes when we add batchsize > 1 support, which should be really easy
    self.heatmap = CPUTensor<Float>(results.heatmap[0])
    self.offsets = CPUTensor<Float>(results.offsets[0])
    self.displacementsFwd = CPUTensor<Float>(results.displacementsFwd[0])
    self.displacementsBwd = CPUTensor<Float>(results.displacementsBwd[0])
    self.config = config
  }

  func decode() -> [Pose] {
    var poses = [Pose]()
    var sortedLocallyMaximumKeypoints = getSortedLocallyMaximumKeypoints()
    while sortedLocallyMaximumKeypoints.count > 0 {
      let rootKeypoint = sortedLocallyMaximumKeypoints.removeFirst()
      if rootKeypoint.isWithinRadiusOfCorrespondingKeypoints(in: poses, radius: config.nmsRadius) {
        continue
      }

      var pose = Pose(resolution: self.config.inputImageSize)
      pose.add(rootKeypoint)

      // Recursivelly parse keypoint tree going in forward direction
      recursivellyAddNextKeypoint(
        after: rootKeypoint,
        into: &pose
      )

      if getPoseScore(for: pose, considering: poses) > config.poseScoreThreshold {
        poses.append(pose)
      }
    }
    return poses
  }

  func recursivellyAddNextKeypoint(after previousKeypoint: Keypoint, into pose: inout Pose) {
    for (nextKeypointIndex, direction) in getNextKeypointIndexAndDirection(previousKeypoint.index) {
      if pose.getKeypoint(nextKeypointIndex) == nil {
        let nextKeypoint = followDisplacement(
          from: previousKeypoint,
          to: nextKeypointIndex,
          using: direction == .fwd ? displacementsFwd : displacementsBwd
        )
        pose.add(nextKeypoint)
        recursivellyAddNextKeypoint(after: nextKeypoint, into: &pose)
      }
    }
  }

  func followDisplacement(from previousKeypoint: Keypoint, to nextKeypointIndex: KeypointIndex, using displacements: CPUTensor<Float>) -> Keypoint {

    let displacementKeypointIndexY = keypointPairToDisplacementIndexMap[Set([previousKeypoint.index, nextKeypointIndex])]!
    let displacementKeypointIndexX = displacementKeypointIndexY + displacements.shape[2] / 2
    let displacementYIndex = getUnstridedIndex(y: previousKeypoint.y)
    let displacementXIndex = getUnstridedIndex(x: previousKeypoint.x)

    // let startt = CFAbsoluteTimeGetCurrent()
    let displacementY = displacements[
      displacementYIndex,
      displacementXIndex,
      displacementKeypointIndexY
    ]
    let displacementX = displacements[
      displacementYIndex,
      displacementXIndex,
      displacementKeypointIndexX
    ]
    // print(CFAbsoluteTimeGetCurrent() - startt)

    let displacedY = getUnstridedIndex(y: previousKeypoint.y + displacementY)
    let displacedX = getUnstridedIndex(x: previousKeypoint.x + displacementX)

    let yOffset = offsets[
      displacedY,
      displacedX,
      nextKeypointIndex.rawValue
    ]
    let xOffset = offsets[
      displacedY,
      displacedX,
      nextKeypointIndex.rawValue + KeypointIndex.allCases.count
    ]

    // If we are getting the offset from an exact point in the heatmap, we should add this
    // offset parting from that exact point in the heatmap, so we just nearest neighbour
    // interpolate it back, then re strech using output stride, and then add said offset.
    let nextY = Float(displacedY * config.outputStride) + yOffset
    let nextX = Float(displacedX * config.outputStride) + xOffset

    return Keypoint(
      y: nextY,
      x: nextX,
      index: nextKeypointIndex,
      score: heatmap[
        displacedY, displacedX, nextKeypointIndex.rawValue
      ]
    )
  }

  func scoreIsMaximumInLocalWindow(heatmapY: Int, heatmapX: Int, score: Float, keypointIndex: Int) -> Bool {
    let yStart = max(heatmapY - config.keypointLocalMaximumRadius, 0)
    let yEnd = min(heatmapY + config.keypointLocalMaximumRadius, heatmap.shape[0] - 1)
    for windowY in yStart...yEnd {
      let xStart = max(heatmapX - config.keypointLocalMaximumRadius, 0)
      let xEnd = min(heatmapX + config.keypointLocalMaximumRadius, heatmap.shape[1] - 1)
      for windowX in xStart...xEnd {
        if heatmap[windowY, windowX, keypointIndex] > score {
          return false
        }
      }
    }
    return true
  }

  func getUnstridedIndex(y: Float) -> Int {
    let downScaled = y / Float(config.outputStride)
    let clamped = min(max(0, downScaled.rounded()), Float(heatmap.shape[0] - 1))
    return Int(clamped)
  }

  func getUnstridedIndex(x: Float) -> Int {
    let downScaled = x / Float(config.outputStride)
    let clamped = min(max(0, downScaled.rounded()), Float(heatmap.shape[1] - 1))
    return Int(clamped)
  }

  func getSortedLocallyMaximumKeypoints() -> [Keypoint] {
    var sortedLocallyMaximumKeypoints = [Keypoint]()
    for heatmapY in 0..<heatmap.shape[0] {
      for heatmapX in 0..<heatmap.shape[1] {
        for keypointIndex in 0..<heatmap.shape[2] {
          let score = heatmap[heatmapY, heatmapX, keypointIndex]

          if score < config.keypointScoreThreshold { continue }
          if scoreIsMaximumInLocalWindow(
            heatmapY: heatmapY,
            heatmapX: heatmapX,
            score: score,
            keypointIndex: keypointIndex
          )  {
            sortedLocallyMaximumKeypoints.append(
              Keypoint(
                heatmapY: heatmapY,
                heatmapX: heatmapX,
                index: keypointIndex,
                score: score,
                offsets: offsets,
                outputStride: config.outputStride
              )
            )
          }
        }
      }
    }
    sortedLocallyMaximumKeypoints.sort {$0.score > $1.score}
    return sortedLocallyMaximumKeypoints
  }

  func getPoseScore(for pose: Pose, considering poses: [Pose]) -> Float {
    var notOverlappedKeypointScoreAccumulator: Float = 0
    for keypoint in pose.keypoints {
      if !keypoint!.isWithinRadiusOfCorrespondingKeypoints(in: poses, radius: config.nmsRadius) {
        notOverlappedKeypointScoreAccumulator += keypoint!.score
      }
    }
    return notOverlappedKeypointScoreAccumulator / Float(KeypointIndex.allCases.count)
  }
}
