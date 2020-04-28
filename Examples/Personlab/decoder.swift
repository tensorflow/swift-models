import TensorFlow

struct PoseDecoder {
  let heatmap: Tensor<Float>
  let offsets: Tensor<Float>
  let displacementsFwd: Tensor<Float>
  let displacementsBwd: Tensor<Float>

  init(for results: PersonlabHeadsResults, with config: Config) {
    // Remove [0] indexes when we add batchsize > 1 support, which should be really easy
    self.heatmap = results.heatmap[0]
    self.offsets = results.offsets[0]
    self.displacementsFwd = results.displacementsFwd[0]
    self.displacementsBwd = results.displacementsBwd[0]
  }

  // TODO: Call as a function? Seem kinda obscene, lol.
  func decode() -> [Pose] {
    // Batch size hardcoded to 1 at the moment, we could fix this with a for loop easily though,
    // as I don't think it makes sense to vectorize this along batch size.
    var poses = [Pose]()
    var keypointPriorityQueue = getKeypointPriorityQueue()
    while keypointPriorityQueue.count > 0 {
      let rootKeypoint = keypointPriorityQueue.dequeue()!
      if rootKeypoint.isWithinRadiusOfCorrespondingPoint(in: poses) {
        continue
      }

      var pose = Pose()
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
    for (nextKeypointIndex, direction) in getNextKeypoint(previousKeypoint.index) {
      if pose.getKeypoint(with: nextKeypointIndex) == nil {
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

  func followDisplacement(from previousKeypoint: Keypoint, to nextKeypointIndex: KeypointIndex, using displacements: Tensor<Float>) -> Keypoint {

    let displacementIndexY = keypointPairToDisplacementIndexMap[Set([previousKeypoint.index, nextKeypointIndex])]!
    let displacementIndexX = displacementIndexY + displacements.shape[2] / 2

    let displacementY = displacements[
      getUnstridedIndex(y: previousKeypoint.y),
      getUnstridedIndex(x: previousKeypoint.x),
      displacementIndexY
    ].scalarized()
    let displacementX = displacements[
      getUnstridedIndex(y: previousKeypoint.y),
      getUnstridedIndex(x: previousKeypoint.x),
      displacementIndexX
    ].scalarized()

    let displacedY = previousKeypoint.y + displacementY
    let displacedX = previousKeypoint.x + displacementX

    let yOffset = offsets[
      getUnstridedIndex(y: displacedY),
      getUnstridedIndex(x: displacedX),
      nextKeypointIndex.rawValue
    ].scalarized()
    let xOffset = offsets[
      getUnstridedIndex(y: displacedY),
      getUnstridedIndex(x: displacedX),
      nextKeypointIndex.rawValue + KeypointIndex.allCases.count
    ].scalarized()

    // If we are getting the offset from an exact point in the heatmap, we should add this
    // offset parting from that exact point in the heatmap, so we just nearest neighbour
    // interpolate it back, then re strech using output stride, and then add said offset.
    let nextY = Float(getUnstridedIndex(y: displacedY) * config.outputStride) + yOffset
    let nextX = Float(getUnstridedIndex(x: displacedX) * config.outputStride) + xOffset

    return Keypoint(
      y: nextY,
      x: nextX,
      index: nextKeypointIndex,
      score: heatmap[
        getUnstridedIndex(y: displacedY), getUnstridedIndex(x: displacedX), nextKeypointIndex.rawValue
      ].scalarized()
    )
  }

  func scoreIsMaximumInLocalWindow(heatmapY: Int, heatmapX: Int, score: Float, keypointIndex: Int) -> Bool {
    let yStart = max(heatmapY - config.keypointLocalMaximumRadius, 0)
    let yEnd = min(heatmapY + config.keypointLocalMaximumRadius, heatmap.shape[0] - 1)
    for windowY in yStart...yEnd {
      let xStart = max(heatmapX - config.keypointLocalMaximumRadius, 0)
      let xEnd = min(heatmapX + config.keypointLocalMaximumRadius, heatmap.shape[1] - 1)
      for windowX in xStart...xEnd {
        if heatmap[windowY, windowX, keypointIndex].scalarized() > score {
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

  func getKeypointPriorityQueue() -> Heap<Keypoint> {
    var keypointPriorityQueue = Heap<Keypoint>(priorityFunction: {$0.score > $1.score})  // TODO: Check order
    for heatmapY in 0..<heatmap.shape[0] {
      for heatmapX in 0..<heatmap.shape[1] {
        for keypointIndex in 0..<heatmap.shape[2] {
          let score = heatmap[heatmapY, heatmapX, keypointIndex].scalarized()

          if score < config.keypointScoreThreshold { continue }
          if scoreIsMaximumInLocalWindow(
            heatmapY: heatmapY,
            heatmapX: heatmapX,
            score: score,
            keypointIndex: keypointIndex
          )  {
            keypointPriorityQueue.enqueue(
              Keypoint(
                heatmapY: heatmapY,
                heatmapX: heatmapX,
                index: keypointIndex,
                score: score,
                offsets: offsets
              )
            )
          }
        }
      }
    }
    return keypointPriorityQueue
  }

  func getPoseScore(for pose: Pose, considering poses: [Pose]) -> Float {
    var notOverlappedKeypointScoreAccumulator: Float = 0
    for keypoint in pose.keypoints {
      if !keypoint!.isWithinRadiusOfCorrespondingPoint(in: poses) {
        notOverlappedKeypointScoreAccumulator += keypoint!.score
      }
    }
    return notOverlappedKeypointScoreAccumulator / Float(KeypointIndex.allCases.count)
  }
}
