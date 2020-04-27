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
    for heatmapY in 0..<heatmap.shape[0] {
      for heatmapX in 0..<heatmap.shape[1] {
        for keypointIndex in 0..<heatmap.shape[2] {
          let rootKeypoint = Keypoint(
            heatmapY: heatmapY,
            heatmapX: heatmapX,
            index: keypointIndex,
            score: heatmap[heatmapY, heatmapX, keypointIndex].scalarized(),
            offsets: offsets
          )
          print("root", rootKeypoint)

          if rootKeypoint.score < config.keypointScoreThreshold { continue }
          if !scoreIsMaximumInLocalWindow(at: rootKeypoint)  { continue }

          var pose = Pose()
          pose.add(rootKeypoint)

          // Recursivelly parse keypoint tree going in forward direction
          recursivellyAddNextKeypoint(
            after: rootKeypoint,
            into: &pose,
            following: getSucceedingtKeypointIndex
          )

          // Recursivelly parse keypoint tree going in backward direction
          recursivellyAddNextKeypoint(
            after: rootKeypoint,
            into: &pose,
            following: getPrecedingKeypointIndex
          )

          poses.append(pose)
          print("pose", pose)
          readLine()
          print()
        }
      }
    }

    return poses
  }

  func recursivellyAddNextKeypoint(
    after previousKeypoint: Keypoint,
    into pose: inout Pose,
    following getNextKeypoint: (KeypointIndex) -> [KeypointIndex?]) {
    for nextKeypointIndex in getNextKeypoint(previousKeypoint.index) {
      if nextKeypointIndex != nil {
        print("previous", previousKeypoint)
        let nextKeypoint = followDisplacement(
          from: previousKeypoint,
          to: nextKeypointIndex!,
          using: displacementsFwd
        )
        print("next", nextKeypoint)
        print()
        pose.add(nextKeypoint)
        recursivellyAddNextKeypoint(after: nextKeypoint, into: &pose, following: getNextKeypoint)
      }
    }
  }

  func followDisplacement(
    from previousKeypoint: Keypoint,
    to nextKeypointIndex: KeypointIndex,
    using displacements: Tensor<Float>
  ) -> Keypoint {
    let displacementIndexY = keypointPairToDisplacementIndexMap[
      Set([previousKeypoint.index, nextKeypointIndex])
    ]!
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

    let nextY = displacedY + yOffset
    let nextX = displacedX + xOffset

    return Keypoint(
      y: nextY,
      x: nextX,
      index: nextKeypointIndex,
      score: heatmap[
        getUnstridedIndex(y: nextY), getUnstridedIndex(x: nextX), nextKeypointIndex.rawValue
      ].scalarized()
    )
  }

  func scoreIsMaximumInLocalWindow(at keypoint: Keypoint) -> Bool {
    let unstridedIndexY = getUnstridedIndex(y: keypoint.y)
    let yStart = max(unstridedIndexY - config.keypointLocalMaximumRadius, 0)
    let yEnd = min(unstridedIndexY + config.keypointLocalMaximumRadius, heatmap.shape[0]) // NOTE: removed a + 1
    for windowY in yStart..<yEnd {
      let unstridedIndexX = getUnstridedIndex(x: keypoint.x)
      let xStart = max(unstridedIndexX - config.keypointLocalMaximumRadius, 0)
      let xEnd = min(unstridedIndexX + config.keypointLocalMaximumRadius, heatmap.shape[1]) // NOTE: removed a + 1
      for windowX in xStart..<xEnd {
        if heatmap[windowY, windowX, keypoint.index.rawValue].scalarized() > keypoint.score {
          return false
        }
      }
    }
    return true
  }

  func getUnstridedIndex(y: Float) -> Int {
    let downScaled = y / Float(config.outputStride)
    let clamped = min(max(0, downScaled), Float(heatmap.shape[0]))
    return Int(clamped)
  }

  func getUnstridedIndex(x: Float) -> Int {
    let downScaled = x / Float(config.outputStride)
    let clamped = min(max(0, downScaled), Float(heatmap.shape[1]))
    return Int(clamped)
  }

}
