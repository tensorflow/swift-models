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
  func decode() {
    // Batch size hardcoded to 1 at the moment, we could fix this with a for loop easily though,
    // as I don't think it makes sense to vectorize this along batch size.
    let poses = [Pose]()
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

          if rootKeypoint.score < config.keypointScoreThreshold { continue }
          if !scoreIsMaximumInLocalWindow(at: rootKeypoint)  { continue }

          var pose = Pose()
          pose.add(rootKeypoint)

          // Follow keypoint structure going forward
          recursivellyAddNextKeypoint(from: rootKeypoint, into: &pose)

          // Follow keypoint structure going backward
        }
      }
    }
  }

  func recursivellyAddNextKeypoint(from previousKeypoint: Keypoint, into pose: inout Pose) {
    for nextKeypointIndex in getKeypointIndex(following: previousKeypoint.index) {
      if nextKeypointIndex != nil {
        let (nextY, nextX) = followDisplacement(from: previousKeypoint, using: displacementsFwd)
        let newKeypoint = Keypoint(
          y: nextY,
          x: nextX,
          index: nextKeypointIndex!,
          score: heatmap[nextY, nextX, nextKeypointIndex!.rawValue].scalarized()
        )
        pose.add(newKeypoint)
        recursivellyAddNextKeypoint(from: newKeypoint, into: &pose)
      }
    }
  }

  func followDisplacement(from keypoint: Keypoint, using displacements: Tensor<Float>) -> (y: Int, x: Int) {

  }

  func scoreIsMaximumInLocalWindow(at keypoint: Keypoint) -> Bool {
    let yStart = max(keypoint.heatmapY - config.keypointLocalMaximumRadius, 0)  // TODO: Added a + 1 here, check if correct
    let yEnd = min(keypoint.heatmapY + config.keypointLocalMaximumRadius + 1, heatmap.shape[0])
    for windowY in yStart...yEnd {
      let xStart = max(keypoint.heatmapX - config.keypointLocalMaximumRadius, 0)  // TODO: Added a + 1 here, check if correct
      let xEnd = min(keypoint.heatmapX + config.keypointLocalMaximumRadius + 1, heatmap.shape[1])
      for windowX in xStart...xEnd {
        if heatmap[windowY, windowX, keypoint.index.rawValue].scalarized() > keypoint.score {
          return false
        }
      }
    }
    return true
  }

}
