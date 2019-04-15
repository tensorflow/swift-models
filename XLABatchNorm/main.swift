import TensorFlow

let batchSize = 128

let cifarDataset = loadCIFAR10()
let testBatches = cifarDataset.test.batched(Int64(batchSize))

// Returns an empty Array if saveResults is false.
func computeBnorm(useXLA: Bool, nResults: Int, epoch: Int64 = 1) -> Array<Tensor<Float>> {
  let model = ConvBN(filterShape: (3, 3, 3, 64), padding: .same, useXLA: useXLA)
  var results:Array<Tensor<Float>> = []
  let trainingShuffled = cifarDataset.training.shuffled(
    sampleCount: 50000, randomSeed: epoch)
  for batch in trainingShuffled.batched(Int64(batchSize)) {
    let images = batch.data
    let res =  model.applied(to: images, in: Context(learningPhase: .training))
    results.append(res);
    if results.count > nResults {
      break
    }
  }
  return results
}

func checkBnormResults() {
  let nResultsToCheck = 100
  let resultsNoXLA = computeBnorm(useXLA: false, nResults: nResultsToCheck)
  let resultsXLA = computeBnorm(useXLA: true, nResults: nResultsToCheck)
  let error: Float = 0.000001
  var i = 0
  print("Checking nearly equal with error: \(error)")
  for (l, r) in zip(resultsNoXLA, resultsXLA) {
    if (abs(l - r) > error) {
      print("Mismatch at \(i):\n")// \(resultsNoXLA[i]) \n vs \(resultsXLA[i])\n")
    }
    i += 1
  }
}


simpleTest()
checkBnormResults()

