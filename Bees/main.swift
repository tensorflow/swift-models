import Datasets
import ImageClassificationModels
import Foundation
import TensorFlow
import PythonKit

let np = Python.import("numpy")

// let c = try! HoneyBeeDanceFrames(
//   directory: URL(fileURLWithPath: "/usr/local/google/home/marcrasi/HoneyBees/frames/seq1_orig_cropped_compressed.avi/"))
// print(c[0])

let d = try! HoneyBeeDanceSegmentation(
  directory: URL(fileURLWithPath: "/usr/local/google/home/marcrasi/HoneyBees/frames"))

var model = UNet()
//model.load(weights: np.load("beeweights.npy", allow_pickle: true))
//model.restore()
var opt = Adam(for: model, learningRate: 1e-4)

for epoch in 0..<500 {
  print("Epoch \(epoch)")

  Context.local.learningPhase = .training
  let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
    sigmoidCrossEntropy(logits: model(d.images), labels: Tensor<Float>(d.annotations))
  }
  opt.update(&model, along: grad)
  print("Loss: \(loss)")
  model.save()
  np.save("beeweights.npy", model.weights())
}

