import Checkpoints
import Foundation
import TensorFlow

func loadModel() -> HandwritingGenerator {
  var m = HandwritingGenerator()

  let remoteCheckpoint = URL(
    string:
      "https://storage.googleapis.com/s4tf-hosted-binaries/checkpoints/Handwriting/graves_handwriting_generation_2018-03-13-02-45epoch_49.tar.gz"
  )!
  do {
    let reader = try CheckpointReader(
      checkpointLocation: remoteCheckpoint, modelName: "handwriting")

    m.attention.linear.weight = Tensor<Float>(reader.loadTensor(named: "attention.linear.weight"))
    m.attention.linear.bias = Tensor<Float>(reader.loadTensor(named: "attention.linear.bias"))
    m.mixture.linear.weight = Tensor<Float>(reader.loadTensor(named: "mixture.linear.weight"))
    m.mixture.linear.bias = Tensor<Float>(reader.loadTensor(named: "mixture.linear.bias"))
    m.rnnCell.ih.weight = Tensor<Float>(reader.loadTensor(named: "rnn_cell.cell.weight_ih"))
    m.rnnCell.ih.bias = Tensor<Float>(reader.loadTensor(named: "rnn_cell.cell.bias_ih"))
    m.rnnCell.hh.weight = Tensor<Float>(reader.loadTensor(named: "rnn_cell.cell.weight_hh"))
    m.rnnCell.hh.bias = Tensor<Float>(reader.loadTensor(named: "rnn_cell.cell.bias_hh"))

    return m
  } catch {
    fatalError("Model checkpoint loading failed with error: \(error)")
  }
}
