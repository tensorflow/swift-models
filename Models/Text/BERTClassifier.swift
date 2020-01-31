// Adapted from: https://gist.github.com/eaplatanios/eae9c1b4141e961c949d6f2e7d424c6f
// Untested.

import TensorFlow
public struct BERTClassifier: Module, Regularizable {
  public var bert: BERT
  public var dense: Dense<Float>

  public var regularizationValue: TangentVector {
    TangentVector(
      bert: bert.regularizationValue,
      dense: dense.regularizationValue)
  }

  public init(bert: BERT, classCount: Int) {
    self.bert = bert
    self.dense = Dense<Float>(inputSize: bert.hiddenSize, outputSize: classCount)
  }

  /// Returns: logits with shape `[batchSize, classCount]`.
  @differentiable(wrt: self)
  public func callAsFunction(_ input: TextBatch) -> Tensor<Float> {
    dense(bert(input)[0])
  }
}
