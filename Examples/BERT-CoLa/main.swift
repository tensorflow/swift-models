import TensorFlow
import Foundation
import TextModels

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(fileURLWithPath: "/tmp/bert_models", isDirectory: true)
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 1)

// Regarding the batch size, note that the way batching is performed currently is that we bucket
// input sequences based on their length (e.g., first bucket contains sequences of length 1 to 10,
// second 11 to 20, etc.). We then keep processing examples in the input data pipeline until a
// bucket contains enough sequences to form a batch. The batch size specified in the task
// constructor specifies the *total number of tokens in the batch* and not the total number of
// sequences. So, if the batch size is set to 1024, the first bucket (i.e., lengths 1 to 10)
// will need 1024 / 10 = 102 examples to form a batch (every sentence in the bucket is padded
// to the max length of the bucket). This kind of bucketing is common practice with NLP models and
// it is done to improve memory usage and computational efficiency when dealing with sequences of
// varied lengths. Note that this is not used in the original BERT implementation released by
// Google and so the batch size setting here is expected to differ from that one.
var colaTask = try CoLA(
  for: bertClassifier,
  taskDirectoryURL: workspaceURL,
  maxSequenceLength: 128,
  batchSize: 1024)

var optimizer = WeightDecayedAdam(
  for: bertClassifier,
  learningRate: LinearlyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter<Float>(2e-5),
      warmUpStepCount: 10,
      warmUpOffset: 0),
    slope: -5e-7, // The LR decays linearly to zero in 100 steps.
    startStep: 10),
  weightDecayRate: 0.01,
  maxGradientGlobalNorm: 1)

print("Training BERT for the CoLA task!")
for step in 0... {
  let loss = colaTask.update(classifier: &bertClassifier, using: &optimizer)
  print("[Step: \(step)]\tLoss: \(loss)")
  if step > 0 && step.isMultiple(of: 10) {
    print("Evaluate BERT for the CoLA task:")
    print(colaTask.evaluate(using: bertClassifier))
  }
}
