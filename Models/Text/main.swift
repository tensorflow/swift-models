import TensorFlow
import Foundation

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(fileURLWithPath: "/tmp/bert_models", isDirectory: true)
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 1)

var colaTask = try CoLA(
  for: bertClassifier,
  taskDirectoryURL: workspaceURL,
  maxSequenceLength: 512,
  batchSize: 32)

var optimizer = WeightDecayedAdam(
  for: bertClassifier,
  learningRate: LinearlyDecayedParameter(
    baseParameter: FixedParameter<Float>(2e-5),
    slope: -2e-8, // The LR decays linearly to zero in 1000 steps.
    startStep: 0),
  maxGradientGlobalNorm: 1)

print("Training BERT for the CoLA task!")
let epochCount = 1000
for epoch in 0..<epochCount {
  let loss = colaTask.update(architecture: &bertClassifier, using: &optimizer)
  print("[Epoch: \(epoch)]\tLoss: \(loss)")
}
