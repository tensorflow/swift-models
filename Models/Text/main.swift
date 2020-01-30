import TensorFlow
import Foundation

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(fileURLWithPath: "/tmp/bert_models", isDirectory: true)
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 2)

var colaTask = try CoLA(for: bertClassifier, taskDirectoryURL: workspaceURL, maxSequenceLength: 50, batchSize: 32)

let learningRate = FixedParameter<Float>(0.01)
var optimizer = WeightDecayedAdam(for: bertClassifier, learningRate: learningRate, maxGradientGlobalNorm: 0.1)

print("Training BERT for the CoLA task!")
let epochCount = 1000
for epoch in 0..<epochCount {
  let loss = colaTask.update(architecture: &bertClassifier, using: &optimizer)
  print("[Epoch: \(epoch)]\tLoss: \(loss)")
}
