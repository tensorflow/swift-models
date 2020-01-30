import TensorFlow
import Foundation

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(fileURLWithPath: "/tmp/bert_models", isDirectory: true)
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 2)

var colaTask = try CoLA(for: bertClassifier, taskDirectoryURL: workspaceURL, maxSequenceLength: 50, batchSize: 32)

let lr = FixedParameter<Float>(2e-5)
let pivotStepCount: UInt64 = 1000
let lr2 = LinearlyWarmedUpParameter(baseParameter: lr, warmUpStepCount: pivotStepCount, warmUpOffset: 0)
let lr3 =  LinearlyDecayedParameter(baseParameter: lr2, slope: 3, startStep: pivotStepCount)
var optimizer = WeightDecayedAdam(for: bertClassifier, learningRate: lr3, maxGradientGlobalNorm: 1)

print("Training BERT for the CoLA task!")
let epochCount = 1000
for epoch in 0..<epochCount {
  let loss = colaTask.update(architecture: &bertClassifier, using: &optimizer)
  print("[Epoch: \(epoch)]\tLoss: \(loss)")
}
