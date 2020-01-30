import TensorFlow
import Foundation
import ModelSupport

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
print("BERT pretrained URL: \(bertPretrained.url)")
// FIXME: This URL is not correct and needs to be fixed.
let workspaceURL = URL(string: "/bert_models")!
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 2)
var colaTask = try CoLA(for: bertClassifier, taskDirectoryURL: workspaceURL, maxSequenceLength: 50, batchSize: 32)

let learningRate = FixedParameter<Float>(0.01)
var optimizer = WeightDecayedAdam(for: bertClassifier, learningRate: learningRate)

let epochCount = 1000
for _ in 0..<epochCount {
  colaTask.update(architecture: &bertClassifier, using: &optimizer)
}
