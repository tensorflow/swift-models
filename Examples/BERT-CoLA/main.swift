// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(fileURLWithPath: "bert_models", isDirectory: true,
                       relativeTo: URL(fileURLWithPath: NSTemporaryDirectory(),
                                       isDirectory: true))
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
let maxSequenceLength = 128
let batchSize = 1024

// Create a function that converts examples to data batches.
let exampleMapFn: (CoLA.Example) -> CoLA.DataBatch = { example -> CoLA.DataBatch in
    let textBatch = bertClassifier.bert.preprocess(
        sequences: [example.sentence],
        maxSequenceLength: maxSequenceLength)
    return CoLA.DataBatch(
        inputs: textBatch, labels: example.isAcceptable.map { Tensor($0 ? 1 : 0) })
    }

var cola = try CoLA(
    exampleMap: exampleMapFn,
    taskDirectoryURL: workspaceURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize,
    dropRemainder: true)

print("Dataset acquired.")

var optimizer = WeightDecayedAdam(
    for: bertClassifier,
    learningRate: LinearlyDecayedParameter(
        baseParameter: LinearlyWarmedUpParameter(
            baseParameter: FixedParameter<Float>(2e-5),
            warmUpStepCount: 10,
            warmUpOffset: 0),
        slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
        startStep: 10),
    weightDecayRate: 0.01,
    maxGradientGlobalNorm: 1)

print("Training BERT for the CoLA task!")
for epoch in 1...10 {
    print("[Epoch \(epoch)]")
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0

    for _ in 1...10 {
        let batch = withDevice(.cpu) { cola.trainDataIterator.next()! }
        let (documents, labels) = (batch.inputs, Tensor<Float>(batch.labels!))
        let (loss, gradients) = valueWithGradient(at: bertClassifier) { model -> Tensor<Float> in
            let logits = model(documents)
            return sigmoidCrossEntropy(
                logits: logits.squeezingShape(at: -1),
                labels: labels,
                reduction: { $0.mean() })
        }

        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&bertClassifier, along: gradients)

        print(
            """
              Training loss: \(trainingLossSum / Float(trainingBatchCount))
            """
        )
    }

    Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var devDataIterator = cola.devDataIterator
    var devPredictedLabels = [Bool]()
    var devGroundTruth = [Bool]()
    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
        let (documents, labels) = (batch.inputs, batch.labels!)
        let logits = bertClassifier(documents)
        let loss = sigmoidCrossEntropy(
            logits: logits.squeezingShape(at: -1),
            labels: Tensor<Float>(labels),
            reduction: { $0.mean() }
        )
        testLossSum += loss.scalarized()
        testBatchCount += 1

        let predictedLabels = sigmoid(logits.squeezingShape(at: -1)) .>= 0.5
        devPredictedLabels.append(contentsOf: predictedLabels.scalars)
        devGroundTruth.append(contentsOf: labels.scalars.map { $0 == 1 })
    }

    let mcc = matthewsCorrelationCoefficient(
        predictions: devPredictedLabels,
        groundTruth: devGroundTruth)

    print(
        """
          MCC: \(mcc)
          Eval loss: \(testLossSum / Float(testBatchCount))
        """
    )
}
