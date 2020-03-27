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
import TensorFlow
import TextModels

extension CoLA {
    @discardableResult
    public mutating func update<O: TextModels.Optimizer>(
        classifier: inout BERTClassifier,
        using optimizer: inout O
    ) -> Float where O.Model == BERTClassifier {
        let batch = withDevice(.cpu) { trainDataIterator.next()! }
        let input = batch.inputs
        let labels = Tensor<Float>(batch.labels!)
        return withLearningPhase(.training) {
            let (loss, gradient) = valueWithGradient(at: classifier) {
                sigmoidCrossEntropy(
                    logits: $0(input).squeezingShape(at: -1),
                    labels: labels,
                    reduction: { $0.mean() })
            }
            optimizer.update(&classifier, along: gradient)
            return loss.scalarized()
        }
    }

    public func evaluate(using classifier: BERTClassifier) -> [String: Float] {
        var devDataIterator = self.devDataIterator
        var devPredictedLabels = [Bool]()
        var devGroundTruth = [Bool]()
        while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
            let predictions = classifier(batch.inputs)
            let predictedLabels = sigmoid(predictions.squeezingShape(at: -1)) .>= 0.5
            devPredictedLabels.append(contentsOf: predictedLabels.scalars)
            devGroundTruth.append(contentsOf: batch.labels!.scalars.map { $0 == 1 })
        }
        return [
            "matthewsCorrelationCoefficient": matthewsCorrelationCoefficient(
                predictions: devPredictedLabels,
                groundTruth: devGroundTruth)
        ]
    }
}

extension CoLA {
    public init(
        for architecture: BERTClassifier,
        taskDirectoryURL: URL,
        maxSequenceLength: Int,
        batchSize: Int
    ) throws {
        // Create a function that converts examples to data batches.
        let exampleMapFn: (Example) -> DataBatch = { example -> DataBatch in
            let textBatch = architecture.bert.preprocess(
                sequences: [example.sentence],
                maxSequenceLength: maxSequenceLength)
            return DataBatch(
                inputs: textBatch, labels: example.isAcceptable.map { Tensor($0 ? 1 : 0) })
        }

        try self.init(
            exampleMap: exampleMapFn, taskDirectoryURL: taskDirectoryURL,
            maxSequenceLength: maxSequenceLength, batchSize: batchSize, dropRemainder: true)
    }
}
