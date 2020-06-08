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

import TensorFlow

public enum TrainingMetrics {
  case accuracy
  case loss
}

public class TrainingStatistics {
  let metrics: Set<TrainingMetrics>
  var totalBatchLoss: Tensor<Float>?
  var totalBatches: Tensor<Float>?
  var totalCorrect: Tensor<Int32>?
  var totalExamples: Int32?

  public init(metrics: Set<TrainingMetrics>) {
    self.metrics = metrics
  }
  
  public func averageLoss() -> Float {
    guard let totalBatches = totalBatches, let totalBatchLoss = totalBatchLoss else {
      return Float.nan
    }
    return (totalBatchLoss / totalBatches).scalarized()
  }

  public func accuracy() -> Float {
    guard let totalCorrect = totalCorrect, let totalExamples = totalExamples else {
      return Float.nan
    }
    return Float(totalCorrect.scalarized()) / Float(totalExamples)
  }

  public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .trainingStart, .validationStart:
      totalBatchLoss = nil
      totalBatches = nil
    case .batchEnd:
      if metrics.contains(.accuracy) {
        measureAccuracy(loop)
      }

      if let loss = loop.lastLoss, metrics.contains(.loss) {
        if let currentTotalBatchLoss = totalBatchLoss {
          totalBatchLoss = currentTotalBatchLoss + loss
          totalBatches = totalBatches! + 1.0
        } else {
          totalBatchLoss = loss
          totalBatches = Tensor<Float>(1.0, on: loss.device)
        }
      }
    default:
      return
    }
  }
  
  func measureAccuracy<L: TrainingLoopProtocol>(_ loop: L) {
    guard let possibleOutput = loop.lastOutput, let possibleTarget = loop.lastTarget else { return }
    guard let output = possibleOutput as? Tensor<Float>, let target = possibleTarget as? Tensor<Int32> else {
      fatalError(
        "For accuracy measurements, the model output must be Tensor<Float>, and the labels must be Tensor<Int>.")
    }
    
    let correct = output.argmax(squeezingAxis: 1) .== target
    let correctGuessCount = Tensor<Int32>(correct).sum()
    if let currentTotalCorrect = totalCorrect {
      totalCorrect = currentTotalCorrect + correctGuessCount
      totalExamples = totalExamples! + Int32(output.shape[0])
    } else {
      totalCorrect = correctGuessCount
      totalExamples = Int32(output.shape[0])
    }
  }
}

