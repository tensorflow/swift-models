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

import Foundation

let progressBarLength = 30

public class TrainingProgress {
  var statistics: TrainingStatistics?
  let metrics: Set<TrainingMetrics>

  public init(metrics: Set<TrainingMetrics> = [.loss]) {
    self.metrics = metrics
    if metrics.contains(.loss) {
      statistics = TrainingStatistics()
    }
  }

  func progressBar(progress: Float, length: Int) -> String {
    let progressSteps = Int(round(Float(length) * progress))
    let leading = String(repeating: "=", count: progressSteps)
    let separator: String
    let trailing: String
    if progressSteps < progressBarLength {
      separator = ">"
      trailing = String(repeating: ".", count: progressBarLength - progressSteps - 1)
    } else {
      separator = ""
      trailing = ""
    }
    return "[\(leading)\(separator)\(trailing)]"
  }

  func metricDescription() -> String {
    var result: String = ""
    if metrics.contains(.loss) {
      result += " - loss: \(String(format: "%.4f", statistics!.averageLoss()))"
    }
    return result
  }

  public func update<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    try statistics?.record(&loop, event: event)

    switch event {
    case .epochStart:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount else {
        return
      }
      print("Epoch \(epochIndex + 1)/\(epochCount)\n")
    case .batchEnd:
      guard let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else {
        return
      }
      let epochProgress = Float(batchIndex + 1) / Float(batchCount)
      let progressBarComponent = progressBar(progress: epochProgress, length: progressBarLength)
      let metricDescriptionComponent = metricDescription()
      print(
        "\u{1B}[1A\u{1B}[K\(batchIndex + 1)/\(batchCount) \(progressBarComponent)\(metricDescriptionComponent)"
      )
    case .validationStart:
      print("")
    default: break
    }
  }
}
