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

/// A progress bar that displays to the console as a model trains, and as validation is performed.
/// It hooks into a TrainingLoop via a callback method.
public class TrainingProgress {
  var statistics: TrainingStatistics?
  public let metrics: Set<TrainingMetrics>
  let liveStatistics: Bool

  /// Initializes the progress bar with the metrics to be displayed (if any), and whether to
  /// provide a live update of training and validation metrics as they are calculated.
  ///
  /// - Parameters:
  ///   - metrics: A set of TrainingMetrics that specify which metrics to monitor and display
  ///     during training and validation. By default, all available metrics are selected.
  ///   - liveStatistics: Whether or not to update the metrics at the command line on every batch
  ///     as it is processed, or if these values should just be provided at the end of an epoch.
  ///     This has an impact on performance, due to materialization of tensors, and updating values
  ///     on every batch can reduce training speed by up to 30%.
  public init(metrics: Set<TrainingMetrics> = [.accuracy, .loss], liveStatistics: Bool = true) {
    self.metrics = metrics
    self.liveStatistics = liveStatistics
    if !metrics.isEmpty {
      statistics = TrainingStatistics(metrics: metrics)
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
    if metrics.contains(.accuracy) {
      result += " - accuracy: \(String(format: "%.4f", statistics!.accuracy()))"
    }

    return result
  }

  /// The callback used to hook into the TrainingLoop. This is updated once per event.
  ///
  /// - Parameters:
  ///   - loop: The TrainingLoop where an event has occurred. This can be accessed to obtain
  ///     the last measure loss and other values.
  ///   - event: The training or validation event that this callback is responding to.
  public func update<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    try statistics?.record(&loop, event: event)

    switch event {
    case .epochStart:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount else {
        return
      }
      print("Epoch \(epochIndex + 1)/\(epochCount)")
    case .batchEnd:
      guard let batchIndex = loop.batchIndex, let batchCount = loop.batchCount else {
        return
      }
      let epochProgress = Float(batchIndex + 1) / Float(batchCount)
      let progressBarComponent = progressBar(progress: epochProgress, length: progressBarLength)
      let metricDescriptionComponent: String
      if liveStatistics || (batchCount == (batchIndex + 1)) {
        metricDescriptionComponent = metricDescription()
      } else {
        metricDescriptionComponent = ""
      }
      print(
        "\r\(batchIndex + 1)/\(batchCount) \(progressBarComponent)\(metricDescriptionComponent)",
        terminator: ""
      )
      fflush(stdout)
    case .epochEnd:
      print("")
    case .validationStart:
      print("")
    default: break
    }
  }
}
