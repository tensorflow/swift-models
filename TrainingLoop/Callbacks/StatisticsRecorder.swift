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

/// A callback-based handler for recording statistics.
///
/// Data produced by this handler can be used by ProgressPrinter, CVSLogger, etc.
public class StatisticsRecorder {
  public var shouldReset:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  public var shouldAccumulate:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  public var shouldCompute:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  var metricMeasurers: [MetricsMeasurer]

  /// Create an instance that records 'metrics' during the training loop.
  /// 
  /// Recording happens every batch by default or
  /// only when last batch ends when 'liveStatistics' is set to false.
  ///
  /// - Parameters:
  ///   - metrics: an array of TrainingMetrics to record.
  public init(metrics: [TrainingMetrics]) {
    metricMeasurers = metrics.map { $0.measurer }

    shouldReset = {
      (
        _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
        _ event: TrainingLoopEvent
      ) -> Bool in
      return event == .trainingStart || event == .validationStart
    }

    shouldAccumulate = {
      (
        _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
        _ event: TrainingLoopEvent
      ) -> Bool in
      return event == .batchEnd
    }

    shouldCompute = {
      (
        _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
        _ event: TrainingLoopEvent
      ) -> Bool in
      return event == .batchEnd
    }
  }

  /// The callback used to hook into the TrainingLoop for recording statistics.
  ///
  /// It will record the statistics into lastStatsLog in the loop where other 
  /// callbacks can consume from.
  ///
  /// - Parameters:
  ///   - loop: The TrainingLoop where an event has occurred. 
  ///   - event: The training or validation event that this callback is responding to.
  public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    guard let batchIndex = loop.batchIndex,
      let batchCount = loop.batchCount,
      let epochIndex = loop.batchIndex,
      let epochCount = loop.epochCount,
      let loss = loop.lastStepLoss,
      let output = loop.lastStepOutput,
      let target = loop.lastStepTarget
    else {
      return
    }

    if shouldReset(batchIndex, batchCount, epochIndex, epochCount, event) {
      resetMetricMeasurers()
      loop.lastStatsLog = nil
    }

    if shouldAccumulate(batchIndex, batchCount, epochIndex, epochCount, event) {
      accumulateMetrics(loss: loss, predictions: output, labels: target)
    }

    if shouldCompute(batchIndex, batchCount, epochIndex, epochCount, event) {
      loop.lastStatsLog = computeMetrics()
    }
  }

  func resetMetricMeasurers() {
    for index in metricMeasurers.indices {
      metricMeasurers[index].reset()
    }
  }

  func accumulateMetrics<Output, Target>(loss: Tensor<Float>, predictions: Output, labels: Target) {
    for index in metricMeasurers.indices {
      metricMeasurers[index].accumulate(loss: loss, predictions: predictions, labels: labels)
    }
  }

  func computeMetrics() -> [(String, Float)] {
    var result: [(String, Float)] = []
    for measurer in metricMeasurers {
      result.append((measurer.name, measurer.measure()))
    }
    return result
  }
}
