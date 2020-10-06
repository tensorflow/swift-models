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

/// A handler for recording training and validation statistics.
///
/// Data produced by this handler can be used by ProgressPrinter, CVSLogger, etc.
public class StatisticsRecorder {
  /// A Closure that returns if should call 'reset' on metricMeasurers.
  public var shouldReset:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  /// A Closure that returns if should call 'accumulate' on metricMeasurers.
  public var shouldAccumulate:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  /// A Closure that returns if should call 'compute' on metricMeasurers.
  public var shouldCompute:
    (
      _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
      _ event: TrainingLoopEvent
    ) -> Bool

  /// Instances of MetricsMeasurers.
  fileprivate var metricMeasurers: [MetricsMeasurer]

  /// Create an instance that records 'metrics' during the training loop.
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

  /// Recording statistics in response of the 'event'.
  ///
  /// It will record the statistics into 'lastStatsLog' in the loop where other 
  /// callbacks can consume from.
  public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    guard let batchIndex = loop.batchIndex,
      let batchCount = loop.batchCount,
      let epochIndex = loop.batchIndex,
      let epochCount = loop.epochCount,
      let loss = loop.lastStepLoss,
      let output = loop.lastStepOutput,
      let target = loop.lastStepTarget
    else {
      // No-Op if trainingLoop doesn't set the required values for stats recording.
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
      result.append((name: measurer.name, value: measurer.measure()))
    }
    return result
  }
}

extension StatisticsRecorder {
  /// The events on which statistics will be reported for training and validation phases.
  public enum ReportTrigger {
    /// Report statistics at end of training and validation, once per epoch.
    case endOfEpoch
    /// Report statistics at end of training and validation, once per batch.
    case endOfBatch
  }

  /// Set the StatisticsRecorder to report statistics when `trigger` is triggered.
  public func setReportTrigger(_ trigger: ReportTrigger) {
    if trigger == .endOfBatch {
      shouldCompute = {
          (
            _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
            _ event: TrainingLoopEvent
          ) -> Bool in
          return event == .batchEnd
        }
    } else {
      shouldCompute = {
          (
            _ batchIndex: Int, _ batchCount: Int, _ epochIndex: Int, _ epochCount: Int,
            _ event: TrainingLoopEvent
          ) -> Bool in
          return event == .batchEnd && batchIndex + 1 == batchCount
        }
    }
  }
}
