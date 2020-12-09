import Foundation
import TensorFlow
import TrainingLoop

/// Returns a TrainingLoop callback that logs training and validation statistics
/// to be consumed by tensorboard. 
public func tensorBoardStatisticsLogger<L: TrainingLoopProtocol>(
  logRootDirectory: String = "run/tensorboard/stats"
) -> TrainingLoopCallback<L> {
  let logDirTimestampPrefix = String(Date().timeIntervalSince1970)
  // globalTrainBatchIndex will hold value iff the stats recorded per batch and in training phrase.
  var globalTrainBatchIndex: Int? = nil
  var summaryWriters: [String: SummaryWriter] = [:]
  let globalQueue = DispatchQueue.global()

  return { (loop, event) throws -> Void in
    if event != .batchEnd { return }

    guard let epochIndex = loop.epochIndex, let batchIndex = loop.batchIndex,
      let batchCount = loop.batchCount, let stats = loop.lastStatsLog
    else {
      return
    }

    let learningPhase = Context.local.learningPhase == .inference ? "validation" : "train"

    if learningPhase == "train" && batchIndex + 1 != batchCount {
      globalTrainBatchIndex = 0
    } else if learningPhase == "validation" {
      globalTrainBatchIndex = nil
    }
    if globalTrainBatchIndex != nil {
      globalTrainBatchIndex = epochIndex * batchCount + batchIndex
    }

    for stat in stats {
      let writerKey = stat.name + "/" + learningPhase
      var writer: SummaryWriter
      if let stored = summaryWriters[writerKey] {
        writer = stored
      } else {
        writer = SummaryWriter(
          logDirectory: URL(fileURLWithPath: logRootDirectory, isDirectory: true)
            .appendingPathComponent(logDirTimestampPrefix).appendingPathComponent(writerKey).path)
        summaryWriters[writerKey] = writer
      }

      globalQueue.async {
        if batchIndex + 1 == batchCount {
          writer.addScalar(tag: "epoch_" + stat.name, value: stat.value, step: Int64(epochIndex))
        }
        if let globalTrainBatchIndex = globalTrainBatchIndex {
          writer.addScalar(
            tag: "batch_" + stat.name, value: stat.value, step: Int64(globalTrainBatchIndex))
        }
      }
    }
  }
}
