import Foundation
import TensorBoard
import TensorFlow

/// Returns a TrainingLoop callback that writes training and validation statistics
/// to be consumed by tensorboard. 
public func tensorBoardStatsWriter<L: TrainingLoopProtocol>(
  logRootDirectory: String = "run/tensorboard/stats"
) -> TrainingLoopCallback<L> {
  var summaryWriters: [String: SummaryWriter] = [:]
  let globalQueue = DispatchQueue.global()

  return { (loop, event) throws -> Void in
    if event != .batchEnd { return }

    guard let epochIndex = loop.epochIndex, let batchIndex = loop.batchIndex,
      let batchCount = loop.batchCount, let stats = loop.lastStatsLog
    else {
      return
    }

    if batchIndex + 1 != batchCount { return }

    let learningPhase = Context.local.learningPhase == .inference ? "validation" : "train"
    for stat in stats {
      let writerKey = stat.name + "/" + learningPhase
      var writer: SummaryWriter
      if let stored = summaryWriters[writerKey] {
        writer = stored
      } else {
        writer = SummaryWriter(
          logDirectory: URL(fileURLWithPath: logRootDirectory, isDirectory: true)
            .appendingPathComponent(writerKey).path)
        summaryWriters[writerKey] = writer
      }

      globalQueue.async {
        writer.addScalar(tag: stat.name, value: stat.value, step: Int64(epochIndex))
      }
    }
  }
}
