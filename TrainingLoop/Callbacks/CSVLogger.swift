import Foundation
import ModelSupport

/// A callback-based handler for logging the statistics to CSV file.
public class CSVLogger {
  public var path: String
  public var liveStatistics: Bool

  let foundationFS: FoundationFileSystem
  let foundationFile: FoundationFile

  /// Create an instance that log statistics during the training loop.
  ///
  /// - Parameters:
  ///   - liveStatistics: whether or not log the statistics lively on each batch.
  public init(withPath path: String = "run/log.csv", liveStatistics: Bool = true) {
    self.path = path
    self.liveStatistics = liveStatistics
    self.foundationFS = FoundationFileSystem()
    self.foundationFile = FoundationFile(path: path)
  }

  /// The callback used to hook into the TrainingLoop for logging statistics.
  ///
  /// - Parameters:
  ///   - loop: The TrainingLoop where an event has occurred. 
  ///   - event: The training or validation event that this callback is responding to.
  public func log<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .batchEnd:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount,
        let batchIndex = loop.batchIndex, let batchCount = loop.batchCount
      else {
        break
      }

      if !liveStatistics && (batchIndex + 1 != batchCount) {
        break
      }

      guard let stats = loop.lastStatsLog else {
        break
      }

      if !FileManager.default.fileExists(atPath: path) {
        try foundationFS.createDirectoryIfMissing(at: String(path[..<path.lastIndex(of: "/")!]))
        try writeHeader(stats: stats)
      }
      try writeDataRow(
        epoch: "\(epochIndex + 1)/\(epochCount)",
        batch: "\(batchIndex + 1)/\(batchCount)",
        stats: stats)
    default:
      break
    }
  }

  func writeHeader(stats: [(String, Float)]) throws {
    let head: String = (["epoch", "batch"] + stats.map { $0.0 }).joined(separator: ", ")
    do {
      try head.write(toFile: path, atomically: true, encoding: .utf8)
    } catch {
      print("Unexpected error in writing header line: \(error).")
      throw error
    }
  }

  func writeDataRow(epoch: String, batch: String, stats: [(String, Float)]) throws {
    let dataRow: Data = (
      "\n" + ([epoch, batch] + stats.map { String($0.1) }).joined(separator: ", ")
    ).data(using: .utf8)!
    do {
      try foundationFile.append(dataRow)
    } catch {
      print("Unexpected error in writing data row: \(error).")
      throw error
    }
  }
}
