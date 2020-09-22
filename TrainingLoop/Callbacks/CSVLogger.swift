import Foundation
import ModelSupport

/// A handler for logging training and validation statistics to a CSV file.
public class CSVLogger {
  /// The path of the file that statistics are logged to.
  public var path: String

  // The boolean variable indicating if header of the CSV file has been written or not.
  fileprivate var headerWritten: Bool

  /// Creates an instance that logs to a file with the given path.
  ///
  /// Throws: File system errors.
  public init(path: String = "run/log.csv") throws {
    self.path = path

    // Create the containing directory if it is missing.
    let containingDir = String(path[..<path.lastIndex(of: "/")!])
    if containingDir != "" {
      try FoundationFileSystem().createDirectoryIfMissing(at: containingDir)
    }
    // Initialize the file with empty string.
    try FoundationFile(path: path).write(Data())

    self.headerWritten = false
  }

  /// Logs the statistics for the 'loop' when 'batchEnd' event happens; 
  /// ignoring other events.
  ///
  /// Throws: File systel errors.
  public func log<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .batchEnd:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount,
        let batchIndex = loop.batchIndex, let batchCount = loop.batchCount,
        let stats = loop.lastStatsLog
      else {
        // No-Op if trainingLoop doesn't set the required values for stats logging.
        return
      }

      if !headerWritten {
        try writeHeader(stats: stats)
        headerWritten = true
      }

      try writeDataRow(
        epoch: "\(epochIndex + 1)/\(epochCount)",
        batch: "\(batchIndex + 1)/\(batchCount)",
        stats: stats)
    default:
      return
    }
  }

  func writeHeader(stats: [(name: String, value: Float)]) throws {
    let header: String = (["epoch", "batch"] + stats.map { $0.name }).joined(separator: ", ") + "\n"
    try FoundationFile(path: path).append(header.data(using: .utf8)!)
  }

  func writeDataRow(epoch: String, batch: String, stats: [(name: String, value: Float)]) throws {
    let dataRow: String = ([epoch, batch] + stats.map { String($0.value) }).joined(separator: ", ")
      + "\n"
    try FoundationFile(path: path).append(dataRow.data(using: .utf8)!)
  }
}
