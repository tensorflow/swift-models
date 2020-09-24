import Foundation
import ModelSupport

public enum CSVLoggerError: Error {
  case InvalidPath
}

/// A handler for logging training and validation statistics to a CSV file.
public class CSVLogger {
  /// The path of the file to which statistics are logged.
  public var path: String

  // True iff the header of the CSV file has been written.
  fileprivate var headerWritten: Bool

  /// Creates an instance that logs to a file with the given `path`.
  ///
  /// Throws: File system errors.
  public init(path: String = "run/log.csv") throws {
    self.path = path

    // Validate the path.
    let url = URL(fileURLWithPath: path)
    if url.pathExtension != "csv" {
      throw CSVLoggerError.InvalidPath
    }
    // Create the containing directory if it is missing.
    try FoundationFileSystem().createDirectoryIfMissing(at: url.deletingLastPathComponent().path)
    // Initialize the file with empty string.
    try FoundationFile(path: path).write(Data())

    self.headerWritten = false
  }

  /// Logs the statistics for the `loop` when `batchEnd` event happens; 
  /// ignoring other events.
  ///
  /// Throws: File system errors.
  public func log<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    switch event {
    case .batchEnd:
      guard let epochIndex = loop.epochIndex, let epochCount = loop.epochCount,
        let batchIndex = loop.batchIndex, let batchCount = loop.batchCount,
        let stats = loop.lastStatsLog
      else {
        return
      }

      if !headerWritten {
        try writeHeader(stats: stats)
        headerWritten = true
      }

      try writeDataRow(
        epochData: "\(epochIndex + 1)/\(epochCount)",
        batchData: "\(batchIndex + 1)/\(batchCount)",
        stats: stats)
    default:
      return
    }
  }

  /// Writes column names to file header.
  /// 
  /// Column names include "epoch", "batch" and `stats` `name`s.
  func writeHeader(stats: [(name: String, value: Float)]) throws {
    let header = (["epoch", "batch"] + stats.lazy.map { $0.name }).joined(separator: ", ") + "\n"
    try FoundationFile(path: path).append(header.data(using: .utf8)!)
  }

  /// Appends a row of statistics log to file with the given value `epochData` for 
  /// "epoch" column, `batchData` for "batch" column, and `value`s of `stats` for corresponding 
  /// columns indicated by `stats` `name`s.
  func writeDataRow(epochData: String, batchData: String, stats: [(name: String, value: Float)]) throws {
    let dataRow = ([epochData, batchData] + stats.lazy.map { String($0.value) }).joined(separator: ", ")
      + "\n"
    try FoundationFile(path: path).append(dataRow.data(using: .utf8)!)
  }
}
