import Foundation

/// A writer for writing model execution summaries to a tensorboard-readable file; the
/// summaries include scalars for logging statistics, graphs for visualizing model etc.
public struct SummaryWriter {
  /// Logger for writing the summaries as protobuf events to the file.
  let eventLogger: EventLogger

  /// Creates an instance with log located at `logDirectory`.
  public init(logDirectory: String) {
    eventLogger = try! EventLogger(logDirectory: logDirectory)
  }

  /// Add training and validation statistics for tensorboard scarlars dashboard. 
  public func addScalar(
    tag: String, value: Float, step: Int64,
    epochTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    var summaryMetadata = TensorboardX_SummaryMetadata()
    summaryMetadata.displayName = displayName ?? tag
    summaryMetadata.summaryDescription = description ?? ""

    var summaryValue = TensorboardX_Summary.Value()
    summaryValue.tag = tag
    summaryValue.simpleValue = value
    summaryValue.metadata = summaryMetadata

    var summary = TensorboardX_Summary()
    summary.value = [summaryValue]

    var event = TensorboardX_Event()
    event.summary = summary
    event.wallTime = epochTime
    event.step = step
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }
}
