import Foundation
import ModelSupport

/// A logger that writes protobuf events into a tensorboard-readable file.
struct EventLogger {
  /// Path of the file.
  let filePath: String

  /// Creates an instance with log located at `logDirectory`; creates 
  /// the log file and add an initial event as well. 
  init(logDirectory: String) throws {
    // Create the directory if it is missing.
    try FoundationFileSystem().createDirectoryIfMissing(at: logDirectory)

    // Create the file.
    let timeStamp = Date().timeIntervalSince1970
    filePath = URL(fileURLWithPath: logDirectory, isDirectory: true).appendingPathComponent(
      "events.out.tfevents." + String(timeStamp).split(separator: ".")[0] + "." + (
        Host.current().localizedName ?? "nil"
      )!).path

    try FoundationFile(path: filePath).write(Data())

    // Add an initial event.
    var initialEvent = TensorboardX_Event()
    initialEvent.wallTime = timeStamp
    initialEvent.fileVersion = "brain.Event:2"
    try add(initialEvent)
  }

  /// Add an event to the log.
  func add(_ event: TensorboardX_Event) throws {
    let data: Data = try event.serializedData()

    var header: Data = Data()
    header.append(contentsOf: UInt64(data.count).littleEndianBuffer)

    var headerCRC: Data = Data()
    headerCRC.append(contentsOf: header.maskedCRC32C().littleEndianBuffer)

    var dataCRC: Data = Data()
    dataCRC.append(contentsOf: data.maskedCRC32C().littleEndianBuffer)

    let file = FoundationFile(path: filePath)

    try file.append(header)
    try file.append(headerCRC)
    try file.append(data)
    try file.append(dataCRC)
  }
}
