// Adapted from: https://github.com/eaplatanios/nca/blob/master/Sources/NCA/Architecture.swift

public struct ArchitectureInput {
  public let text: TextBatch?

  public init(text: TextBatch? = nil) {
    self.text = text
  }
}
