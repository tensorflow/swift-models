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

import Foundation

public struct FoundationFileSystem: FileSystem {
  public init() {}
  
  public func createDirectoryIfMissing(at path: String) throws {
      guard !FileManager.default.fileExists(atPath: path) else { return }
      try FileManager.default.createDirectory(
          atPath: path,
          withIntermediateDirectories: true,
          attributes: nil)
  }

  public func open(_ filename: String, mode: Set<FileMode>) -> File {
    return FoundationFile(path: filename, mode: mode)
  }
}

public struct FoundationFile: File {
  public let location: URL
  public let mode: Set<FileMode>
  
  public init(path: String, mode: Set<FileMode>) {
    self.location = URL(fileURLWithPath: path)
    self.mode = mode
  }
  
  public func read() throws -> Data {
    precondition(mode.contains(.read))
    return try Data(contentsOf: location, options: .alwaysMapped)
  }
  
  public func read(position: Int, count: Int) throws -> Data {
    precondition(mode.contains(.read))
    // TODO: Incorporate file offset.
    return try Data(contentsOf: location, options: .alwaysMapped)
  }

  public func write(_ value: Data) throws {
    precondition(mode.contains(.write))
    if mode.contains(.append) {
      guard let handle = FileHandle(forWritingAtPath: location.path) else {
        throw FoundationFileError.fileNotWriteable(path: location.path)
      }
      handle.seekToEndOfFile()
      handle.write(value)
      handle.closeFile()
    } else {
      try self.write(value, position: 0)
    }
  }

  public func write(_ value: Data, position: Int) throws {
    precondition(mode.contains(.write))
    // TODO: Incorporate file offset.
    try value.write(to: location)
  }
}

public enum FoundationFileError: Error {
  case fileNotWriteable(path: String)
}
