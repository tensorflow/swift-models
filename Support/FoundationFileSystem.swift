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

  public func open(_ filename: String) -> File {
    return FoundationFile(path: filename)
  }
    
  public func copy(source: URL, dest: URL) throws {
    try FileManager.default.copyItem(at: source, to: dest)
  }
}

public struct FoundationFile: File {
  public let location: URL
  
  public init(path: String) {
    self.location = URL(fileURLWithPath: path)
  }
  
  public func read() throws -> Data {
    return try Data(contentsOf: location, options: .alwaysMapped)
  }
  
  public func read(position: Int, count: Int) throws -> Data {
    // TODO: Incorporate file offset.
    return try Data(contentsOf: location, options: .alwaysMapped)
  }

  public func write(_ value: Data) throws {
    try self.write(value, position: 0)
  }

  public func write(_ value: Data, position: Int) throws {
    // TODO: Incorporate file offset.
    try value.write(to: location)
  }

  /// Appends the bytes in 'value' to the file.
  public func append(_ value: Data) throws {
    let fileHandler = try FileHandle(forUpdating: location)
    try fileHandler.seekToEnd()
    try fileHandler.write(contentsOf: value)
    try fileHandler.close()
  }
}
