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

public protocol FileSystem {
  /// Creates a directory at a path, if missing. If the directory exists, this does nothing.
  ///
  /// - Parameters:
  ///   - path: The path of the desired directory.
  func createDirectoryIfMissing(at path: String) throws

  /// Opens a file at the specified location for reading or writing.
  ///
  /// - Parameters:
  ///   - path: The path of the file to be opened.
  func open(_ path: String) -> File
  
  static var defaultFileSystem: FileSystem { get }
}

public protocol File {
  func read() throws -> Data
  func read(position: Int, count: Int) throws -> Data
  func write(_ value: Data) throws
  func write(_ value: Data, position: Int) throws
}
