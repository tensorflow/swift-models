// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public func createDirectoryIfMissing(at path: String) throws {
    guard !FileManager.default.fileExists(atPath: path) else { return }
    try FileManager.default.createDirectory(
        atPath: path,
        withIntermediateDirectories: true,
        attributes: nil)
}

public func download(from source: URL, to destinationDirectory: URL) throws {
    try createDirectoryIfMissing(at: destinationDirectory.path)

    let fileName = source.lastPathComponent
    let destinationFile = destinationDirectory.appendingPathComponent(fileName).path

    let downloadedFile = try Data(contentsOf: source)
    try downloadedFile.write(to: URL(fileURLWithPath: destinationFile))
}

/// Collect all file URLs under a folder `url`, potentially recursing through all subfolders.
/// Optionally filters some extension (only jpeg or txt files for instance).
///
/// - Parameters:
///   - url: The folder to explore.
///   - recurse: Will explore all subfolders if set to `true`.
///   - extensions: Only keeps URLs with extensions in that array if it's provided
public func collectURLs(
    under directory: URL, 
    recurse: Bool = false, 
    filtering extensions: [String]? = nil
) -> [URL] {
    var files: [URL] = []
    do {
        let dirContents = try FileManager.default.contentsOfDirectory(
            at: directory, 
            includingPropertiesForKeys: [.isDirectoryKey], 
            options: [.skipsHiddenFiles])
        for content in dirContents {
            if content.hasDirectoryPath && recurse {
                files += collectURLs(under: content, recurse: recurse, filtering: extensions)
            } else if content.isFileURL && (extensions == nil 
                || extensions!.contains(dirContents[0].pathExtension.lowercased())) {
                files.append(content)
            }
        }
    } catch {
        fatalError("Could not explore this folder: \(error)")
    }
    return files
}
