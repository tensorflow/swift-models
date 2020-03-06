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

/// Creates a directory at a path, if missing. If the directory exists, this does nothing.
///
/// - Parameters:
///   - path: The path of the desired directory.
public func createDirectoryIfMissing(at path: String) throws {
    guard !FileManager.default.fileExists(atPath: path) else { return }
    try FileManager.default.createDirectory(
        atPath: path,
        withIntermediateDirectories: true,
        attributes: nil)
}

/// Downloads a remote file and places it either within a target directory or at a target file name.
/// If `destination` has been explicitly specified as a directory (setting `isDirectory` to true
/// when appending the last path component), the file retains its original name and is placed within
/// this directory. If `destination` isn't marked in this fashion, the file is saved as a file named 
/// after `destination` and its last path component. If the encompassing directory is missing in
/// either case, it is created.
/// 
/// - Parameters:
///   - source: The remote URL of the file to download.
///   - destination: Either the local directory to place the file in, or the local filename.
public func download(from source: URL, to destination: URL) throws {
    let destinationFile: String
    if destination.hasDirectoryPath {
        try createDirectoryIfMissing(at: destination.path)
        let fileName = source.lastPathComponent
        destinationFile = destination.appendingPathComponent(fileName).path
    } else {
        try createDirectoryIfMissing(at: destination.deletingLastPathComponent().path)
        destinationFile = destination.path
    }

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
    under directory: URL, recurse: Bool = false, filtering extensions: [String]? = nil
) -> [URL] {
    var files: [URL] = []
    do {
        let dirContents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        for content in dirContents {
            if content.hasDirectoryPath && recurse {
                files += collectURLs(under: content, recurse: recurse, filtering: extensions)
            } else if content.isFileURL
                && (extensions == nil
                    || extensions!.contains(dirContents[0].pathExtension.lowercased()))
            {
                files.append(content)
            }
        }
    } catch {
        fatalError("Could not explore this folder: \(error)")
    }
    return files
}

/// Extracts a compressed file to a specified directory. This keys off of either the explicit
/// file extension or one determined from the archive to determine which unarchiving method to use.
/// This optionally deletes the original archive when done.
///
/// - Parameters:
///   - archive: The source archive file, assumed to be on the local filesystem.
///   - localStorageDirectory: A directory that the archive will be unpacked into.
///   - fileExtension: An optional explicitly-specified file extension for the archive, determining
///     how it is unpacked.
///   - deleteArchiveWhenDone: Whether or not the original archive is deleted when the extraction
///     process has been completed. This defaults to false.
public func extractArchive(
    at archive: URL, to localStorageDirectory: URL, fileExtension: String? = nil,
    deleteArchiveWhenDone: Bool = false
) {
    let archivePath = archive.path

    #if os(macOS)
        var binaryLocation = "/usr/bin/"
    #else
        var binaryLocation = "/bin/"
    #endif

    let toolName: String
    let arguments: [String]
    switch fileExtension ?? archive.pathExtension {
    case "gz":
        toolName = "gunzip"
        arguments = [archivePath]
    case "tar.gz", "tgz":
        toolName = "tar"
        arguments = ["xzf", archivePath, "-C", localStorageDirectory.path]
    case "zip":
        binaryLocation = "/usr/bin/"
        toolName = "unzip"
        arguments = [archivePath, "-d", localStorageDirectory.path]
    default:
        printError(
            "Unable to find archiver for extension \(fileExtension ?? archive.pathExtension).")
        exit(-1)
    }
    let toolLocation = "\(binaryLocation)\(toolName)"

    let task = Process()
    task.executableURL = URL(fileURLWithPath: toolLocation)
    task.arguments = arguments
    do {
        try task.run()
        task.waitUntilExit()
    } catch {
        printError("Failed to extract \(archivePath) with error: \(error)")
        exit(-1)
    }

    if FileManager.default.fileExists(atPath: archivePath) && deleteArchiveWhenDone {
        do {
            try FileManager.default.removeItem(atPath: archivePath)
        } catch {
            printError("Could not remove archive, error: \(error)")
            exit(-1)
        }
    }
}
