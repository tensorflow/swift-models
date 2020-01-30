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
import TensorFlow
import ModelSupport

#if os(Linux)
import FoundationNetworking
#endif

public typealias Activation<Scalar: TensorFlowFloatingPoint> =
    @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

extension KeyPathIterable {
    public mutating func clipByGlobalNorm<Scalar: TensorFlowFloatingPoint>(clipNorm: Scalar) {
        let clipNorm = Tensor<Scalar>(clipNorm)
        var globalNorm = Tensor<Scalar>(zeros: [])
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            globalNorm += self[keyPath: kp].squared().sum()
        }
        globalNorm = sqrt(globalNorm)
        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            self[keyPath: kp] *= clipNorm / max(globalNorm, clipNorm)
        }
    }
}

extension Tensor {
    /// Returns this tensor reshaped to a matrix (i.e., a rank-2 tensor).
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedToMatrix() -> Tensor {
        reshaped(to: [-1, shape[rank - 1]])
    }

    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: TensorShape) -> Tensor {
        reshaped(to: TensorShape(
            originalShape[0..<originalShape.count - 1].dimensions + [shape[rank - 1]]))
    }

    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal func reshapedFromMatrix(originalShape: Tensor<Int32>) -> Tensor {
        reshaped(toShape: Tensor<Int32>(concatenating: [
            originalShape[0..<originalShape.shape[0] - 1],
            Tensor<Int32>([Int32(shape[rank - 1])])
        ]))
    }
}

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///   - from: URL to download data from.
///   - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
internal func maybeDownload(from url: URL, to destination: URL) throws {
#if false
    try ModelSupport.download(from: url, to: destination)
#endif
    if !FileManager.default.fileExists(atPath: destination.path) {
        // Create any potentially missing directories.
        try FileManager.default.createDirectory(
            atPath: destination.deletingLastPathComponent().path,
            withIntermediateDirectories: true)

        // Create the URL session that will be used to download the dataset.
        let semaphore = DispatchSemaphore(value: 0)
        let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)

        // Download the data to a temporary file and then copy that file to
        // the destination path.
        print("Downloading \(url).")
        let task = session.downloadTask(with: url)
        task.resume()

        // Wait for the download to finish.
        semaphore.wait()
    }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let destinationFileUrl: URL
    let semaphore: DispatchSemaphore
    let numBytesFrequency: Int64

    internal var logCount: Int64 = 0

    init(
        destinationFileUrl: URL,
        semaphore: DispatchSemaphore,
        numBytesFrequency: Int64 = 1024 * 1024
    ) {
        self.destinationFileUrl = destinationFileUrl
        self.semaphore = semaphore
        self.numBytesFrequency = numBytesFrequency
    }

    internal func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) -> Void {
        do {
            try FileManager.default.moveItem(at: location, to: destinationFileUrl)
        } catch (let writeError) {
            print("Error writing file \(location.path) : \(writeError)")
        }
        print("Downloaded successfully to \(location.path).")
        semaphore.signal()
    }
}

internal func extract(zipFileAt source: URL, to destination: URL) throws {
    print("Extracting file at '\(source.path)'.")
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", "unzip -d \(destination.path) \(source.path)"]
    try process.run()
    process.waitUntilExit()
}

internal func extract(tarGZippedFileAt source: URL, to destination: URL) throws {
    print("Extracting file at '\(source.path)'.")
    try FileManager.default.createDirectory(
        at: destination,
        withIntermediateDirectories: false)
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", "tar -C \(destination.path) -xzf \(source.path)"]
    try process.run()
    process.waitUntilExit()
}

internal func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
    try Data(contentsOf: fileURL).withUnsafeBytes {
        $0.split(separator: UInt8(ascii: "\n")).map {
            $0.split(separator: UInt8(ascii: "\t"), omittingEmptySubsequences: false)
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
}
