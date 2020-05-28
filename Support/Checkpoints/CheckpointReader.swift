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

// The TensorFlow v2 checkpoint format is described in the following:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/tensor_bundle/tensor_bundle.h
// and consists of an index file (with a `.index` extension) and a series of sharded data files that 
// have the same base file name, but extensions of the form `.data-00001-of-00020`. The index file
// contains key-value pairs of metadata that provide shapes of tensors and where to read in the
// shards to obtain their raw bytes.

import Foundation
import TensorFlow

/// A Swift-native TensorFlow v2 checkpoint reader that can download all checkpoint files from 
/// remote locations and store them in a local temporary directory. This reader has no dependencies
/// on the TensorFlow runtime or libraries.
open class CheckpointReader {
    let header: Tensorflow_BundleHeaderProto
    let metadata: [String: Tensorflow_BundleEntryProto]
    var shardCache: [URL: Data] = [:]

    /// The local checkpoint location.
    public let localCheckpointLocation: URL

    /// The number of tensors stored in the checkpoint.
    public var tensorCount: Int { metadata.count }

    /// The names of the tensors stored in the checkpoint.
    public var tensorNames: [String] { [String](metadata.keys) }

    /// CRC verification during checkpoint loading is enabled by default, but can be selectively
    /// disabled to speed up reads in debug builds or test cases.
    public var isCRCVerificationEnabled: Bool = true

    /// Initializes the checkpoint reader from either a local or remote directory. If remote, 
    /// automatically downloads the checkpoint files into a temporary directory.
    ///
    /// - Parameters:
    ///   - checkpointLocation: Either a URL to the checkpoint files, where the last component is the file 
    ///     base of the checkpoint files, or a URL to an archive containing the checkpoint files.
    ///   - modelName: A distinct name for the model, to ensure that checkpoints with the same base 
    ///     name but for different models don't collide when downloaded.
    public init(checkpointLocation: URL, modelName: String, additionalFiles: [String] = []) throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
            modelName, isDirectory: true)

        // If this is an archive, download if necessary and point to the locally extracted files.
        let finalCheckpointLocation: URL
        if checkpointLocation.isArchive {
            finalCheckpointLocation = try CheckpointReader.downloadAndExtractArchive(
                from: checkpointLocation, to: temporaryDirectory)
        } else {
            finalCheckpointLocation = checkpointLocation
        }

        // If URL that was passed in was a file, or if an archive was downloaded and extracted,
        // read the local checkpoint from the filesystem. Otherwise, download the index first and 
        // determine what other files to download.
        let checkpointBase = finalCheckpointLocation.lastPathComponent
        let indexReader: CheckpointIndexReader
        if finalCheckpointLocation.isFileURL {
            self.localCheckpointLocation = finalCheckpointLocation
            indexReader = try CheckpointIndexReader(
                file: finalCheckpointLocation.appendingPathExtension("index"))
            self.header = try indexReader.readHeader()
        } else {
            let temporaryCheckpointBase = temporaryDirectory.appendingPathComponent(checkpointBase)
            self.localCheckpointLocation = temporaryCheckpointBase
            let localIndexFileLocation = temporaryCheckpointBase.appendingPathExtension("index")
            if FileManager.default.fileExists(atPath: localIndexFileLocation.path) {
                indexReader = try CheckpointIndexReader(file: localIndexFileLocation)
                self.header = try indexReader.readHeader()
            } else {
                // The index file contains the number of shards, so obtain that first.
                try CheckpointReader.downloadIndexFile(
                    from: finalCheckpointLocation, to: temporaryDirectory)
                indexReader = try CheckpointIndexReader(file: localIndexFileLocation)
                self.header = try indexReader.readHeader()

                try CheckpointReader.downloadCheckpointFiles(
                    from: finalCheckpointLocation, to: temporaryDirectory,
                    shards: Int(self.header.numShards), additionalFiles: additionalFiles)
            }
        }

        self.metadata = try indexReader.readAllKeysAndValues()
    }

    /// Downloads an archive file, if necessary, and then extracts it and finds the name of the
    /// index file. The returned URL contains the path and the base name for the checkpoint.
    static func downloadAndExtractArchive(from checkpointLocation: URL, to temporaryDirectory: URL)
        throws -> URL
    {
        func findCheckpointBase(in directory: URL) throws -> URL? {
            guard
                let directoryEnumerator = FileManager.default.enumerator(
                    at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles)
            else {
                return nil
            }
            for case let location as URL in directoryEnumerator {
                let resourceValues = try location.resourceValues(forKeys: [.isDirectoryKey])
                if !(resourceValues.isDirectory ?? false) && location.path.hasSuffix(".index") {
                    return URL(
                        fileURLWithPath: String(location.path.prefix(location.path.count - 6)))
                }
            }

            return nil
        }

        if let checkpointBase = try findCheckpointBase(in: temporaryDirectory) {
            return checkpointBase
        }

        let archiveLocation: URL
        if checkpointLocation.isFileURL {
            archiveLocation = checkpointLocation
            try createDirectoryIfMissing(at: temporaryDirectory.path)
        } else {
            try download(from: checkpointLocation, to: temporaryDirectory)
            archiveLocation = temporaryDirectory.appendingPathComponent(
                checkpointLocation.lastPathComponent)
        }

        extractArchive(at: archiveLocation, to: temporaryDirectory, deleteArchiveWhenDone: false)

        guard let checkpointBase = try findCheckpointBase(in: temporaryDirectory) else {
            fatalError("Unable to find checkpoint index in downloaded archive.")
        }

        return checkpointBase
    }

    /// Constructs the file names for checkpoint components from a base URL and downloads them to a
    /// target directory.
    static func downloadIndexFile(from checkpointLocation: URL, to temporaryDirectory: URL) throws {
        let indexFile = checkpointLocation.appendingPathExtension("index")
        try download(from: indexFile, to: temporaryDirectory)
    }

    /// Constructs the file names for checkpoint components from a base URL and downloads them to a
    /// target directory.
    static func downloadCheckpointFiles(
        from checkpointLocation: URL, to temporaryDirectory: URL, shards: Int,
        additionalFiles: [String]
    ) throws {
        for shard in 0..<shards {
            let shardLocation = self.shardFile(
                location: checkpointLocation, shard: shard, totalShards: shards)
            try download(from: shardLocation, to: temporaryDirectory)
        }
        let checkpointDirectory = checkpointLocation.deletingLastPathComponent()
        for file in additionalFiles {
            let additionalFile = checkpointDirectory.appendingPathComponent(file)
            try download(from: additionalFile, to: temporaryDirectory)
        }
    }

    /// Builds the specific file name from a base URL for a given data shard, out of a total number
    /// of shards.
    static func shardFile(location: URL, shard: Int, totalShards: Int) -> URL {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumIntegerDigits = 5
        formatter.maximumFractionDigits = 0
        formatter.hasThousandSeparators = false
        formatter.usesGroupingSeparator = false
        let currentShard = formatter.string(from: shard as NSNumber)!
        let totalShards = formatter.string(from: totalShards as NSNumber)!
        return location.appendingPathExtension(
            "data-\(currentShard)-of-\(totalShards)"
        )
    }

    /// Returns `true` if the checkpoint contains a tensor with the provided name.
    public func containsTensor(named name: String) -> Bool {
        return metadata[name] != nil
    }

    /// Returns the shape of the tensor with the provided name stored in the checkpoint.
    public func shapeOfTensor(named name: String) -> TensorShape {
        guard let bundleEntry = metadata[name] else {
            fatalError("No tensor named \(name) exists.")
        }
        guard bundleEntry.hasShape else {
            fatalError("Bundle entry for \(name) is missing a shape parameter.")
        }

        return TensorShape(bundleEntry.shape.dim.map { Int($0.size) })
    }

    /// Returns the scalar type of the tensor with the provided name stored in the checkpoint.
    public func scalarTypeOfTensor(named name: String) -> Any.Type {
        guard let bundleEntry = metadata[name] else {
            fatalError("No tensor named \(name) exists.")
        }

        switch bundleEntry.dtype {
        case .dtBool: return Bool.self
        case .dtInt8: return Int8.self
        case .dtUint8: return UInt8.self
        case .dtInt16: return Int16.self
        case .dtUint16: return UInt16.self
        case .dtInt32: return Int32.self
        case .dtUint32: return UInt32.self
        case .dtInt64: return Int64.self
        case .dtUint64: return UInt64.self
        case .dtBfloat16: return BFloat16.self
        case .dtFloat: return Float.self
        case .dtDouble: return Double.self
        case .dtString: return String.self
        default: fatalError("Unsupported tensor data type: \(bundleEntry.dtype)")
        }
    }

    /// Loads and returns the value of the tensor with the provided name stored in the checkpoint.
    public func loadTensor<Scalar: _TensorFlowDataTypeCompatible>(
        named name: String
    ) -> ShapedArray<Scalar> {
        guard let bundleEntry = metadata[name] else {
            fatalError("No tensor named \(name) exists.")
        }
        guard bundleEntry.hasShape else {
            fatalError("Bundle entry for \(name) is missing a shape parameter.")
        }

        let shape = bundleEntry.shape.dim.map { Int($0.size) }
        let shard = Int(bundleEntry.shardID)
        let shardFile = CheckpointReader.shardFile(
            location: localCheckpointLocation, shard: shard, totalShards: Int(header.numShards))

        let shardBytes = shardData(for: shardFile)
        let tensorData = shardBytes.subdata(
            in: Int(bundleEntry.offset)..<Int(bundleEntry.offset + bundleEntry.size))

        if isCRCVerificationEnabled {
            let readCRC32C = bundleEntry.crc32C
            let calculatedCRC32C = tensorData.maskedCRC32C()
            guard readCRC32C == calculatedCRC32C else {
                fatalError(
                    "Tensor \(name) had a bad CRC, expected: \(calculatedCRC32C), read: \(readCRC32C)."
                )
            }
        }

        let scalarArray = tensorData.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Scalar.self))
        }

        return ShapedArray<Scalar>(shape: shape, scalars: scalarArray)
    }

    func shardData(for file: URL) -> Data {
        if let shardBytes = shardCache[file] {
            return shardBytes
        } else {
            do {
                // It is far too slow to read the shards in each time a tensor is accessed, so we
                // read the entire shard into an in-memory cache on first access. A better approach
                // to mapping these files may be needed, because .alwaysMapped doesn't seem to help
                // as much as it should.
                let shardBytes = try Data(contentsOf: file, options: .alwaysMapped)
                shardCache[file] = shardBytes
                return shardBytes
            } catch {
                fatalError("Could not read tensor from \(file.path).")
            }
        }
    }
}

extension Tensorflow_TensorShapeProto {
    var shapeArray: [Int] {
        return self.dim.map { Int($0.size) }
    }
}

extension Data {
    static var crc32CLookupTable: [UInt32] = {
        (0...255).map { index -> UInt32 in
            var lookupValue = UInt32(index)
            for _ in 0..<8 {
                lookupValue =
                    (lookupValue % 2 == 0) ? (lookupValue >> 1) : (0x82F6_3B78 ^ (lookupValue >> 1))
            }
            return lookupValue
        }
    }()

    func crc32C() -> UInt32 {
        var crc32: UInt32 = 0xFFFF_FFFF

        self.withUnsafeBytes { buffer in
            let totalBytes = self.count
            var index = 0
            while index < totalBytes {
                let byte = buffer[index]
                let lookupIndex = Int((crc32 ^ (UInt32(byte) & 0xFF)) & 0xFF)
                crc32 = (crc32 >> 8) ^ Data.crc32CLookupTable[lookupIndex]
                index = index &+ 1
            }
        }

        return crc32 ^ 0xFFFF_FFFF
    }

    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/hash/crc32c.h
    func maskedCRC32C() -> UInt32 {
        let crc32 = self.crc32C()
        let maskDelta: UInt32 = 0xA282_EAD8
        return ((crc32 &>> 15) | (crc32 &<< 17)) &+ maskDelta
    }
}

extension URL {
    var isArchive: Bool {
        switch self.pathExtension {
        case "gz", "zip", "tar.gz", "tgz": return true
        default: return false
        }
    }
}
