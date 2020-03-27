//
//  File.swift
//  
//
//  Created by Andre Carrera on 3/26/20.
//

import Foundation
import TensorFlow


let BOS_WORD = "<s>"
let EOS_WORD = "</s>"
let BLANK_WORD = "<blank>"

public func subsequentMask(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}

public struct WMT2014EnDe {
    public let directoryURL: URL
    public let trainExamples: [Example]
    public let devExamples: [Example]
//    public let testExamples: [Example]
    public let maxSequenceLength: Int
    public let batchSize: Int

    public typealias ExampleIterator = IndexingIterator<[Example]>
    public typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
    public typealias TrainDataIterator = PrefetchIterator<
        GroupedIterator<MapIterator<RepeatExampleIterator, TextBatch>>
    >
    public typealias DevDataIterator = GroupedIterator<MapIterator<ExampleIterator, TextBatch>>
//    private typealias TestDataIterator = DevDataIterator

    public var trainDataIterator: TrainDataIterator
    public var devDataIterator: DevDataIterator
//    private var testDataIterator: TestDataIterator
}

extension WMT2014EnDe {
    public struct Example {
        public let id: String
        public let sourceSentence: String
        public let targetSentence: String

        public init(id: String, sourceSentence: String, targetSentence: String) {
            self.id = id
            self.sourceSentence = sourceSentence
            self.targetSentence = targetSentence
        }
    }
    
    
    private static let trainGermanURL = URL(string: "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de")!
    private static let trainEnglishURL = URL(string: "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en")!
    
//    internal enum FileType: String {
//        case train = "train"
//        case dev = "dev"
//        case test = "test"
//    }
    
    static func load(fromFile fileURL: URL) throws -> [String] {
        try Data(contentsOf: fileURL).withUnsafeBytes {
            $0.split(separator: UInt8(ascii: "\n"))
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
}

extension WMT2014EnDe {
    /// Tokenized text passage.
    public struct TextBatch: KeyPathIterable {
        /// IDs that correspond to the vocabulary used while tokenizing.
        /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
        public var tokenIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
        // aka src
        
        public var targetTokenIds: Tensor<Int32>
        // aka tgt
        
        /// IDs of the token types (e.g., sentence A and sentence B in BERT).
        /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
        //    public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
        
        /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
        /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
        public var mask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
        
        public var targetMask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
        
        public var targetTruth: Tensor<Int32>
        
        public var tokenCount: Int32
        
        public init(source: Tensor<Int32>, target: Tensor<Int32>, sourcePadId: Int32, targetPadId: Int32) {
            self.tokenIds = source
            self.mask = Tensor<Float>(Tensor(zerosLike: source)
                .replacing(with: Tensor(onesLike: source), where: source .!= Tensor.init(sourcePadId))
                .expandingShape(at: 1))
            
            let rangeExceptLast = 0..<(target.shape[1] - 1)
            self.targetTokenIds = target[0...,rangeExceptLast]
            self.targetTruth = target[0..., 1...]
            self.targetMask = TextBatch.makeStandardMask(target: self.targetTokenIds, pad: targetPadId)
            self.tokenCount = Tensor(zerosLike: targetTruth)
                .replacing(with: Tensor(onesLike: targetTruth), where: self.targetTruth .!= Tensor.init(targetPadId))
                .sum().scalar!
            
        }
        
        public init(tokenIds: Tensor<Int32>, targetTokenIds: Tensor<Int32>, mask: Tensor<Float>, targetMask: Tensor<Float>, targetTruth: Tensor<Int32>, tokenCount: Int32) {
            self.tokenIds = tokenIds
            self.targetTokenIds = targetTokenIds
            self.mask = mask
            self.targetMask = targetMask
            self.targetTruth = targetTruth
            self.tokenCount = tokenCount
        }
        
        static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
            var targetMask = Tensor(zerosLike: target)
                .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
                .expandingShape(at: -2)
            targetMask *= subsequentMask(size: target.shape.last!)
            return Tensor<Float>(targetMask)
        }
    }
}

extension WMT2014EnDe {
    public init(
        mapExample: @escaping (Example) -> TextBatch,
        taskDirectoryURL: URL,
    maxSequenceLength: Int,
    batchSize: Int) throws {
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize
        
        self.directoryURL = taskDirectoryURL.appendingPathComponent("Translation")
        let dataURL = directoryURL.appendingPathComponent("data")
        
        let trainGermanDataPath = dataURL.appendingPathExtension("source")
        let trainEnglishDataPath = dataURL.appendingPathExtension("target")
        print("downloading datasets")
        try maybeDownload(from: WMT2014EnDe.trainGermanURL, to: trainGermanDataPath)
        try maybeDownload(from: WMT2014EnDe.trainEnglishURL, to: trainEnglishDataPath)
        print("loading datasets")
        let loadedGerman = try WMT2014EnDe.load(fromFile: trainGermanDataPath)
        let loadedEnglish = try WMT2014EnDe.load(fromFile: trainEnglishDataPath)
        
        let examples = WMT2014EnDe.combine(sourceSequences: loadedGerman, targetSequences: loadedEnglish)
        (self.trainExamples, self.devExamples) = WMT2014EnDe.split(examples: examples, with: 0.7)
        print("creating batches")
        self.trainDataIterator = trainExamples.shuffled().makeIterator()
        .repeated()
        .shuffled(bufferSize: 1000)
        .map(mapExample)
            .grouped(keyFn: { $0.tokenIds.scalarCount},
                     sizeFn: { _ in batchSize / maxSequenceLength},
                     reduceFn: WMT2014EnDe.reduceDataBatches(_:))
            .prefetched(count: 2)
        
        self.devDataIterator = devExamples.makeIterator()
            .map(mapExample)
            .grouped(keyFn: {$0.tokenIds.scalarCount},
                 sizeFn: { _ in batchSize / maxSequenceLength },
                 reduceFn: WMT2014EnDe.reduceDataBatches(_:))
    }
    
    static func reduceDataBatches(_ batches: [TextBatch]) -> TextBatch {
        return TextBatch(tokenIds: Tensor(batches.map{ $0.tokenIds.squeezingShape(at: 0) }), // this should be fine
                         targetTokenIds: Tensor(batches.map{ $0.targetTokenIds.squeezingShape(at: 0) }),
                         mask: Tensor(batches.map{ $0.mask.squeezingShape(at: 0) }),
                         targetMask: Tensor(batches.map{ $0.targetMask.squeezingShape(at: 0) }),
                         targetTruth: Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) }),
                         tokenCount: batches.map { $0.tokenCount }.reduce(0, +))
    }
    
    static func combine(sourceSequences: [String], targetSequences: [String]) -> [Example] {
        zip(sourceSequences, targetSequences).enumerated().map { (offset: Int, element: Zip2Sequence<[String], [String]>.Element) -> Example in
            Example(id: String(offset), sourceSentence: element.0, targetSentence: element.1)
        }
    }
    
    static func split(examples: [Example], with trainPercent: Double) -> (train: [Example], val: [Example]) {
        let splitIndex = Int(Double(examples.count) * trainPercent)
        let trainSplit = examples[0..<splitIndex]
        let valSplit = examples[splitIndex..<examples.count]
        return (Array(trainSplit), Array(valSplit))
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
public func maybeDownload(from url: URL, to destination: URL) throws {
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

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
