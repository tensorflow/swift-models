//
//  File.swift
//  
//
//  Created by Andre Carrera on 3/26/20.
//

import Foundation
import TensorFlow
import ModelSupport

let BOS_WORD = "<s>"
let EOS_WORD = "</s>"
let BLANK_WORD = "<blank>"

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
        GroupedIterator<MapIterator<RepeatExampleIterator, TranslationBatch>>
    >
    public typealias DevDataIterator = GroupedIterator<MapIterator<ExampleIterator, TranslationBatch>>
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
    public init(
        mapExample: @escaping (Example) -> TranslationBatch,
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
    
    static func reduceDataBatches(_ batches: [TranslationBatch]) -> TranslationBatch {
        return TranslationBatch(tokenIds: Tensor(batches.map{ $0.tokenIds.squeezingShape(at: 0) }), // this should be fine
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
