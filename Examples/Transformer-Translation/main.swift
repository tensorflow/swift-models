//
//  main.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/7/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//


import TensorFlow
import TranslationModels
import Foundation

struct TranslationTask {
    let directoryURL: URL
    var trainDataIterator: IndexingIterator<[TextBatch]>
    var trainData: [TextBatch]
    var textProcessor: TextProcessor
    var sourceVocabSize: Int {
        textProcessor.sourceVocabulary.count
    }
    var targetVocabSize: Int {
        textProcessor.targetVocabulary.count
    }
    var trainDataSize: Int {
        trainData.count
    }
    init(taskDirectoryURL: URL, maxSequenceLength: Int, batchSize: Int) throws {
        self.directoryURL = taskDirectoryURL.appendingPathComponent("Translation")
        let dataURL = directoryURL.appendingPathComponent("data")

        let spanishLanguagePath = dataURL.appendingPathExtension("source")
        try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_es.txt")!, to: spanishLanguagePath)
        let englishPath = dataURL.appendingPathExtension("target")
        try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_en.txt")!, to: englishPath)
        print("loading datasets")
        let loadedSpanish = try TranslationTask.load(fromFile: spanishLanguagePath)
        let loadedEnglish = try TranslationTask.load(fromFile: englishPath)
        
        let tokenizer = BasicTokenizer()
        
        print("preprocessing dataset")
        self.textProcessor = TextProcessor(tokenizer: tokenizer, sourceVocabulary: .init(), targetVocabulary: .init())
        self.trainData = textProcessor.preprocess(source: loadedSpanish, target: loadedEnglish, maxSequenceLength: maxSequenceLength, batchSize: batchSize)

        self.trainDataIterator = self.trainData.makeIterator()
    }
    
    static func load(fromFile fileURL: URL) throws -> [String] {
        try Data(contentsOf: fileURL).withUnsafeBytes {
            $0.split(separator: UInt8(ascii: "\n"))
            .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
    
    mutating func update(model: inout TransformerModel, using optimizer: inout Adam<TransformerModel>) -> Float {
        let batch = withDevice(.gpu) { trainDataIterator.next()! }
        let labels = batch.targetTruth.reshaped(to: [-1])
        let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
        let result = withLearningPhase(.training) { () -> Float in
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(
                    logits: $0.generate(input: batch).reshaped(to: [resultSize, -1]),
                    labels: labels )
            }
            optimizer.update(&model, along: grad)
            return loss.scalarized()
        }
        return result
    }
}

let workspaceURL = URL(fileURLWithPath: "transformer", isDirectory: true,
relativeTo: URL(fileURLWithPath: NSTemporaryDirectory(),
                isDirectory: true))

var translationTask = try TranslationTask(taskDirectoryURL: workspaceURL, maxSequenceLength: 200, batchSize: 40)

var model = TransformerModel(sourceVocabSize: translationTask.sourceVocabSize, targetVocabSize: translationTask.targetVocabSize)

let epochs = 3
var optimizer = Adam.init(for: model, learningRate: 5e-7)
for step in 0..<translationTask.trainDataSize {
    let loss = translationTask.update(model: &model, using: &optimizer)
    print("current loss: \(loss)")
}
