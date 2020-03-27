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
import Datasets

struct WMTTranslationTask {
    var textProcessor: TextProcessor
    var dataset: WMT2014EnDe
    var sourceVocabSize: Int {
        textProcessor.sourceVocabulary.count
    }
    var targetVocabSize: Int {
        textProcessor.targetVocabulary.count
    }
    static let englishVocabURL = URL(string: "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en")!
    static let germanVocabURL = URL(string: "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de")!
    
    init(taskDirectoryURL: URL, maxSequenceLength: Int, batchSize: Int) throws {
        let tokenizer = BasicTokenizer()
        
        let germanVocabPath = taskDirectoryURL.appendingPathExtension("de")
        let englishVocabPath = taskDirectoryURL.appendingPathExtension("en")
        
        try maybeDownload(from: WMTTranslationTask.germanVocabURL, to: germanVocabPath)
        try maybeDownload(from: WMTTranslationTask.englishVocabURL, to: englishVocabPath)
        
        let sourceVocabulary = try Vocabulary(fromFile: germanVocabPath)
        let targetVocabulary = try Vocabulary(fromFile: englishVocabPath)
        
        self.textProcessor = TextProcessor(tokenizer: tokenizer, sourceVocabulary: sourceVocabulary ,targetVocabulary: targetVocabulary, maxSequenceLength: maxSequenceLength, batchSize: batchSize)
        self.dataset = try WMT2014EnDe(mapExample: self.textProcessor.preprocess, taskDirectoryURL: taskDirectoryURL, maxSequenceLength: maxSequenceLength, batchSize: batchSize)
    }
    
    static func load(fromFile fileURL: URL) throws -> [String] {
        try Data(contentsOf: fileURL).withUnsafeBytes {
            $0.split(separator: UInt8(ascii: "\n"))
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
        }
    }
    
    mutating func update(model: inout TransformerModel, using optimizer: inout Adam<TransformerModel>, for batch: WMT2014EnDe.TextBatch) -> Float {
        let labels = batch.targetTruth.reshaped(to: [-1])
        let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
        let padIndex = Int32(textProcessor.targetVocabulary.id(forToken: "<blank>")!)
        let result = withLearningPhase(.training) { () -> Float in
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(logits: $0.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex)
            }
            optimizer.update(&model, along: grad)
            return loss.scalarized()
        }
        return result
    }
    
    /// returns validation loss
    func validate(model: inout TransformerModel, for batch: WMT2014EnDe.TextBatch) -> Float {
        let labels = batch.targetTruth.reshaped(to: [-1])
        let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
        let padIndex = Int32(textProcessor.targetVocabulary.id(forToken: "<blank>")!)
        let result = withLearningPhase(.inference) { () -> Float in
            softmaxCrossEntropy(logits: model.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex).scalarized()
        }
        return result
    }
}

let workspaceURL = URL(fileURLWithPath: "transformer", isDirectory: true,
                       relativeTo: URL(fileURLWithPath: NSTemporaryDirectory(),
                                       isDirectory: true))

var translationTask = try WMTTranslationTask(taskDirectoryURL: workspaceURL, maxSequenceLength: 50, batchSize: 150)

var model = TransformerModel(sourceVocabSize: translationTask.sourceVocabSize, targetVocabSize: translationTask.targetVocabSize)

func greedyDecode(model: TransformerModel, input: WMT2014EnDe.TextBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input)
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    for _ in 0..<maxLength {
        let decoderInput = WMT2014EnDe.TextBatch(tokenIds: input.tokenIds,
                                     targetTokenIds: ys,
                                     mask: input.mask,
                                     targetMask: Tensor<Float>(subsequentMask(size: ys.shape[1])),
                                     targetTruth: input.targetTruth,
                                     tokenCount: input.tokenCount)
        let out = model.decode(input: decoderInput, memory: memory)
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1)
    }
    return ys
}

let epochs = 3
var optimizer = Adam.init(for: model, learningRate: 5e-4)
for epoch in 0..<epochs {
    print("Start epoch \(epoch)")
    var iterator = translationTask.dataset.trainDataIterator
    for step in 0... { // TODO getCount!
        let batch = withDevice(.cpu) { iterator.next()! }
        let loss = translationTask.update(model: &model, using: &optimizer, for: batch)
        print("current loss at step \(step): \(loss)")
        
        if step % 100 == 0 {
            var validationLosses = [Float]()
            var valIterator = translationTask.dataset.devDataIterator
            for _ in 0... {
                let valBatch = withDevice(.cpu){ valIterator.next()! }
                let loss = translationTask.validate(model: &model, for: valBatch)
                validationLosses.append(loss)
            }
            let averageLoss = validationLosses.reduce(0, +) / Float(validationLosses.count)
            print("Average validation loss at step \(step): \(averageLoss)")
        }
    }
}

//let batch = translationTask.dataset.devExamples[0]
//let exampleIndex = 1
//let source = WMT2014EnDe.TextBatch(tokenIds: batch.tokenIds[exampleIndex].expandingShape(at: 0),
//                       targetTokenIds: batch.targetTokenIds[exampleIndex].expandingShape(at: 0),
//                       mask: batch.mask[exampleIndex].expandingShape(at: 0),
//                       targetMask: batch.targetMask[exampleIndex].expandingShape(at: 0),
//                       targetTruth: batch.targetTruth[exampleIndex].expandingShape(at: 0),
//                       tokenCount: batch.tokenCount)
//let startId = Int32(translationTask.textProcessor.targetVocabulary.id(forToken: "<s>")!)
//let endId = Int32(translationTask.textProcessor.targetVocabulary.id(forToken: "</s>")!)
//
//Context.local.learningPhase = .inference
//
//let out = greedyDecode(model: model, input: source, maxLength: 50, startSymbol: startId)
//
//func decode(tensor: Tensor<Float>, vocab: Vocabulary) -> String {
//    var words = [String]()
//    for scalar in tensor.scalars {
//        if Int(scalar) == endId {
//            break
//        } else if let token = vocab.token(forId: Int(scalar)) {
//            words.append(token)
//        }
//    }
//    return words.joined(separator: " ")
//}
