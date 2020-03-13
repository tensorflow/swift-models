//
//  main.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/7/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//


import TensorFlow
import TranslationModels
//import Text
//import Foundation

//let spanishPath = URL(fileURLWithPath: "./spanish")
//try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_es.txt")!, to: spanishPath)
//
//let englishPath = URL(fileURLWithPath: "./english")
//try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_en.txt")!, to: englishPath)


// 1. separate into vocabulary.

// 2. then create a dataset

// 3. then train.

// 1 and 2 can be done using BERT.preprocess

let spanishSource = ["Hola, yo soy santiago.", "Estoy feliz ahorita."]
let englishSource = ["Hello, my name is james.", "I'm happy right now."]

let tokenizer = BasicTokenizer()

let MAX_LENGTH = 200

let batchSize = 1000

var textProcessor = TextProcessor(tokenizer: tokenizer, sourceVocabulary: .init(), targetVocabulary: .init())

let batches: [TextBatch] = textProcessor.preprocess(source: spanishSource, target: englishSource, maxSequenceLength: MAX_LENGTH, batchSize: batchSize)

let transformer: TransformerModel = TransformerModel(sourceVocabSize: textProcessor.sourceVocabulary.count, targetVocabSize: textProcessor.targetVocabulary.count)


let epochs = 3

for _ in 0..<epochs {
    
    for textBatch in batches {
        let output = transformer.callAsFunction(textBatch)
        
    }
    
}
//let transformerModel = TransformerModel
