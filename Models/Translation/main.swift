//
//  main.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/7/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//


import TensorFlow
import Foundation

let spanishPath = URL(fileURLWithPath: "./spanish")
try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_es.txt")!, to: spanishPath)

let englishPath = URL(fileURLWithPath: "./english")
try maybeDownload(from: URL(string: "https://raw.githubusercontent.com/nickwalton/translation/master/gc_2010-2017_conglomerated_20171009_en.txt")!, to: englishPath)


// 1. separate into vocabulary.

// 2. then create a dataset

// 3. then train.

// 1 and 2 can be done using BERT.preprocess
