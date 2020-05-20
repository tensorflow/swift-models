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

import ModelSupport
import TextModels

let gpt: GPT2 = try GPT2()

// Set temperature.
if CommandLine.arguments.count >= 2 {
  guard let temperature = Float(CommandLine.arguments[1]) else {
    fatalError("Could not parse command line argument '\(CommandLine.arguments[1])' as a float")
  }
  gpt.temperature = temperature
} else {
  gpt.temperature = 1.0
}

// Use seed text.
if CommandLine.arguments.count == 3 {
    gpt.seed = gpt.embedding(for: CommandLine.arguments[2])
    print(CommandLine.arguments[2], terminator: "")
}

for _ in 0..<100 {
    do {
        try print(gpt.generate(), terminator: "")
    } catch GPT2.GPT2Error.invalidEncoding(let id) {
        print("ERROR: Invalid encoding: \(id)")
    } catch BytePairEncoder.BPEError.unsupported {
        print(" ", terminator: "")
    } catch {
        fatalError("ERROR: Unexpected error: \(error).")
    }
}
print()

// The following illustrates how to write out a checkpoint from this model and read it back in.
/*
import Foundation
let temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("Transformer")
try gpt.writeCheckpoint(to: temporaryDirectory, name: "model2.ckpt")

let recreatedmodel = try GPT2(checkpoint: temporaryDirectory.appendingPathComponent("model2.ckpt"))
*/
