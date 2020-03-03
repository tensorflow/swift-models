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

let gpt: GPT2 = try GPT2()

// Set temperature.
gpt.temperature = CommandLine.arguments.count >= 2
                      ? Float(CommandLine.arguments[1])!
                      : Float(1.0)

// Use seed text.
if CommandLine.arguments.count == 3 {
    gpt.seed = gpt.embedding(for: CommandLine.arguments[2])
    print(CommandLine.arguments[2], terminator: "")
}

for _ in 0..<100 {
    do {
      try print(gpt.generate(), terminator: "")
    } catch {
      continue
    }
}
print()

// The following illustrates how to write out a checkpoint from this model and load it back in.
/*
var tensors = [String: Tensor<Float>]()
recursivelyObtainTensors(model, scope: "model", tensors: &tensors, separator: "/")

let writer = CheckpointWriter(tensors: tensors)
try writer.write(to: temporaryDirectory, name: "model2.ckpt")

let reader2 = try CheckpointReader(
    checkpointLocation: temporaryDirectory.appendingPathComponent("model2.ckpt"),
    modelName: "Transformer",
    additionalFiles: [])

let recreatedmodel = TransformerLM(reader: reader2, config: config, scope: "model")
*/
