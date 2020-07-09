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

import Checkpoints
import Foundation
import ModelSupport
import TensorFlow

extension GPT2 {
  public func writeCheckpoint(
    to location: URL, name: String, fileSystem: FileSystem = FoundationFileSystem()
  ) throws {
    try model.writeCheckpoint(to: location, name: name, fileSystem: fileSystem)
  }
}

extension TransformerLM: Checkpointable {
  public var ignoredTensorPaths: Set<String> {
    return ["Attention.scale"]
  }
  
  public var tensorNameMap: (String) -> String {
    return { name in
      let components = name.split(separator: "/")
      guard components.count >= 1 else { return name }
      let normNames = ["offset": "b", "scale": "g"]
      let denseNames = ["weight": "w", "bias": "b"]
      let feedForwardNames = ["dense1": "c_fc", "dense2": "c_proj"]
      let selfAttentionNames = ["wqkv": "c_attn", "wo": "c_proj"]

      switch components[0] {
      case "layers":
        let layerIndex = Int(components[1].dropFirst().dropLast())!
        let base = "model/h\(layerIndex)"
        switch components[2] {
        case "feedForward":
          return
            "\(base)/mlp/\(feedForwardNames[String(components[3])]!)/\(denseNames[String(components[5])]!)"
        case "feedForwardNorm":
          return "\(base)/ln_2/\(normNames[String(components[3])]!)"
        case "selfAttention":
          return
            "\(base)/attn/\(selfAttentionNames[String(components[3])]!)/\(denseNames[String(components[5])]!)"
        case "selfAttentionNorm":
          return "\(base)/ln_1/\(normNames[String(components[3])]!)"
        default:
          return name
        }
      case "norm":
        return "model/ln_f/\(normNames[String(components[1])]!)"
      case "positionalEmbeddings":
        return "model/wpe"
      case "embedding":
        return "model/wte"
      default:
        return name
      }
    }
  }
}
