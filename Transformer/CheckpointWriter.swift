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

import TensorFlow

protocol ExportableLayer {
    var nameMappings: [String: String] { get }
}

extension TransformerLM: ExportableLayer {
    var nameMappings: [String: String] {
        ["layers": "", "norm": "ln_f", "embedding": "", "positionalEmbeddings": "wpe"]
    }
}

extension Embedding: ExportableLayer {
    var nameMappings: [String: String] { ["weight": "wte"] }
}

extension LayerNorm: ExportableLayer {
    var nameMappings: [String: String] { ["offset": "b", "scale": "g"] }
}

extension Dense: ExportableLayer {
    var nameMappings: [String: String] { ["weight": "w", "bias": "b"] }
}

extension TimeDistributed: ExportableLayer {
    var nameMappings: [String: String] { ["dense": ""] }
}

extension FeedForward: ExportableLayer {
    var nameMappings: [String: String] { ["dense1": "c_fc", "dense2": "c_proj"] }
}

extension MultiHeadAttention: ExportableLayer {
    var nameMappings: [String: String] { ["wqkv": "c_attn", "wo": "c_proj"] }
}

extension EncoderLayer: ExportableLayer {
    var nameMappings: [String: String] {
        [
            "selfAttention": "attn", "selfAttentionNorm": "ln_1", "feedForward": "mlp",
            "feedForwardNorm": "ln_2",
        ]
    }
}

extension Array: ExportableLayer {
    var nameMappings: [String: String] { ["h": "h"] }
}

func recursivelyObtainTensors(
    _ obj: Any, scope: String? = nil, tensors: inout [String: Tensor<Float>], separator: String
) {
    let m = Mirror(reflecting: obj)
    let nameMappings: [String: String]
    if let exportableLayer = obj as? ExportableLayer {
        nameMappings = exportableLayer.nameMappings
    } else {
        nameMappings = [:]
    }

    var repeatedLabels: [String: Int] = [:]
    func suffix(for label: String) -> String {
        if let currentSuffix = repeatedLabels[label] {
            repeatedLabels[label] = currentSuffix + 1
            return "\(currentSuffix + 1)"
        } else {
            repeatedLabels[label] = 0
            return "0"
        }
    }

    let hasSuffix = (m.children.first?.label == nil)

    var path = scope
    for child in m.children {
        let label = child.label ?? "h"

        if let remappedLabel = nameMappings[label] {
            let labelSuffix = hasSuffix ? suffix(for: remappedLabel) : ""
            let conditionalSeparator = remappedLabel == "" ? "" : separator

            path = (scope != nil ? scope! + conditionalSeparator : "") + remappedLabel + labelSuffix
            if let tensor = child.value as? Tensor<Float> {
                tensors[path!] = tensor
            }
        }
        recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
    }
}
