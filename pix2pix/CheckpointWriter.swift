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
import Checkpoints
import Tensor
import CTensorFlow
import Foundation

protocol ExportableLayer {
    var nameMappings: [String: String] { get }
}

extension NetG: ExportableLayer {
    var nameMappings: [String: String] {
        ["module": "module"]
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

extension Conv2D: ExportableLayer {
    var nameMappings: [String : String] { ["filter": "fil", "bias":"bias"]}
}

extension TransposedConv2D: ExportableLayer {
    var nameMappings: [String : String] { ["filter": "fil", "bias":"bias"]}
}

extension BatchNorm: ExportableLayer {
    var nameMappings: [String : String] { ["axis": "axis", "momentum": "mom", "offset": "off", "scale": "sc", "epsilon": "eps", "runningMean": "rmean", "runningVariance": "rvar"] }
}

extension Dropout: ExportableLayer {
    var nameMappings: [String : String] { ["probability": "prob"]}

}

extension UNetSkipConnection: ExportableLayer {
    var nameMappings: [String: String] {
        [
            "downConv": "dc", "downNorm": "dn", "upConv": "uc", "upNorm": "un", "dropOut": "drop", "submodule": "submod"
        ]
    }
}

extension UNetSkipConnectionInnermost: ExportableLayer {
    var nameMappings: [String: String] {
         [
            "downConv": "dc", "upConv": "uc", "upNorm": "un"
         ]
    }
}

extension UNetSkipConnectionOutermost: ExportableLayer {
    var nameMappings: [String: String] {
         [
            "downConv": "dc", "upConv": "uc", "submodule": "submod"
         ]
     }
}

extension Array: ExportableLayer {
    var nameMappings: [String: String] { ["h": "h"] }
}

public func recursivelyObtainTensors (
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
            else if let tensor = child.value as? TensorFlow.Parameter<Swift.Float> {
                let newTensor = tensor.value
                tensors[path!] = newTensor
            } else {
                // TODO: need to add facility to read/write scalars
                // not a tensor but still capture and convert to tensor
                if child.value is Int {
                    guard let val = child.value as? Int else {
                        fatalError()
                    }
                    let cast = Float(val)
                    let tens = Tensor<Float>([cast])
                    tensors[path!] = tens
                } else if child.value is Float {
                    guard let val = child.value as? Float else {
                        fatalError()
                    }
                    let tens = Tensor<Float>([val])
                    tensors[path!] = tens
                } else if child.value is Double {
                    guard let val = child.value as? Double else {
                        fatalError()
                    }
                    let cast = Float(val)
                    let tens = Tensor<Float>([cast])
                    tensors[path!] = tens
                }
            }
        }
        recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
    }
}



