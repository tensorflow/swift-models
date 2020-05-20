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

struct BenchmarkSettings: Codable {
    let duration: BenchmarkDuration
    let batchSize: Int
    let iterations: Int
    let warmupBatches: Int
    let synthetic: Bool
    let backend: Backend
}

enum Backend: String, Codable {
    case eager = "eager"
    case x10 = "x10"
}

enum BenchmarkDuration {
    case batches(_ value: Int)
    case epochs(_ value: Int)
}

extension BenchmarkDuration: Codable {
    enum CodingKeys: String, CodingKey {
        case epochs, batches
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try container.decodeIfPresent(Int.self, forKey: .batches) {
            self = .batches(value)
        } else if let value = try container.decodeIfPresent(Int.self, forKey: .epochs) {
            self = .epochs(value)
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: decoder.codingPath,
                    debugDescription: "Could not decode the duration."))
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .batches(let value):
            try container.encode(value, forKey: .batches)
        case .epochs(let value):
            try container.encode(value, forKey: .epochs)
        }
    }
}
