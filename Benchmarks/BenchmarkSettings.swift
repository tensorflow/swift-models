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
    let batches: Int
    let batchSize: Int
    let iterations: Int
    let epochs: Int

    func withDefaults(_ defaults: BenchmarkSettings) -> BenchmarkSettings {
        let newBatches = batches == -1 ? defaults.batches : batches
        let newBatchSize = batchSize == -1 ? defaults.batchSize : batchSize
        let newIterations = iterations == -1 ? defaults.iterations : iterations
        let newEpochs = epochs == -1 ? defaults.epochs : epochs
        return BenchmarkSettings(
            batches: newBatches, batchSize: newBatchSize,
            iterations: newIterations, epochs: newEpochs)
    }
}
