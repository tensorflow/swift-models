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

import Foundation

func currentRunId(logDir: URL, runIdFileName: String = ".run") -> Int {
    var runId = 0

    let runIdURL = logDir.appendingPathComponent(runIdFileName, isDirectory: false)
    if let data = try? Data(contentsOf: runIdURL) {
        runId = data.withUnsafeBytes { $0.baseAddress?.assumingMemoryBound(to: Int.self).pointee ?? 0 }
    }

    runId += 1

    let data = Data(bytes: &runId, count: MemoryLayout<Int>.stride)

    try? data.write(to: runIdURL)

    return runId
}
