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

public enum TrainingMetrics {
    case loss
}

public class TrainingStatistics {
    var batchLosses: [Float] = []
    
    func averageLoss() -> Float {
        return batchLosses.reduce(0.0, +) / Float(batchLosses.count)
    }

    public func record<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
        switch event {
        case .epochStart:
            batchLosses.removeAll()
        case .batchEnd:
            if let loss = loop.lastLoss {
                batchLosses.append(loss.scalarized())
            }
        default:
            return
        }
    }
}
