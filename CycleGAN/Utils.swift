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

import TensorFlow

public typealias Tensorf = Tensor<Float>

#if os(macOS)
    func random() -> UInt32 {
        arc4random()
    }
#endif

public protocol FeatureChannelInitializable: Layer {
    init(featureCount: Int)
}

extension BatchNorm: FeatureChannelInitializable {
    public init(featureCount: Int) {
        self.init(featureCount: featureCount, axis: -1, momentum: 0.99, epsilon: 0.001)
    }
}

extension InstanceNorm2D: FeatureChannelInitializable {
    public init(featureCount: Int) {
        self.init(featureCount: featureCount, epsilon: Tensor(1e-5))
    }
}
