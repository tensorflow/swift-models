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
import TensorFlow

// ThreadSafe and concurrentMap based on https://talk.objc.io/episodes/S01E90-concurrent-map
// TODO: build a proper separate module that does the parallel processing
public final class ThreadSafe<A> {
    var _value: A
    let queue = DispatchQueue(label: "ThreadSafe")
    init(_ value: A) { self._value = value }
  
    var value: A {
        return queue.sync { _value }
    }
    func atomically(_ transform: (inout A) -> ()) {
        queue.sync { transform(&self._value) }
    }
}

public extension Array {
    func _concurrentMap<B>(nthreads:Int?=nil, _ transform: (Element) -> B) -> [B] {
        let result = ThreadSafe(Array<B?>(repeating: nil, count: count))
        let nt = nthreads ?? count
        let cs = (count-1)/nt+1
        DispatchQueue.concurrentPerform(iterations: nt) { i in
            let min = i*cs
            let max = min+cs>count ? count : min+cs
            for idx in (min..<max) {
                let element = self[idx]
                let transformed = transform(element)
                result.atomically { $0[idx] = transformed }
            }
        }
        return result.value.map { $0! }
    }
}
