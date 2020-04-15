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
import ModelSupport
import TensorFlow
import Batcher

public class PairedImages {
    public struct ImagePair: Collatable {
        public init(collating: [PairedImages.ImagePair]) {
            self.source = .init(stacking: collating.map(\.source))
            self.target = .init(stacking: collating.map(\.target))
        }
        
        public init(source: Tensorf, target: Tensorf) {
            self.source = source
            self.target = target
        }
        
        var source: Tensorf
        var target: Tensorf
    }
    var batcher: Batcher<[ImagePair]>
    
    public init(folderAURL: URL, folderBURL: URL) throws {
        let folderAContents = try FileManager.default
                                             .contentsOfDirectory(at: folderAURL,
                                                                  includingPropertiesForKeys: [.isDirectoryKey],
                                                                  options: [.skipsHiddenFiles])
                                             .filter { $0.pathExtension == "jpg" }

        let imageTensors = folderAContents.map { (url: URL) -> ImagePair in
            let tensorA = Image(jpeg: url).tensor / 127.5 - 1.0
            
            let tensorBImageURL = folderBURL.appendingPathComponent(url.lastPathComponent.replacingOccurrences(of: "_A.jpg", with: "_B.jpg"))
            let tensorB = Image(jpeg: tensorBImageURL).tensor / 127.5 - 1.0
            
            return ImagePair(source: tensorA, target: tensorB)
        }
        
        self.batcher = Batcher(on: imageTensors,
                               batchSize: 1,
                               shuffle: true)
    }
}
