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

import Foundation
import TensorFlow

public struct Image {
    public enum ByteOrdering {
        case bgr
        case rgb
    }
    
    enum ImageTensor {
        case float(data: Tensor<Float>)
        case uint8(data: Tensor<UInt8>)
    }
    
    let imageData: ImageTensor
    
    public var tensor: Tensor<Float> {
        switch self.imageData {
        case let .float(data): return data
        case let .uint8(data): return Tensor<Float>(data)
        }
    }
    
    public init(tensor: Tensor<UInt8>) {
        self.imageData = .uint8(data: tensor)
    }
    
    public init(tensor: Tensor<Float>) {
        self.imageData = .float(data: tensor)
    }
    
    public init(jpeg url: URL, byteOrdering: ByteOrdering = .rgb) {
        let loadedData = _Raw.readFile(filename: StringTensor(url.absoluteString))
        let loadedJpeg = _Raw.decodeJpeg(contents: loadedData, channels: 3, dctMethod: "")
        if byteOrdering == .bgr {
            self.imageData = .uint8(
                data: _Raw.reverse(loadedJpeg, dims: Tensor<Bool>([false, false, false, true])))
        } else {
            self.imageData = .uint8(data: loadedJpeg)
        }
    }
    
    public func save(to url: URL, format: _Raw.Format = .rgb, quality: Int64 = 95) {
        let outputImageData: Tensor<UInt8>
        switch format {
        case .grayscale:
            switch self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                let lowerBound = data.min(alongAxes: [0, 1])
                let upperBound = data.max(alongAxes: [0, 1])
                let adjustedData = (data - lowerBound) * (255.0 / (upperBound - lowerBound))
                outputImageData = Tensor<UInt8>(adjustedData)
            }
        case .rgb:
            switch self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                outputImageData = Tensor<UInt8>(
                    _Raw.clipByValue(t: data, clipValueMin: Tensor(0), clipValueMax: Tensor(255)))
            }
        default:
            print("Image saving isn't supported for the format \(format).")
            exit(-1)
        }
        
        let encodedJpeg = _Raw.encodeJpeg(
            image: outputImageData, format: format, quality: quality, xmpMetadata: "")
        _Raw.writeFile(filename: StringTensor(url.absoluteString), contents: encodedJpeg)
    }
    
    public func resized(to size: (Int, Int)) -> Image {
        switch self.imageData {
        case let .uint8(data):
            return Image(
                tensor: _Raw.resizeBilinear(
                    images: Tensor<UInt8>(data.rankLifted()),
                    size: Tensor<Int32>([Int32(size.0), Int32(size.1)])).squeezingShape(at: 0))
        case let .float(data):
            return Image(
                tensor: _Raw.resizeBilinear(
                    images: Tensor<Float>(data.rankLifted()),
                    size: Tensor<Int32>([Int32(size.0), Int32(size.1)])).squeezingShape(at: 0))
        }

    }
}

public func saveImage(_ tensor: Tensor<Float>, size: (Int?, Int?) = (nil, nil),
                      format: _Raw.Format = .rgb, directory: String, name: String,
                      quality: Int64 = 95) throws {
    try createDirectoryIfMissing(at: directory)
    let image = Image(tensor: tensor)
    let resizedImage = size.0 != nil && size.1 != nil ? image.resized(to: (size.0!, size.1!)) : image
    let outputURL = URL(fileURLWithPath: "\(directory)\(name).jpg")
    resizedImage.save(to: outputURL, format: format, quality: quality)
}
