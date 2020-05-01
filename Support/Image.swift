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
@_implementationOnly import STBImage
import TensorFlow

// Image loading and saving is inspired by t-ae's Swim library: https://github.com/t-ae/swim
// and uses the stb_image single-file C headers from https://github.com/nothings/stb .

public struct Image {
    public enum ByteOrdering {
        case bgr
        case rgb
    }

    public enum Colorspace {
        case rgb
        case grayscale
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
        if byteOrdering == .bgr {
            // TODO: Add BGR byte reordering.
            fatalError("BGR byte ordering is currently unsupported.")
        } else {
            guard FileManager.default.fileExists(atPath: url.path) else {
                // TODO: Proper error propagation for this.
                fatalError("File does not exist at: \(url.path).")
            }

            var width: Int32 = 0
            var height: Int32 = 0
            var bpp: Int32 = 0
            guard let bytes = stbi_load(url.path, &width, &height, &bpp, 0) else {
                // TODO: Proper error propagation for this.
                fatalError("Unable to read image at: \(url.path).")
            }

            let data = [UInt8](UnsafeBufferPointer(start: bytes, count: Int(width * height * bpp)))
            stbi_image_free(bytes)
            var loadedTensor = Tensor<UInt8>(
                shape: [Int(height), Int(width), Int(bpp)], scalars: data)
            if bpp == 1 {
                loadedTensor = loadedTensor.broadcasted(to: [Int(height), Int(width), 3])
            }
            self.imageData = .uint8(data: loadedTensor)
        }
    }

    public func save(to url: URL, format: Colorspace = .rgb, quality: Int64 = 95) {
        let outputImageData: Tensor<UInt8>
        let bpp: Int32

        switch format {
        case .grayscale:
            bpp = 1
            switch self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                let lowerBound = data.min(alongAxes: [0, 1])
                let upperBound = data.max(alongAxes: [0, 1])
                let adjustedData = (data - lowerBound) * (255.0 / (upperBound - lowerBound))
                outputImageData = Tensor<UInt8>(adjustedData)
            }
        case .rgb:
            bpp = 3
            switch self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                outputImageData = Tensor<UInt8>(data.clipped(min: 0, max: 255))
            }
        }

        let height = Int32(outputImageData.shape[0])
        let width = Int32(outputImageData.shape[1])
        outputImageData.scalars.withUnsafeBufferPointer { bytes in
            let status = stbi_write_jpg(
                url.path, width, height, bpp, bytes.baseAddress!, Int32(quality))
            guard status != 0 else {
                // TODO: Proper error propagation for this.
                fatalError("Unable to save image to: \(url.path).")
            }
        }
    }

    public func resized(to size: (Int, Int)) -> Image {
        switch self.imageData {
        case let .uint8(data):
            let resizedImage = resize(images: Tensor<Float>(data), size: size, method: .bilinear)
            return Image(tensor: Tensor<UInt8>(resizedImage))
        case let .float(data):
            let resizedImage = resize(images: data, size: size, method: .bilinear)
            return Image(tensor: resizedImage)
        }
    }
}

public func saveImage(
    _ tensor: Tensor<Float>, shape: (Int, Int), size: (Int, Int)? = nil,
    format: Image.Colorspace = .rgb, directory: String, name: String,
    quality: Int64 = 95
) throws {
    try createDirectoryIfMissing(at: directory)

    let channels: Int
    switch format {
    case .rgb: channels = 3
    case .grayscale: channels = 1
    }

    let reshapedTensor = tensor.reshaped(to: [shape.0, shape.1, channels])
    let image = Image(tensor: reshapedTensor)
    let resizedImage = size != nil ? image.resized(to: (size!.0, size!.1)) : image
    let outputURL = URL(fileURLWithPath: "\(directory)\(name).jpg")
    resizedImage.save(to: outputURL, format: format, quality: quality)
}
