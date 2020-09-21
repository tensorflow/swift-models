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
        case rgba
        case grayscale
    }
  
    public enum Format {
        case jpeg(quality: Float)
        case png
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

    public init(_ tensor: Tensor<UInt8>) {
        self.imageData = .uint8(data: tensor)
    }

    public init(_ tensor: Tensor<Float>) {
        self.imageData = .float(data: tensor)
    }

    public init(contentsOf url: URL, byteOrdering: ByteOrdering = .rgb) {
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

    public func save(to url: URL, colorspace: Colorspace = .rgb, format: Format = .jpeg(quality: 95)) {
        let outputImageData: Tensor<UInt8>
        let bpp: Int32

        switch colorspace {
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
        case .rgba:
            bpp = 4
            switch self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                outputImageData = Tensor<UInt8>(data.clipped(min: 0, max: 255))
            }
        }
        
        let height = Int32(outputImageData.shape[0])
        let width = Int32(outputImageData.shape[1])
        outputImageData.scalars.withUnsafeBufferPointer { bytes in
            switch format {
            case let .jpeg(quality):
                let status = stbi_write_jpg(
                    url.path, width, height, bpp, bytes.baseAddress!, Int32(round(quality)))
                guard status != 0 else {
                    // TODO: Proper error propagation for this.
                    fatalError("Unable to save image to: \(url.path).")
                }
            case .png:
                let status = stbi_write_png(
                    url.path, width, height, bpp, bytes.baseAddress!, 0)
                guard status != 0 else {
                    // TODO: Proper error propagation for this.
                    fatalError("Unable to save image to: \(url.path).")
                }
            }
        }
    }

    public func resized(to size: (Int, Int)) -> Image {
        switch self.imageData {
        case let .uint8(data):
            let resizedImage = resize(images: Tensor<Float>(data), size: size, method: .bilinear)
            return Image(Tensor<UInt8>(resizedImage))
        case let .float(data):
            let resizedImage = resize(images: data, size: size, method: .bilinear)
            return Image(resizedImage)
        }
    }
  
    func premultiply(_ input: Tensor<Float>) -> Tensor<Float> {
        let alphaChannel = input.slice(
            lowerBounds: [0, 0, 3], sizes: [input.shape[0], input.shape[1], 1])
        let colorComponents = input.slice(
            lowerBounds: [0, 0, 0], sizes: [input.shape[0], input.shape[1], 3])
        let adjustedColorComponents = colorComponents * alphaChannel / 255.0
        return Tensor(concatenating: [adjustedColorComponents, alphaChannel], alongAxis: 2)
    }
    
    public func premultipliedAlpha() -> Image {
        switch self.imageData {
        case let .uint8(data):
            guard data.shape[2] == 4  else { return self }
            return Image(premultiply(Tensor<Float>(data)))
        case let .float(data):
            guard data.shape[2] == 4  else { return self }
            return Image(premultiply(data))
        }
    }
}

public func saveImage(
    _ tensor: Tensor<Float>, shape: (Int, Int)? = nil, size: (Int, Int)? = nil,
    colorspace: Image.Colorspace = .rgb, directory: String, name: String,
    format: Image.Format = .jpeg(quality: 95)
) throws {
    try createDirectoryIfMissing(at: directory)

    let channels: Int
    switch colorspace {
    case .rgb: channels = 3
    case .rgba: channels = 4
    case .grayscale: channels = 1
    }
  
    let reshapedTensor: Tensor<Float>
    if let shape = shape {
        reshapedTensor = tensor.reshaped(to: [shape.0, shape.1, channels])
    } else {
        guard tensor.shape.rank == 3 else {
            fatalError("Input tensor must be of rank 3 (was \(tensor.shape.rank), or a shape must be specified.)")
        }
        if tensor.shape[2] > channels {
            // Need to premultiply alpha channel before saving as RGB.
            let alphaChannel = tensor.slice(
                lowerBounds: [0, 0, 3], sizes: [tensor.shape[0], tensor.shape[1], 1])
            let colorComponents = tensor.slice(
                lowerBounds: [0, 0, 0], sizes: [tensor.shape[0], tensor.shape[1], 3])
            reshapedTensor = (255.0 - alphaChannel) + colorComponents
        } else {
            reshapedTensor = tensor
        }
    }
    let image = Image(reshapedTensor)

    let fileExtension: String
    switch format {
    case .jpeg: fileExtension = "jpg"
    case .png: fileExtension = "png"
    }
    let resizedImage = size != nil ? image.resized(to: (size!.0, size!.1)) : image
    let outputURL = URL(fileURLWithPath: "\(directory)/\(name).\(fileExtension)")
    resizedImage.save(to: outputURL, colorspace: colorspace, format: format)
}
