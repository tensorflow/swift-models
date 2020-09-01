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
          // TODO: Support RGBA / RGB for PNG.
            switch format {
            case .jpeg: bpp = 3
            case .png: bpp = 4
            }
            
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
            return Image(tensor: Tensor<UInt8>(resizedImage))
        case let .float(data):
            let resizedImage = resize(images: data, size: size, method: .bilinear)
            return Image(tensor: resizedImage)
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
    switch (colorspace, format) {
    case (.rgb, .jpeg): channels = 3
    case (.rgb, .png): channels = 4
    case (.grayscale, _): channels = 1
    }
  
    let reshapedTensor: Tensor<Float>
    if let shape = shape {
        reshapedTensor = tensor.reshaped(to: [shape.0, shape.1, channels])
    } else {
        guard tensor.shape.rank == 3 else {
            fatalError("Input tensor must be of rank 3 (was \(tensor.shape.rank), or a shape must be specified.)")
        }
        reshapedTensor = tensor
    }
    let image = Image(tensor: reshapedTensor)

    let fileExtension: String
    switch format {
    case .jpeg: fileExtension = "jpg"
    case .png: fileExtension = "png"
    }
    let resizedImage = size != nil ? image.resized(to: (size!.0, size!.1)) : image
    let outputURL = URL(fileURLWithPath: "\(directory)/\(name).\(fileExtension)")
    resizedImage.save(to: outputURL, colorspace: colorspace, format: format)
}

public func dumpImageToStringDebugFile(
  _ tensor: Tensor<Float>, directory: String, name: String
) throws {
  try createDirectoryIfMissing(at: directory)
  let height = tensor.shape[0]
  let width = tensor.shape[1]
  let channels = tensor.shape[2]
  let rawValues = tensor.scalars
  
  var debugString = ""
  var index = 0
  for _ in 0..<height {
    for _ in 0..<width {
      if channels == 3 {
        debugString += String(format: "%03.3f,%03.3f,%03.3f ", rawValues[index], rawValues[index + 1], rawValues[index + 2])
        index += 3
      } else {
        debugString += String(format: "%.0f,%.0f,%.0f,%.0f ", rawValues[index], rawValues[index + 1], rawValues[index + 2], rawValues[index + 3])
//        debugString += String(format: "%03.3f,%03.3f,%03.3f,%03.3f ", rawValues[index], rawValues[index + 1], rawValues[index + 2], rawValues[index + 3])
        index += 4
      }
    }
    debugString += "\n"
  }
  
  let outputURL = URL(fileURLWithPath: "\(directory)/\(name).txt")
  try debugString.write(to: outputURL, atomically: true, encoding: .utf8)
}

public typealias Point = (x: Int, y: Int)

/// Draw line using Bresenham's line drawing algorithm
public func drawLine(
  on imageTensor: inout Tensor<Float>,
  from pt1: Point,
  to pt2: Point,
  color: (r: Float, g: Float, b: Float) = (255.0, 255.0, 255.0)
) {
  var pt1 = pt1
  var pt2 = pt2
  let colorTensor = Tensor<Float>([color.r, color.g, color.b])

  // Rearrange points for current octant
  let steep = abs(pt2.y - pt1.y) > abs(pt2.x - pt1.x)
  if steep {
      pt1 = Point(x: pt1.y, y: pt1.x)
      pt2 = Point(x: pt2.y, y: pt2.x)
  }
  if pt2.x < pt1.x {
      (pt1, pt2) = (pt2, pt1)
  }

  // Handle rearranged points
  let dX = pt2.x - pt1.x
  let dY = pt2.y - pt1.y
  let slope = abs(Float(dY) / Float(dX))
  let yStep = dY >= 0 ? 1 : -1

  var error: Float = 0
  var currentY = pt1.y
  for currentX in pt1.x...pt2.x {
    let xIndex = steep ? currentY : currentX
    let yIndex = steep ? currentX : currentY
    if xIndex >= imageTensor.shape[1] || yIndex >= imageTensor.shape[0] {
      break
    }
    imageTensor[yIndex, xIndex] = colorTensor
    error += slope
    if error >= 0.5 {
        currentY += yStep
        error -= 1
    }
  }
}
