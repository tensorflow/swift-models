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

/// A high-level representation of an image, encapsulating common image saving, loading, and
/// manipulation operations. The loading and saving functionality is inspired by
/// [t-ae's Swim library](https://github.com/t-ae/swim) and uses
/// [the stb_image single-file C headers](https://github.com/nothings/stb) .
public struct Image {
    public enum ByteOrdering {
        case bgr
        case rgb
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

    /// Returns a floating-point tensor that represents the content of the image. Internally, images
    /// can be represented by either floating-point values or bytes for efficiency, and this
    /// provides a unified means of extracting a floating-point tensor from that storage.
    public var tensor: Tensor<Float> {
        switch self.imageData {
        case let .float(data): return data
        case let .uint8(data): return Tensor<Float>(data)
        }
    }

    /// Initializes an image from a rank-3 tensor of bytes. The bytes are assumed to be in the range
    /// of 0 - 255.
    public init(_ tensor: Tensor<UInt8>) {
        precondition(tensor.rank == 3)
        self.imageData = .uint8(data: tensor)
    }

    /// Initializes an image from a rank-3 tensor of floats. The floats are assumed to be in the
    /// range of 0.0 - 255.0.
    public init(_ tensor: Tensor<Float>) {
        precondition(tensor.rank == 3)
        self.imageData = .float(data: tensor)
    }

    /// Loads an image from a local file, with the image format determined from the file extension.
    /// - Parameters:
    ///   - url: The location of the image file.
    ///   - byteOrdering: Whether to treat the image as having RGB (default) or BGR channel ordering.
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

    /// Saves an image to a local file.
    /// - Parameters:
    ///   - url: The destination for the image file.
    ///   - format: The file format, with associated parameters. The default is a JPEG at 95% quality.
    public func save(to url: URL, format: Format = .jpeg(quality: 95)) {
        let outputImageData: Tensor<UInt8>
        switch self.imageData {
        case let .uint8(data):
            outputImageData = data
        case let .float(data):
            outputImageData = Tensor<UInt8>(data.clipped(min: 0, max: 255))
        }
        let bpp: Int32 = Int32(outputImageData.shape[2])
        
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

    /// Returns a resized image, resampled using bilinear filtering.
    /// - Parameters:
    ///   - size: A tuple representing the width, height of the resulting image.
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
    
    /// Returns a version of this image with premultiplied alpha.
    public func premultipliedAlpha() -> Image {
        switch self.imageData {
        case let .uint8(data):
            guard data.shape[2] == 4 else { return self }
            return Image(premultiply(Tensor<Float>(data)))
        case let .float(data):
            guard data.shape[2] == 4 else { return self }
            return Image(premultiply(data))
        }
    }
}

public extension Tensor where Scalar == Float {
  /// Saves the tensor as a still image file. This must be a rank-3 tensor, with channels in the
  /// 0.0 - 255.0 range.
  /// - Parameters:
  ///   - directory: The target directory to host the image file. If it does not exist, it
  ///     will be created.
  ///   - name: The name of the resulting image file, without extension.
  ///   - format: The file format, with associated parameters. The default is a JPEG at 95% quality.
  func saveImage(
    directory: String, name: String, format: Image.Format = .jpeg(quality: 95)
  ) throws {
    precondition(self.rank == 3)
    try createDirectoryIfMissing(at: directory)
    
    let fileExtension: String
    switch format {
    case .jpeg: fileExtension = "jpg"
    case .png: fileExtension = "png"
    }
    
    let outputURL = URL(fileURLWithPath: "\(directory)/\(name).\(fileExtension)")
    let image = Image(self)
    image.save(to: outputURL, format: format)
  }
  
  /// Treats the tensor as an image and overlays it on a white background. This must be a rank-3
  /// tensor, with channels in the 0.0 - 255.0 range. Also, it assumes that the image uses
  /// premultiplied alpha.
  func overlaidOnWhite() -> Tensor {
    precondition(self.rank == 3)
    precondition(self.shape[2] == 4)
    let alphaChannel = self.slice(
        lowerBounds: [0, 0, 3], sizes: [self.shape[0], self.shape[1], 1])
    let colorComponents = self.slice(
        lowerBounds: [0, 0, 0], sizes: [self.shape[0], self.shape[1], 3])
    return (255.0 - alphaChannel) + colorComponents
  }
  
  /// Treats the tensor as a grayscale image and normalizes it to a 0.0 - 255.0 range. This must be
  /// a rank-1 or rank-2 tensor. The minimum and maximum channel values are remapped to 0.0 and
  /// 255.0, respectively, and all values rescaled to that range.
  func normalizedToGrayscale() -> Tensor {
    let lowerBound = self.min(alongAxes: [0, 1])
    let upperBound = self.max(alongAxes: [0, 1])
    return (self - lowerBound) * (255.0 / (upperBound - lowerBound))
  }
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
