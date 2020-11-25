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

/// A write-only representation of a GIF that animates between sequential frames, constructed from
/// a stack of Tensors.
struct AnimatedImage {
  let frames: [Tensor<Float>]

  /// Creates an animated image representation from a stack of Tensors.
  ///
  /// - Parameters:
  ///   - frames: A sequential array of Tensors that represent frames of animation within the image.
  ///     The individual Tensors must be rank 3, ordered as [width, height, channels] and the
  ///     channels are assumed to be in the range 0.0 - 255.0.
  public init(_ frames: [Tensor<Float>]) {
    guard frames.count > 0 else { fatalError("No frames were provided to animated image.") }
    guard frames[0].rank == 3 else {
      fatalError("Rank-3 tensors are needed, and rank \(frames[0].rank) frames were provided.")
    }
    guard frames.allSatisfy({ $0.shape == frames[0].shape }) else {
      fatalError("All animation frames need to have identical shape.")
    }
    self.frames = frames
  }

  func quantize(_ frames: [Tensor<Float>]) -> (quantizedFrames: [[UInt8]], palette: [UInt8]) {
    return (frames[0].shape[2] == 1) ? quantizeGrey(frames) : quantizeColor(frames)
  }

  func quantizeGrey(_ frames: [Tensor<Float>]) -> (quantizedFrames: [[UInt8]], palette: [UInt8]) {
    let palette: [UInt8] = GIF.greyscale64Palette

    var quantizedFrames: [[UInt8]] = []
    for frame in frames {
      let scalars = frame.scalars
      var quantizedFrame: [UInt8] = []
      for scalar in scalars {
        let gray = UInt8(max(min(scalar, 255.0), 0.0))
        let grayQuantized = gray / 4
        let lookup = grayQuantized
        quantizedFrame.append(lookup)
      }
      quantizedFrames.append(quantizedFrame)
    }
    return (quantizedFrames: quantizedFrames, palette: palette)
  }
    
  func quantizeColor(_ frames: [Tensor<Float>]) -> (quantizedFrames: [[UInt8]], palette: [UInt8]) {
    // TODO: Adapt the following to the colors in the input image.
    let palette: [UInt8] = GIF.defaultPalette

    let colorComponents = frames.first!.shape[2]
    var quantizedFrames: [[UInt8]] = []
    for frame in frames {
      let scalars = frame.scalars
      var quantizedFrame: [UInt8] = []
      for index in stride(from: 0, to: scalars.count, by: colorComponents) {
        let red: UInt8
        let green: UInt8
        let blue: UInt8
        if colorComponents == 4 {
          // I'm placing values with an alpha channel on a white background.
          let alpha = scalars[index + 3]
          red = UInt8(max(min(round(scalars[index] + (255.0 - alpha)), 255.0), 0.0))
          green = UInt8(max(min(round(scalars[index + 1] + (255.0 - alpha)), 255.0), 0.0))
          blue = UInt8(max(min(round(scalars[index + 2] + (255.0 - alpha)), 255.0), 0.0))
        } else {
          red = UInt8(max(min(scalars[index], 255.0), 0.0))
          green = UInt8(max(min(scalars[index + 1], 255.0), 0.0))
          blue = UInt8(max(min(scalars[index + 2], 255.0), 0.0))
        }
        let redQuantized = red / 51
        let greenQuantized = green / 51
        let blueQuantized = blue / 51
        let lookup = redQuantized * 6 * 6 + greenQuantized * 6 + blueQuantized + 16
        quantizedFrame.append(lookup)
      }
      quantizedFrames.append(quantizedFrame)
    }
    return (quantizedFrames: quantizedFrames, palette: palette)
  }

  /// Saves the animated GIF from the internal stack of Tensors. A default color quantization scheme
  /// is applied to map input colors to an output color table.
  ///
  /// - Parameters:
  ///   - url: The target location of the animated GIF file.
  ///   - delay: The delay (in hundredths of a second) to insert between each frame.
  ///   - loop: Whether the animation should loop indefinitely.
  public func save(to url: URL, delay: Int, loop: Bool = true) throws {
    let width = frames[0].shape[1]
    let height = frames[0].shape[0]

    let (quantizedFrames, palette) = quantize(frames)
    var gif = GIF(width: width, height: height, palette: palette, loop: loop)

    for frame in quantizedFrames {
      gif.append(frame: frame, delay: delay)
    }

    gif.close()
    try gif.bytes.write(to: url)
  }
}

public extension Array where Element == Tensor<Float> {
  /// Saves an array of Tensors as an animated GIF. The individual Tensors must be rank 3, ordered
  /// as [width, height, channels] and the channels are assumed to be in the range 0.0 - 255.0.
  ///
  /// - Parameters:
  ///   - directory: The target directory to host the animated GIF file. If it does not exist, it
  ///     will be created.
  ///   - name: The name of the resulting image file, without extension.
  ///   - delay: The delay (in hundredths of a second) to insert between each frame.
  ///   - loop: Whether the animation should loop indefinitely.
  func saveAnimatedImage(directory: String, name: String, delay: Int, loop: Bool = true) throws {
    try createDirectoryIfMissing(at: directory)

    let image = AnimatedImage(self)

    let outputURL = URL(fileURLWithPath: "\(directory)/\(name).gif")
    try image.save(to: outputURL, delay: delay, loop: loop)
  }
}
