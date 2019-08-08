import Foundation
import TensorFlow
import FastStyleTransfer

extension TransformerNet: ImportableLayer {}

/// Updates `obj` with values from command line arguments according to `params` map.
func parseArguments<T>(into obj: inout T, with params: [String: WritableKeyPath<T, String?>]) {
    for arg in CommandLine.arguments.dropFirst() {
        if !arg.starts(with: "--") { continue }
        let parts = arg.split(separator: "=", maxSplits: 2)
        let name = String(parts[0][parts[0].index(parts[0].startIndex, offsetBy: 2)...])
        if let path = params[name], parts.count == 2 {
            obj[keyPath: path] = String(parts[1])
        }
    }
}

enum FileError: Error {
    case fileNotFound
}

/// Updates `model` with parameters from numpy archive file in `path`.
func importWeights(_ model: inout TransformerNet, from path: String) throws {
    guard FileManager.default.fileExists(atPath: path) else {
        throw FileError.fileNotFound
    }
    // Names don't match exactly, and axes in filters need to be reversed.
    let map = [
        "conv1.conv2d.filter": ("conv1.conv2d.weight", [3, 2, 1, 0]),
        "conv2.conv2d.filter": ("conv2.conv2d.weight", [3, 2, 1, 0]),
        "conv3.conv2d.filter": ("conv3.conv2d.weight", [3, 2, 1, 0]),
        "deconv1.conv2d.filter": ("deconv1.conv2d.weight", [3, 2, 1, 0]),
        "deconv2.conv2d.filter": ("deconv2.conv2d.weight", [3, 2, 1, 0]),
        "deconv3.conv2d.filter": ("deconv3.conv2d.weight", [3, 2, 1, 0]),
        "res1.conv1.conv2d.filter": ("res1.conv1.conv2d.weight", [3, 2, 1, 0]),
        "res1.conv2.conv2d.filter": ("res1.conv2.conv2d.weight", [3, 2, 1, 0]),
        "res1.in1.scale": ("res1.in1.weight", nil),
        "res1.in1.offset": ("res1.in1.bias", nil),
        "res1.in2.scale": ("res1.in2.weight", nil),
        "res1.in2.offset": ("res1.in2.bias", nil),
        "res2.conv1.conv2d.filter": ("res2.conv1.conv2d.weight", [3, 2, 1, 0]),
        "res2.conv2.conv2d.filter": ("res2.conv2.conv2d.weight", [3, 2, 1, 0]),
        "res2.in1.scale": ("res2.in1.weight", nil),
        "res2.in1.offset": ("res2.in1.bias", nil),
        "res2.in2.scale": ("res2.in2.weight", nil),
        "res2.in2.offset": ("res2.in2.bias", nil),
        "res3.conv1.conv2d.filter": ("res3.conv1.conv2d.weight", [3, 2, 1, 0]),
        "res3.conv2.conv2d.filter": ("res3.conv2.conv2d.weight", [3, 2, 1, 0]),
        "res3.in1.scale": ("res3.in1.weight", nil),
        "res3.in1.offset": ("res3.in1.bias", nil),
        "res3.in2.scale": ("res3.in2.weight", nil),
        "res3.in2.offset": ("res3.in2.bias", nil),
        "res4.conv1.conv2d.filter": ("res4.conv1.conv2d.weight", [3, 2, 1, 0]),
        "res4.conv2.conv2d.filter": ("res4.conv2.conv2d.weight", [3, 2, 1, 0]),
        "res4.in1.scale": ("res4.in1.weight", nil),
        "res4.in1.offset": ("res4.in1.bias", nil),
        "res4.in2.scale": ("res4.in2.weight", nil),
        "res4.in2.offset": ("res4.in2.bias", nil),
        "res5.conv1.conv2d.filter": ("res5.conv1.conv2d.weight", [3, 2, 1, 0]),
        "res5.conv2.conv2d.filter": ("res5.conv2.conv2d.weight", [3, 2, 1, 0]),
        "res5.in1.scale": ("res5.in1.weight", nil),
        "res5.in1.offset": ("res5.in1.bias", nil),
        "res5.in2.scale": ("res5.in2.weight", nil),
        "res5.in2.offset": ("res5.in2.bias", nil),
        "in1.scale": ("in1.weight", nil),
        "in1.offset": ("in1.bias", nil),
        "in2.scale": ("in2.weight", nil),
        "in2.offset": ("in2.bias", nil),
        "in3.scale": ("in3.weight", nil),
        "in3.offset": ("in3.bias", nil),
        "in4.scale": ("in4.weight", nil),
        "in4.offset": ("in4.bias", nil),
        "in5.scale": ("in5.weight", nil),
        "in5.offset": ("in5.bias", nil),
    ]
    model.unsafeImport(fromNumpyArchive: path, map: map)
}

/// Loads from `file` and returns JPEG image as HxWxC tensor of floats in (0..1) range.
func loadJpegAsTensor(from file: String) throws -> Tensor<Float> {
    guard FileManager.default.fileExists(atPath: file) else {
        throw FileError.fileNotFound
    }
    let imgData = Raw.readFile(filename: StringTensor(file))
    return Tensor<Float>(Raw.decodeJpeg(contents: imgData, channels: 3, dctMethod: "")) / 255
}

/// Clips & converts HxWxC `tensor` of floats to byte range and saves as JPEG.
func saveTensorAsJpeg(_ tensor: Tensor<Float>, to file: String) {
    let clipped = Raw.clipByValue(t: tensor, clipValueMin: Tensor(0), clipValueMax: Tensor(255))
    let jpg = Raw.encodeJpeg(image: Tensor<UInt8>(clipped), format: .rgb, xmpMetadata: "")
    Raw.writeFile(filename: StringTensor(file), contents: jpg)
}
