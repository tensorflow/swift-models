import TensorFlow
import Files
import Foundation

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

    public var tensor: Tensorf {
        switch self.imageData {
        case let .float(data): return data
        case let .uint8(data): return Tensor<Float>(data)
        }
    }

    public init(tensor: Tensor<UInt8>) {
        self.imageData = .uint8(data: tensor)
    }

    public init(tensor: Tensorf) {
        self.imageData = .float(data: tensor)
    }

    public init(jpeg url: URL, byteOrdering: ByteOrdering = .rgb) {
        let loadedFile = _Raw.readFile(filename: StringTensor(url.absoluteString))
        let loadedJpeg = _Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")
        if byteOrdering == .bgr {
            self.imageData = .uint8(
                data: _Raw.reverse(loadedJpeg, dims: Tensor<Bool>([false, false, false, true])))
        } else {
            self.imageData = .uint8(data: loadedJpeg)
        }
    }

    public func save(to url: URL, format: _Raw.Format = .grayscale, quality: Int64 = 95) {
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
                    images: Tensor<UInt8>([data]),
                    size: Tensor<Int32>([Int32(size.0), Int32(size.1)])))
        case let .float(data):
            return Image(
                tensor: _Raw.resizeBilinear(
                    images: Tensor<Float>([data]),
                    size: Tensor<Int32>([Int32(size.0), Int32(size.1)])))
        }

    }
}
