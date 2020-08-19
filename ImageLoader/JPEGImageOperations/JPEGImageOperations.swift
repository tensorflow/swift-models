import Foundation
import TurboJPEG
 
#if os(iOS) || os(macOS) || os(tvOS) || os(watchOS)
import Darwin
#elseif os(Android) || os(Linux)
import Glibc
#elseif os(Windows)
import ucrt
#else
#error ("C library not known for the target OS")
#endif
 
public enum pixelFormats: Int32 {
    case RGB888 = 0   // TJPF_RGB
    case BGR888 = 1   // TJPF_BGR
    case RGBA8888 = 2 // TJPF_RGBA
    case BGRA8888 = 3 // TJPF_BGRA
    case ARGB8888 = 4 // TJPF_ARGB
    case ABGR8888 = 5 // TJPF_ABGR
    case RGBA8880 = 6 // TJPF_RGBX
    case BGRA8880 = 7 // TJPF_BGRX
    case ARGB0888 = 8 // TJPF_XRGB
    case ABGR0888 = 9 // TJPF_XBGR
    case YUV400 = 10  // TJPF_GREY
    
    var channelCount: Int32 {
        switch self {
        case .RGB888:
            return 3
        case .BGR888:
            return 3
        case .RGBA8888:
            return 4
        case .BGRA8888:
            return 4
        case .ARGB8888:
            return 4
        case .ABGR8888:
            return 4
        case .RGBA8880:
            return 3
        case .BGRA8880:
            return 3
        case .ARGB0888:
            return 3
        case .ABGR0888:
            return 3
        case .YUV400:
            return 1
        }
    }
}
 
public struct ImageData {
    var height: Int32
    var width: Int32
    var data: UnsafeMutablePointer<UInt8>
    var formatProperties: pixelFormats
 
    init(height: Int32, width: Int32, data: UnsafeMutablePointer<UInt8>, imageFormat: pixelFormats) {
        self.height = height
        self.width = width
        self.data = data
        self.formatProperties = imageFormat
    }
}
 
func LoadJPEG(atPath path: String, imageFormat: pixelFormats) -> ImageData? {
    
    var width: Int32 = 0
    var height: Int32 = 0
    
    let data: Data
    
    if FileManager.default.fileExists(atPath: path) {
        data = FileManager.default.contents(atPath: path)!
    } else {
        // File not present
        fatalError("File does not exist at: \(path).")
    }
    
    let jpegSize: UInt = UInt(data.count)
    let baseAddress: UnsafeRawPointer = data.withUnsafeBytes { return $0.baseAddress! }
    let finPointer = UnsafeMutablePointer<UInt8>(mutating: baseAddress.assumingMemoryBound(to: UInt8.self))
    
    var decompressor = tjInitDecompress()
    /* Initializes `width` and `height` variables */
    tjDecompressHeader(decompressor, finPointer , jpegSize, &width, &height)
    
    let imgBuf = tjAlloc(imageFormat.channelCount * width * height)
    
    /* Decompresses the JPEG Image from `jpegBuf` into `imgBuf` buffer
     - Decompresses `jpegBuf` which has image data
     - uses `jpegSize` as size of image in bytes
     - Image gets decompressed into `imgBuf` buffer
     - `width` = width of Image
     - pitch = 0
     - `height` = height of image
     - pixelFormat = 0 which denotes TJPF_RGB Pixel Format to which image is being decompressed.
     - flags = 0
     */
    tjDecompress2(decompressor, finPointer, UInt(jpegSize), imgBuf, width, 0, height, imageFormat.rawValue, 0)
    
    /* Free/Destroy instances and buffers */
    tjDestroy(decompressor)
    decompressor = nil
    
    let res = ImageData.init(height: height, width: width, data: imgBuf!, imageFormat: imageFormat)
    return res
    
}
 
func SaveJPEG(atPath path : String, image: ImageData) -> Int32 {
    
    do {
        try FileManager.default.removeItem(atPath: path)
    } catch {
        // File not present
    }
    
    var jpegBuf: UnsafeMutablePointer<UInt8>?
    
    var retVal: Int32 = -1
    let outQual: Int32 = 95
    var jpegSize: UInt = 0
    
    var compressor = tjInitCompress()
    /* Compress the Image Data from `buffer` into `jpegBuf`
     - Compresses image.data
     - `width` = width of Image
     - pitch = 0
     - `height` = height of image
     - `pixelFormat` = 0
     - Image gets compressed into `jpegBuf` buffer
     - initializes `jpegSize` as size of image in bytes
     - `outSubsamp` = 0
     - `outQual` = the image quality of the generated JPEG image (1 = worst, 100 = best), 95 taken as default
     - `flags` = 0
     */
    tjCompress2(compressor, image.data, image.width, 0, image.height, image.formatProperties.rawValue, &jpegBuf, &jpegSize, 0, outQual, 0)
 
    let bufferPointer  = UnsafeMutableBufferPointer.init(start: jpegBuf, count: Int(jpegSize))
    let jpegData = Data.init(buffer: bufferPointer)
    
    do {
        FileManager.default.createFile(atPath: path, contents: jpegData)
        retVal = 0
    } catch {
        // File not created
        fatalError("Error during image saving: \(error)")
    }
    
    /* Free/Destroy instances and buffers */
    tjDestroy(compressor)
    compressor = nil
    tjFree(jpegBuf)
    jpegBuf = nil
    
    return retVal
}