import Foundation
import ImageOps

// this function loads and saves image
func imageLoadSave(imagePath: String,  savedImagePath: String) -> Int32 {

    let url = (imagePath as NSString).utf8String
    let filenamepointer = UnsafeMutablePointer<Int8>(mutating: url)!
    
    var width: Int32 = 0
    var align: Int32 = 0
    var height: Int32 = 0
    var pixelFormat: Int32 = 0
    let inSubsamp: Int32 = 0
    
    let imgBuffer = tjJPEGLoadCompressedImage(filename: filenamepointer, width: &width, align: &align, height: &height, pixelFormat: &pixelFormat, inSubsamp: inSubsamp, flags: 0)
    
    let url2 = (savedImagePath as NSString).utf8String
    let filenamepointer2 = UnsafeMutablePointer<Int8>(mutating: url2)!
    
    let retVal = tjJPEGSaveImage(filename: filenamepointer2, buffer: imgBuffer, width: width, pitch: 0, height: height, pixelFormat: 0, outSubsamp: inSubsamp, flags: 0)
    return retVal
}

