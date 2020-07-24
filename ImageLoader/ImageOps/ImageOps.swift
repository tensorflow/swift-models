
import Foundation
import TurboJPEG

// Loads image
public func tjJPEGLoadCompressedImage( filename: UnsafePointer<Int8>?, width: inout Int32, align: inout Int32,  height: inout Int32, pixelFormat: inout Int32, inSubsamp: Int32, flags: Int) -> UnsafeMutablePointer<UInt8>? {
  
    pixelFormat = -1
 
    /* Read the JPEG file into memory. */
    var jpegFile = fopen(filename, "rb")
    fseek(jpegFile, 0, SEEK_END)
    let size = ftell(jpegFile)
    fseek(jpegFile, 0, SEEK_SET)
    let jpegSize = CUnsignedLongLong(size)
    var jpegBuf = (tjAlloc(Int32(jpegSize)))
    fread(jpegBuf, Int(jpegSize), 1, jpegFile)
    fclose(jpegFile)
    jpegFile = nil
    
    var tjInstance = tjInitDecompress()
    /* Initializes `width` and `height` variables */
    tjDecompressHeader(tjInstance, jpegBuf, UInt(jpegSize), &width, &height)
    
    let imgBuf = tjAlloc(3 * width * height)
    
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
    tjDecompress2(tjInstance, jpegBuf, UInt(jpegSize), imgBuf, width, 0, height, 0, 0)
    
    /* Free/Destroy instances and buffers */
    tjFree(jpegBuf)
    jpegBuf = nil
    tjDestroy(tjInstance)
    tjInstance = nil
    
    return imgBuf
   
}
 //Saves image
public func tjJPEGSaveImage(filename: UnsafePointer<Int8>?, buffer: UnsafePointer<UInt8>?, width: Int32, pitch: Int32, height: Int32, pixelFormat: Int32, outSubsamp: Int32, flags: Int32) -> Int32 {
    
    /* Create new file */
    var jpegFile = fopen(filename, "wb")
    var jpegBuf: UnsafeMutablePointer<UInt8>?
    
    var retVal: Int32 = -1
    let outQual: Int32 = 95
    var jpegSize: CUnsignedLong = 0
    
    var tjInstance = tjInitCompress();
    /* Compress the Image Data from `buffer` into `jpegBuf`
        - Compresses `buffer` which has image data
        - `width` = width of Image
        - pitch = 0
        - `height` = height of image
        - `pixelFormat` = format to which image is being compressed.
        - Image gets compressed into `jpegBuf` buffer
        - initializes `jpegSize` as size of image in bytes
        - `outSubsamp` = the level of chrominance subsampling to be used when generating the JPEG image
        - `outQual` = the image quality of the generated JPEG image (1 = worst, 100 = best), 95 taken as default
        - `flags` = the bitwise OR of one or more of the flags
    */
    tjCompress2(tjInstance, buffer, width, 0, height, pixelFormat, &jpegBuf, &jpegSize, outSubsamp, outQual, flags)
    
    if (fwrite(jpegBuf, Int(jpegSize), 1, jpegFile) == 1){
        retVal = 0
    }
    
    /* Free/Destroy instances and buffers */
    tjDestroy(tjInstance)
    tjInstance = nil
    fclose(jpegFile)
    jpegFile = nil
    tjFree(jpegBuf)
    jpegBuf = nil
    
    return retVal;
    
}