import Foundation

/// Convert an integer represented in its `endian` who takes `count`  
/// bytes in length to an array of bytes.
func toByteArr<T: BinaryInteger>(endian: T, count: Int) -> [UInt8] {
  var _endian = endian
  let bytePtr = withUnsafePointer(to: &_endian) {
    $0.withMemoryRebound(to: UInt8.self, capacity: count) {
      UnsafeBufferPointer(start: $0, count: count)
    }
  }
  return [UInt8](bytePtr)
}

extension UInt64 {
  var littleEndianBuffer: [UInt8] {
    toByteArr(endian: self.littleEndian, count: 8)
  }

  var bigEndianBuffer: [UInt8] {
    toByteArr(endian: self.bigEndian, count: 8)
  }
}

extension UInt32 {
  var littleEndianBuffer: [UInt8] {
    toByteArr(endian: self.littleEndian, count: 4)
  }

  var bigEndianBuffer: [UInt8] {
    toByteArr(endian: self.bigEndian, count: 4)
  }
}

/// This is taken from https://github.com/tensorflow/swift-models/blob/master/Checkpoints/CheckpointReader.swift#L299
/// Need to take to a common place later.
extension Data {
  static var crc32CLookupTable: [UInt32] = {
    (0...255).map { index -> UInt32 in
      var lookupValue = UInt32(index)
      for _ in 0..<8 {
        lookupValue = (lookupValue % 2 == 0)
          ? (lookupValue >> 1) : (0x82F6_3B78 ^ (lookupValue >> 1))
      }
      return lookupValue
    }
  }()

  func crc32C() -> UInt32 {
    var crc32: UInt32 = 0xFFFF_FFFF

    self.withUnsafeBytes { buffer in
      let totalBytes = self.count
      var index = 0
      while index < totalBytes {
        let byte = buffer[index]
        let lookupIndex = Int((crc32 ^ (UInt32(byte) & 0xFF)) & 0xFF)
        crc32 = (crc32 >> 8) ^ Data.crc32CLookupTable[lookupIndex]
        index = index &+ 1
      }
    }

    return crc32 ^ 0xFFFF_FFFF
  }

  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/hash/crc32c.h
  func maskedCRC32C() -> UInt32 {
    let crc32 = self.crc32C()
    let maskDelta: UInt32 = 0xA282_EAD8
    return ((crc32 &>> 15) | (crc32 &<< 17)) &+ maskDelta
  }
}
