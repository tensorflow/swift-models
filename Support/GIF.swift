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

// The following GIF writer has been ported to Swift from https://github.com/lecram/gifenc

import Foundation

class Node {
  let key: Int
  var children: [UInt8: Node] = [:]
  
  init(key: Int) {
    self.key = key
  }
  
  static func trie(degree: Int, keys: inout Int) -> Node {
    let root = Node(key: 0)
    for key in 0..<degree {
      root.children[UInt8(key)] = Node(key: key)
    }
    keys = degree + 2
    return root
  }
}

struct GIF {
  let width: Int
  let height: Int
  let depth: Int
  let palette: [UInt8]?
  var bytes: Data
  let loop: Bool
  var partial: UInt = 0
  var offset: Int = 0
  var buffer: [UInt8]
  
  static let vga: [UInt8] = [
    0x00, 0x00, 0x00,
    0xAA, 0x00, 0x00,
    0x00, 0xAA, 0x00,
    0xAA, 0x55, 0x00,
    0x00, 0x00, 0xAA,
    0xAA, 0x00, 0xAA,
    0x00, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55,
    0xFF, 0x55, 0x55,
    0x55, 0xFF, 0x55,
    0xFF, 0xFF, 0x55,
    0x55, 0x55, 0xFF,
    0xFF, 0x55, 0xFF,
    0x55, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF,
  ]

  static var defaultPalette: [UInt8] {
    var palette: [UInt8] = vga
    for red in stride(from: 0, through: 255, by: 51) {
      for green in stride(from: 0, through: 255, by: 51) {
        for blue in stride(from: 0, through: 255, by: 51) {
          palette.append(contentsOf: [UInt8(red), UInt8(green), UInt8(blue)])
        }
      }
    }
    for grey in 0..<24 {
      let greyValue = UInt8(grey * 0xFF / 25)
      palette.append(contentsOf: [greyValue, greyValue, greyValue])
    }
    
    return palette
  }
  
  init(width: Int, height: Int, depth: Int = 8, palette: [UInt8]? = nil, loop: Bool = true) {
    self.width = width
    self.height = height
    self.depth = depth
    self.palette = palette
    self.bytes = Data()
    self.loop = loop
    self.buffer = [UInt8](repeating: 0, count: 255)
    
    addHeader()
  }
  
  mutating func addHeader() {
    bytes.append(string: "GIF89a")
    bytes.append(littleEndian: width)
    bytes.append(littleEndian: height)
    bytes.append(contentsOf: [0xF0 | (UInt8(depth) - 1), 0x00, 0x00])
    
    if let providedPalette = palette {
      bytes.append(contentsOf: providedPalette)
    } else if depth <= 4 {
      bytes.append(contentsOf: GIF.vga.prefix(3 << depth))
    } else {
      bytes.append(contentsOf: GIF.defaultPalette)
    }
    if loop {
      putLoop()
    }
  }
  
  mutating func append(frame: [UInt8], delay: Int) {
    append(delay: delay)
    // TODO: Optimize bounding box to only provided updated part.
    
    bytes.append(string: ",")
    bytes.append(littleEndian: 0) // X
    bytes.append(littleEndian: 0) // Y
    bytes.append(littleEndian: width)
    bytes.append(littleEndian: height)
    bytes.append(contentsOf: [0x00, UInt8(depth)])
    
    var keySize = depth + 1
    let degree = 1 << depth
    putKey(degree, size: keySize)
    var keys = 0
    var root = Node.trie(degree: degree, keys: &keys)
    var node = root
    for i in 0..<height {
      for j in 0..<width {
        let pixel = frame[i * width + j] & UInt8(degree - 1)
        if let child = node.children[pixel] {
          node = child
        } else {
          putKey(node.key, size: keySize)
          if keys < 0x1000 {
            if keys == (1 << keySize) {
              keySize += 1
            }
            node.children[pixel] = Node(key: keys)
            keys += 1
          } else {
            putKey(degree, size: keySize)
            root = Node.trie(degree: degree, keys: &keys)
            node = root
            keySize = depth + 1
          }
          node = root.children[pixel]!
        }
      }
    }
    
    putKey(node.key, size: keySize)
    putKey(degree + 1, size: keySize)
    endKey()
  }
  
  mutating func append(delay: Int) {
    bytes.append(string: "!")
    bytes.append(contentsOf: [0xF9, 0x04, 0x04])
    bytes.append(littleEndian: delay)
    bytes.append(string: "\0\0")
  }
  
  mutating func putKey(_ key: Int, size: Int) {
    var byteOffset = offset / 8
    let bitOffset = offset % 8
    partial |= UInt(key) << bitOffset
    var bitsToWrite = bitOffset + size
    while (bitsToWrite >= 8) {
      buffer[byteOffset] = UInt8(partial & 0xFF)
      byteOffset += 1
      if (byteOffset == 0xFF) {
        bytes.append(0xFF) // "\xFF"
        bytes.append(contentsOf: buffer)
        byteOffset = 0
      }
      partial >>= 8
      bitsToWrite -= 8
    }
    offset = (offset + size) % (0xFF * 8)
  }

  mutating func endKey() {
    var byteOffset = offset / 8
    if ((offset % 8) > 1) {
      buffer[byteOffset] = UInt8(partial & 0xFF)
      byteOffset += 1
    }
    bytes.append(UInt8(byteOffset))
    bytes.append(contentsOf: buffer.prefix(byteOffset))
    bytes.append(string: "\0")
    offset = 0
    partial = 0
  }

  mutating func putLoop() {
    bytes.append(string: "!")
    bytes.append(contentsOf: [0xFF, 0x0B])
    bytes.append(string: "NETSCAPE2.0")
    bytes.append(contentsOf: [0x03, 0x01])
    // TODO: Specify number of loops, rather than looping forever.
    bytes.append(littleEndian: 0)
    bytes.append(string: "\0")
  }
  
  mutating func close() {
    bytes.append(string: ";")
  }
}

extension Data {
  mutating func append(littleEndian value: Int) {
    self.append(UInt8(value & 0xff))
    self.append(UInt8(value >> 8))
  }
  
  mutating func append(string: String) {
    self.append(string.data(using: String.Encoding.utf8, allowLossyConversion: true)!)
  }
}
