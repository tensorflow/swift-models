typealias uint = UInt32
typealias siz = UInt64
typealias byte = UInt8
typealias BB = [Double]

struct RLE {
    var h: siz
    var w: siz
    var m: siz
    var cnts: [uint]

    init(h: siz, w: siz, m: siz, cnts: [uint]) {
        self.h = h
        self.w = w
        self.m = m
        self.cnts = cnts
    }
}

struct RLEs {
    var R: [RLE]
    var n: siz
}

struct Masks {
    var mask: [byte]
    var h: siz
    var w: siz
    var n: siz
}
