typealias uint = UInt32
typealias siz = UInt64
typealias byte = UInt8
typealias BB = [Double]

struct COCORLE {
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

struct COCORLEs {
    var R: [COCORLE]
    var n: siz
}

struct COCOMasks {
    var mask: [byte]
    var h: siz
    var w: siz
    var n: siz
}
