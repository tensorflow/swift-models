import Foundation

/// Coco dataset API that loads annotation file and prepares 
/// data structures for data set access.
public struct COCO {
    public typealias Dataset = [String: Any]
    public typealias Info = [String: Any]
    public typealias Annotation = [String: Any]
    public typealias AnnotationId = Int
    public typealias Image = [String: Any]
    public typealias ImageId = Int
    public typealias Category = [String: Any]
    public typealias CategoryId = Int

    public var dataset: Dataset
    public var info: Info
    public var anns: [AnnotationId: Annotation]
    public var cats: [CategoryId: Category]
    public var imgs: [ImageId: Image]
    public var imgToAnns: [ImageId: [Annotation]]
    public var catToImgs: [CategoryId: [ImageId]]

    public init(fromFile fileURL: URL) throws {
        let contents = try String(contentsOfFile: fileURL.path)
        let data = contents.data(using: .utf8)!
        let parsed = try JSONSerialization.jsonObject(with: data)
        self.dataset = parsed as! Dataset
        self.info = [:]
        self.anns = [:]
        self.cats = [:]
        self.imgs = [:]
        self.imgToAnns = [:]
        self.catToImgs = [:]
        self.createIndex()
    }

    mutating func createIndex() {
        if let info = dataset["info"] {
            self.info = info as! Info
        }
        if let annotations = dataset["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let ann_id = ann["id"] as! AnnotationId
                let image_id = ann["image_id"] as! ImageId
                self.imgToAnns[image_id, default: []].append(ann)
                self.anns[ann_id] = ann
            }
        }
        if let images = dataset["images"] {
            let imgs = images as! [Image]
            for img in imgs {
                let img_id = img["id"] as! ImageId
                self.imgs[img_id] = img
            }
        }
        if let categories = dataset["categories"] {
            let cats = categories as! [Category]
            for cat in cats {
                let cat_id = cat["id"] as! CategoryId
                self.cats[cat_id] = cat
            }
        }
        if dataset["annotations"] != nil && dataset["categories"] != nil {
            let anns = dataset["annotations"] as! [Annotation]
            for ann in anns {
                let cat_id = ann["category_id"] as! CategoryId
                let image_id = ann["image_id"] as! ImageId
                self.catToImgs[cat_id, default: []].append(image_id)
            }
        }
    }

    /// Get annotation ids that satisfy given filter conditions.
    public func getAnnotationIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = [],
        areaRange: [[Double]] = [],
        isCrowd: Int? = nil
    ) -> [AnnotationId] {
        let filterByImageId = imageIds.count != 0
        let filterByCategoryId = imageIds.count != 0
        let filterByAreaRange = areaRange.count != 0
        let filterByIsCrowd = isCrowd != nil

        var anns: [Annotation] = []
        if filterByImageId {
            for imageId in imageIds {
                if let imageAnns = self.imgToAnns[imageId] {
                    for imageAnn in imageAnns {
                        anns.append(imageAnn)
                    }
                }
            }
        } else {
            anns = self.dataset["annotations"] as! [Annotation]
        }

        var annIds: [AnnotationId] = []
        for ann in anns {
            if filterByCategoryId {
                let categoryId = ann["category_id"] as! CategoryId
                if !categoryIds.contains(categoryId) {
                    continue
                }
            }
            if filterByAreaRange {
                let area = ann["area"] as! [Double]
                if !areaLessThan(areaRange[0], area) || !areaLessThan(area, areaRange[1]) {
                    continue
                }
            }
            if filterByIsCrowd {
                let annIsCrowd = ann["iscrowd"] as! Int
                if annIsCrowd != isCrowd! {
                    continue
                }
            }
            let id = ann["id"] as! AnnotationId
            annIds.append(id)
        }
        return annIds
    }

    /// A helper function that decides if one area is less than the other.
    func areaLessThan(_ left: [Double], _ right: [Double]) -> Bool {
        // TODO: 
        return false
    }

    /// Get category ids that satisfy given filter conditions.
    public func getCategoryIds(
        categoryNames: [String] = [],
        supercategoryNames: [String] = [],
        categoryIds: [CategoryId] = []
    ) -> [CategoryId] {
        let filterByName = categoryNames.count != 0
        let filterBySupercategory = supercategoryNames.count != 0
        let filterById = categoryIds.count != 0
        var categoryIds: [CategoryId] = []
        let cats = self.dataset["categories"] as! [Category]
        for cat in cats {
            let name = cat["name"] as! String
            let supercategory = cat["supercategory"] as! String
            let id = cat["id"] as! CategoryId
            if filterByName && !categoryNames.contains(name) {
                continue
            }
            if filterBySupercategory && !supercategoryNames.contains(supercategory) {
                continue
            }
            if filterById && !categoryIds.contains(id) {
                continue
            }
            categoryIds.append(id)
        }
        return categoryIds
    }

    /// Get image ids that satisfy given filter conditions.
    public func getImageIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = []
    ) -> [ImageId] {
        if imageIds.count == 0 && categoryIds.count == 0 {
            return Array(self.imgs.keys)
        } else {
            var ids = Set(imageIds)
            for (i, catId) in categoryIds.enumerated() {
                if i == 0 && ids.count == 0 {
                    ids = Set(self.catToImgs[catId]!)
                } else {
                    ids = ids.intersection(Set(self.catToImgs[catId]!))
                }
            }
            return Array(ids)
        }
    }

    /// Load annotations with specified ids.
    public func loadAnnotations(ids: [AnnotationId] = []) -> [Annotation] {
        var anns: [Annotation] = []
        for id in ids {
            anns.append(self.anns[id]!)
        }
        return anns
    }

    /// Load categories with specified ids.
    public func loadCategories(ids: [CategoryId] = []) -> [Category] {
        var cats: [Category] = []
        for id in ids {
            cats.append(self.cats[id]!)
        }
        return cats
    }

    /// Load images with specified ids.
    public func loadImages(ids: [ImageId] = []) -> [Image] {
        var imgs: [Image] = []
        for id in ids {
            imgs.append(self.imgs[id]!)
        }
        return imgs
    }

    /// Convert segmentation in an annotation to RLE.
    public func annotationToRLE(_ ann: Annotation) -> RLE {
        let imgId = ann["image_id"] as! ImageId
        let img = self.imgs[imgId]!
        let h = img["height"] as! Int
        let w = img["width"] as! Int
        let segm = ann["segmentation"]
        if let polygon = segm as? [Any] {
            let rles = Mask.fromObject(polygon, width: w, height: h)
            return Mask.merge(rles)
        } else if let segmDict = segm as? [String: Any] {
            if segmDict["counts"] is [Any] {
                return Mask.fromObject(segmDict, width: w, height: h)[0]
            } else if let countsStr = segmDict["counts"] as? String {
                return RLE(fromString: countsStr, width: w, height: h)
            } else {
                fatalError("unrecognized annotation: \(ann)")
            }
        } else {
            fatalError("unrecognized annotation: \(ann)")
        }
    }

    /// Convert segmentation in an anotation to binary mask.
    public func annotationToMask(_ ann: Annotation) -> Mask {
        fatalError("todo")
    }

    /// Download images from mscoco.org server.
    public func downloadImages() {
        fatalError("todo")
    }
}

public struct Mask {
    static func merge(_ rles: [RLE], intersect: Bool = false) -> RLE {
        return RLE(merging: rles, intersect: intersect)
    }

    static func fromBoundingBoxes(_ bboxes: [[Double]], width w: Int, height h: Int) -> [RLE] {
        var rles: [RLE] = []
        for bbox in bboxes {
            let rle = RLE(fromBoundingBox: bbox, width: w, height: h)
            rles.append(rle)
        }
        return rles
    }

    static func fromPolygons(_ polys: [[Double]], width w: Int, height h: Int) -> [RLE] {
        var rles: [RLE] = []
        for poly in polys {
            let rle = RLE(fromPolygon: poly, width: w, height: h)
            rles.append(rle)
        }
        return rles
    }

    static func fromUncompressedRLEs(_ arr: [[String: Any]], width w: Int, height h: Int) -> [RLE] {
        var rles: [RLE] = []
        for elem in arr {
            let counts = elem["counts"] as! [Int]
            let m = counts.count
            var cnts = [UInt32](repeating: 0, count: m)
            for i in 0..<m {
                cnts[i] = UInt32(counts[i])
            }
            let size = elem["size"] as! [Int]
            let h = size[0]
            let w = size[1]
            rles.append(RLE(width: w, height: h, m: cnts.count, counts: cnts))
        }
        return rles
    }

    static func fromObject(_ obj: Any, width w: Int, height h: Int) -> [RLE] {
        // encode rle from a list of json deserialized objects
        if let arr = obj as? [[Double]] {
            assert(arr.count > 0)
            if arr[0].count == 4 {
                return fromBoundingBoxes(arr, width: w, height: h)
            } else {
                assert(arr[0].count > 4)
                return fromPolygons(arr, width: w, height: h)
            }
        } else if let arr = obj as? [[String: Any]] {
            assert(arr.count > 0)
            assert(arr[0]["size"] != nil)
            assert(arr[0]["counts"] != nil)
            return fromUncompressedRLEs(arr, width: w, height: h)
            // encode rle from a single json deserialized object
        } else if let arr = obj as? [Double] {
            if arr.count == 4 {
                return fromBoundingBoxes([arr], width: w, height: h)
            } else {
                assert(arr.count > 4)
                return fromPolygons([arr], width: w, height: h)
            }
        } else if let dict = obj as? [String: Any] {
            assert(dict["size"] != nil)
            assert(dict["counts"] != nil)
            return fromUncompressedRLEs([dict], width: w, height: h)
        } else {
            fatalError("input type is not supported")
        }
    }
}

public struct RLE {
    var width: Int = 0
    var height: Int = 0
    var m: Int = 0
    var counts: [UInt32] = []

    init(width w: Int, height h: Int, m: Int, counts: [UInt32]) {
        self.width = w
        self.height = h
        self.m = m
        self.counts = counts
    }

    init(fromString str: String, width w: Int, height h: Int) {
        let data = str.data(using: .utf8)!
        let bytes = [UInt8](data)
        self.init(fromBytes: bytes, width: w, height: h)
    }

    init(fromBytes bytes: [UInt8], width w: Int, height h: Int) {
        var m: Int = 0
        var p: Int = 0
        var cnts = [UInt32](repeating: 0, count: bytes.count)
        while p < bytes.count {
            var x: Int = 0
            var k: Int = 0
            var more: Int = 1
            while more != 0 {
                let c = Int8(bitPattern: bytes[p]) - 48
                x |= (Int(c) & 0x1f) << 5 * k
                more = Int(c) & 0x20
                p += 1
                k += 1
                if more == 0 && (c & 0x10) != 0 {
                    x |= -1 << 5 * k
                }
            }
            if m > 2 {
                x += Int(cnts[m - 2])
            }
            cnts[m] = UInt32(truncatingIfNeeded: x)
            m += 1
        }
        self.init(width: w, height: h, m: m, counts: cnts)
    }

    init(fromBoundingBox bb: [Double], width w: Int, height h: Int) {
        let xs = bb[0]
        let ys = bb[1]
        let xe = bb[2]
        let ye = bb[3]
        let xy: [Double] = [xs, ys, xs, ye, xe, ye, xe, ys]
        self.init(fromPolygon: xy, width: w, height: h)
    }

    init(fromPolygon xy: [Double], width w: Int, height h: Int) {
        // upsample and get discrete points densely along the entire boundary
        var k: Int = xy.count / 2
        var j: Int = 0
        var m: Int = 0
        let scale: Double = 5
        var x = [Int](repeating: 0, count: k + 1)
        var y = [Int](repeating: 0, count: k + 1)
        for j in 0..<k { x[j] = Int(scale * xy[j * 2 + 0] + 0.5) }
        x[k] = x[0]
        for j in 0..<k { y[j] = Int(scale * xy[j * 2 + 1] + 0.5) }
        y[k] = y[0]
        for j in 0..<k { m += max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1 }
        var u = [Int](repeating: 0, count: m)
        var v = [Int](repeating: 0, count: m)
        m = 0
        for j in 0..<k {
            var xs: Int = x[j]
            var xe: Int = x[j + 1]
            var ys: Int = y[j]
            var ye: Int = y[j + 1]
            let dx: Int = abs(xe - xs)
            let dy: Int = abs(ys - ye)
            var t: Int
            let flip: Bool = (dx >= dy && xs > xe) || (dx < dy && ys > ye)
            if flip {
                t = xs
                xs = xe
                xe = t
                t = ys
                ys = ye
                ye = t
            }
            let s: Double = dx >= dy ? Double(ye - ys) / Double(dx) : Double(xe - xs) / Double(dy)
            if dx >= dy {
                for d in 0...dx {
                    t = flip ? dx - d : d
                    u[m] = t + xs
                    let vm = Double(ys) + s * Double(t) + 0.5
                    v[m] = vm.isNaN ? 0 : Int(vm)
                    m += 1
                }
            } else {
                for d in 0...dy {
                    t = flip ? dy - d : d
                    v[m] = t + ys
                    let um = Double(xs) + s * Double(t) + 0.5
                    u[m] = um.isNaN ? 0 : Int(um)
                    m += 1
                }
            }
        }
        // get points along y-boundary and downsample
        k = m
        m = 0
        var xd: Double
        var yd: Double
        x = [Int](repeating: 0, count: k)
        y = [Int](repeating: 0, count: k)
        for j in 1..<k {
            if u[j] != u[j - 1] {
                xd = Double(u[j] < u[j - 1] ? u[j] : u[j] - 1)
                xd = (xd + 0.5) / scale - 0.5
                if floor(xd) != xd || xd < 0 || xd > Double(w - 1) { continue }
                yd = Double(v[j] < v[j - 1] ? v[j] : v[j - 1])
                yd = (yd + 0.5) / scale - 0.5
                if yd < 0 { yd = 0 } else if yd > Double(h) { yd = Double(h) }
                yd = ceil(yd)
                x[m] = Int(xd)
                y[m] = Int(yd)
                m += 1
            }
        }
        // compute rle encoding given y-boundary points
        k = m
        var a = [UInt32](repeating: 0, count: k + 1)
        for j in 0..<k { a[j] = UInt32(x[j] * Int(h) + y[j]) }
        a[k] = UInt32(h * w)
        k += 1
        a.sort()
        var p: UInt32 = 0
        for j in 0..<k {
            let t: UInt32 = a[j]
            a[j] -= p
            p = t
        }
        var b = [UInt32](repeating: 0, count: k)
        j = 0
        m = 0
        b[m] = a[j]
        m += 1
        j += 1
        while j < k {
            if a[j] > 0 {
                b[m] = a[j]
                m += 1
                j += 1
            } else {
                j += 1
            }
            if j < k {
                b[m - 1] += a[j]
                j += 1
            }
        }
        self.init(width: w, height: h, m: m, counts: b)
    }

    init(merging rles: [RLE], intersect: Bool) {
        var c: UInt32
        var ca: UInt32
        var cb: UInt32
        var cc: UInt32
        var ct: UInt32
        var v: Bool
        var va: Bool
        var vb: Bool
        var vp: Bool
        var a: Int
        var b: Int
        var w: Int = rles[0].width
        var h: Int = rles[0].height
        var m: Int = rles[0].m
        var A: RLE
        var B: RLE
        let n = rles.count
        if n == 0 { self.init(width: 0, height: 0, m: 0, counts: []) }
        if n == 1 { self.init(width: w, height: h, m: m, counts: rles[0].counts) }
        var cnts = [UInt32](repeating: 0, count: h * w + 1)
        for a in 0..<m {
            cnts[a] = rles[0].counts[a]
        }
        for i in 1..<n {
            B = rles[i]
            if B.height != h || B.width != w {
                h = 0
                w = 0
                m = 0
                break
            }
            A = RLE(width: w, height: h, m: m, counts: cnts)
            ca = A.counts[0]
            cb = B.counts[0]
            v = false
            va = false
            vb = false
            m = 0
            a = 1
            b = 1
            cc = 0
            ct = 1
            while ct > 0 {
                c = min(ca, cb)
                cc += c
                ct = 0
                ca -= c
                if ca == 0 && a < A.m {
                    ca = A.counts[a]
                    a += 1
                    va = !va
                }
                ct += ca
                cb -= c
                if cb == 0 && b < B.m {
                    cb = B.counts[b]
                    b += 1
                    vb = !vb
                }
                ct += cb
                vp = v
                if intersect {
                    v = va && vb
                } else {
                    v = va || vb
                }
                if v != vp || ct == 0 {
                    cnts[m] = cc
                    m += 1
                    cc = 0
                }
            }
        }
        self.init(width: w, height: h, m: m, counts: cnts)
    }
}
