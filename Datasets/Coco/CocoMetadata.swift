import Foundation

struct CocoInfo: Decodable {
    var description: String
    var url: String
    var version: String
    var year: Int
    var contributor: String
    var date_created: String
    init(
        description: String, url: String, version: String, year: Int, contributor: String,
        date_created: String
    ) {
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created
    }
}
struct CocoLicense: Decodable {
    var url: String
    var id: Int
    var name: String
    init(url: String, id: Int, name: String) {
        self.url = url
        self.id = id
        self.name = name
    }
}
struct CocoImage: Decodable {
    var license: Int
    var file_name: String
    var coco_url: String
    var height: Int
    var width: Int
    var date_captured: String
    var flickr_url: String
    var id: Int
    init(
        license: Int, file_name: String, coco_url: String, height: Int, width: Int,
        date_captured: String, flickr_url: String, id: Int
    ) {
        self.license = license
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id
    }
}
struct CocoSize: Decodable {
    var _1: Int
    var _2: Int
    init(_1: Int, _2: Int) {
        self._1 = _1
        self._2 = _2
    }
}
struct CocoBoundingBox: Decodable {
    var _1: Float
    var _2: Float
    var _3: Float
    var _4: Float
    init(_1: Float, _2: Float, _3: Float, _4: Float) {
        self._1 = _1
        self._2 = _2
        self._3 = _3
        self._4 = _4
    }
}
enum CocoSegmentation {
    case polygon(values: [[Float]])
    case uncompressedRLE(counts: [Int], size: CocoSize)
    case compressedRLE(counts: String, size: CocoSize)
}
extension CocoSegmentation: Decodable {
    // TODO: implement decoder
    init(from decoder: Decoder) throws {
        self = .polygon(values: [])
    }
}
struct CocoAnnotation: Decodable {
    var image_id: Int
    var id: Int
    var caption: String
    var segmentation: CocoSegmentation
    var area: Float
    var iscrowd: Int
    var bbox: CocoBoundingBox
    var category_id: Int
    var ignore: Int
    var num_keypoints: Int
    var keypoints: [Int]
    init(
        image_id: Int, id: Int, caption: String, segmentation: CocoSegmentation, area: Float,
        iscrowd: Int, bbox: CocoBoundingBox, category_id: Int, ignore: Int, num_keypoints: Int,
        keypoints: [Int]
    ) {
        self.image_id = image_id
        self.id = id
        self.caption = caption
        self.segmentation = segmentation
        self.area = area
        self.iscrowd = iscrowd
        self.bbox = bbox
        self.category_id = category_id
        self.ignore = ignore
        self.num_keypoints = num_keypoints
        self.keypoints = keypoints
    }
}
struct CocoCategory: Decodable {
    var supercategory: String
    var id: Int
    var name: String
    var keypoints: [String]
    var skeleton: [CocoSize]
    init(supercategory: String, id: Int, name: String, keypoints: [String], skeleton: [CocoSize]) {
        self.supercategory = supercategory
        self.id = id
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton
    }
}
struct CocoMetadata: Decodable {
    var info: CocoInfo
    var licenses: [CocoLicense]
    var images: [CocoImage]
    var annotations: [CocoAnnotation]
    var type: String
    var categories: [CocoCategory]
    init(
        info: CocoInfo, licenses: [CocoLicense], images: [CocoImage], annotations: [CocoAnnotation],
        type: String, categories: [CocoCategory]
    ) {
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.type = type
        self.categories = categories
    }
    init(fromFile fileURL: URL) throws {
        let contents = try String(contentsOfFile: fileURL.path)
        let data = contents.data(using: .utf8)!
        self = try JSONDecoder().decode(CocoAnnotationsFile.self, from: data)
    }
}
