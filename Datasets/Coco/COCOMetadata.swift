import Foundation

public struct COCOInfo: Decodable {
    var description: String
    var url: String
    var version: String
    var year: Int
    var contributor: String
    var date_created: String
    public init(
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
public struct COCOLicense: Decodable {
    var url: String
    var id: Int
    var name: String
    public init(url: String, id: Int, name: String) {
        self.url = url
        self.id = id
        self.name = name
    }
}
public struct COCOImage: Decodable {
    var license: Int
    var file_name: String
    var coco_url: String
    var height: Int
    var width: Int
    var date_captured: String
    var flickr_url: String
    var id: Int
    public init(
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
public struct COCOSize: Decodable {
    var _1: Int
    var _2: Int
    public init(_1: Int, _2: Int) {
        self._1 = _1
        self._2 = _2
    }
}
public struct COCOBoundingBox: Decodable {
    var _1: Float
    var _2: Float
    var _3: Float
    var _4: Float
    public init(_1: Float, _2: Float, _3: Float, _4: Float) {
        self._1 = _1
        self._2 = _2
        self._3 = _3
        self._4 = _4
    }
}
public enum COCOSegmentation {
    case polygon(values: [[Float]])
    case uncompressedRLE(counts: [Int], size: COCOSize)
    case compressedRLE(counts: String, size: COCOSize)
}
extension COCOSegmentation: Decodable {
    // TODO: implement decoder
    public init(from decoder: Decoder) throws {
        self = .polygon(values: [])
    }
}
public struct COCOAnnotation: Decodable {
    var image_id: Int
    var id: Int
    var caption: String
    var segmentation: COCOSegmentation
    var area: Float
    var iscrowd: Int
    var bbox: COCOBoundingBox
    var category_id: Int
    var ignore: Int
    var num_keypoints: Int
    var keypoints: [Int]
    public init(
        image_id: Int, id: Int, caption: String, segmentation: COCOSegmentation, area: Float,
        iscrowd: Int, bbox: COCOBoundingBox, category_id: Int, ignore: Int, num_keypoints: Int,
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
public struct COCOCategory: Decodable {
    var supercategory: String
    var id: Int
    var name: String
    var keypoints: [String]
    var skeleton: [COCOSize]
    public init(
        supercategory: String, id: Int, name: String, keypoints: [String], skeleton: [COCOSize]
    ) {
        self.supercategory = supercategory
        self.id = id
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton
    }
}
public struct COCOMetadata: Decodable {
    var info: COCOInfo
    var licenses: [COCOLicense]
    var images: [COCOImage]
    var annotations: [COCOAnnotation]
    var type: String
    var categories: [COCOCategory]
    public init(
        info: COCOInfo, licenses: [COCOLicense], images: [COCOImage], annotations: [COCOAnnotation],
        type: String, categories: [COCOCategory]
    ) {
        self.info = info
        self.licenses = licenses
        self.images = images
        self.annotations = annotations
        self.type = type
        self.categories = categories
    }
    public init(fromFile fileURL: URL) throws {
        let contents = try String(contentsOfFile: fileURL.path)
        let data = contents.data(using: .utf8)!
        self = try JSONDecoder().decode(COCOMetadata.self, from: data)
    }
}
