import Foundation

/// Coco dataset API that loads annotation file and prepares 
/// data structures for data set access.
public struct COCO {
    public typealias Metadata = [String: Any]
    public typealias Info = [String: Any]
    public typealias Annotation = [String: Any]
    public typealias AnnotationId = Int
    public typealias Image = [String: Any]
    public typealias ImageId = Int
    public typealias Category = [String: Any]
    public typealias CategoryId = Int

    public var metadata: Metadata
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
        self.metadata = parsed as! Metadata
        self.info = [:]
        self.anns = [:]
        self.cats = [:]
        self.imgs = [:]
        self.imgToAnns = [:]
        self.catToImgs = [:]
        self.createIndex()
    }

    mutating func createIndex() {
        if let info = metadata["info"] {
            self.info = info as! Info
        }
        if let annotations = metadata["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let ann_id = ann["id"] as! AnnotationId
                let image_id = ann["image_id"] as! ImageId
                self.imgToAnns[image_id, default: []].append(ann)
                self.anns[ann_id] = ann
            }
        }
        if let images = metadata["images"] {
            let imgs = images as! [Image]
            for img in imgs {
                let img_id = img["id"] as! ImageId
                self.imgs[img_id] = img
            }
        }
        if let categories = metadata["categories"] {
            let cats = categories as! [Category]
            for cat in cats {
                let cat_id = cat["id"] as! CategoryId
                self.cats[cat_id] = cat
            }
        }
        if let annotations = metadata["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let cat_id = ann["category_id"] as! CategoryId
                let image_id = ann["image_id"] as! ImageId
                self.catToImgs[cat_id, default: []].append(image_id)
            }
        }
    }

    /// Get annotation ids that satisfy given filter conditions.
    func getAnnotationIds() {}

    /// Get category ids that satisfy given filter conditions.
    func getCategoryIds() {}

    /// Get image ids that satisfy given filter conditions.
    func getImgageIds() {}

    /// Load annotations with specified ids.
    func loadAnnotations() {}

    /// Load categories with specified ids.
    func loadCategories() {}

    /// Load images with specified ids.
    func loadImages() {}

    /// Convert segmentation in an anotation to binary mask.
    func annotationToMask() {}

    /// Display the specified annotations.
    func showAnnotations() {}

    /// Load algorithm results and create API for accessing them.
    func loadResults() {}

    /// Download images from mscoco.org server.
    func downloadImages() {}
}
