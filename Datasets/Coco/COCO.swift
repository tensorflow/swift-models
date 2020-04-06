import Foundation

/// Coco dataset API that loads annotation file and prepares 
/// data structures for data set access.
public struct COCO {
    let metadata: COCOMetadata
    let anns: [String: Any]
    let cats: [String: Any]
    let imgs: [String: Any]
    let imgToAnns: [String: Any]
    let catToImgs: [String: Any]

    public init(fromFile fileURL: URL) {
        self.metadata = try! COCOMetadata(fromFile: fileURL)
        self.anns = [:]
        self.cats = [:]
        self.imgs = [:]
        self.imgToAnns = [:]
        self.catToImgs = [:]
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
