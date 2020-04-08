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
        if let annotations = dataset["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let cat_id = ann["category_id"] as! CategoryId
                let image_id = ann["image_id"] as! ImageId
                self.catToImgs[cat_id, default: []].append(image_id)
            }
        }
    }

    /// Get annotation ids that satisfy given filter conditions.
    func getAnnotationIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = [],
        areaRange: [[Float]] = [],
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
                let area = ann["area"] as! [Float]
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
    private func areaLessThan(_ left: [Float], _ right: [Float]) -> Bool {
        // TODO: 
        return false
    }

    /// Get category ids that satisfy given filter conditions.
    func getCategoryIds(
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
    func getImageIds(
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
    func loadAnnotations(ids: [AnnotationId] = []) -> [Annotation] {
        var anns: [Annotation] = []
        for id in ids {
            anns.append(self.anns[id]!)
        }
        return anns
    }

    /// Load categories with specified ids.
    func loadCategories(ids: [CategoryId] = []) -> [Category] {
        var cats: [Category] = []
        for id in ids {
            cats.append(self.cats[id]!)
        }
        return cats
    }

    /// Load images with specified ids.
    func loadImages(ids: [ImageId] = []) -> [Image] {
        var imgs: [Image] = []
        for id in ids {
            imgs.append(self.imgs[id]!)
        }
        return imgs
    }

    /// Convert segmentation in an annotation to RLE.
    func annotationToRLE() {}

    /// Convert segmentation in an anotation to binary mask.
    func annotationToMask() {}

    /// Download images from mscoco.org server.
    func downloadImages() {}
}
