import Datasets
import Foundation
import SwiftProtobuf

let coco = COCOVariant.loadInstancesVal2017()
print("Loaded COCO dataset")
print("Info:")
for (k, v) in coco.info {
    print("  \(k): \(v)")
}
print("Categories: \(coco.categories.count)")
print("Images: \(coco.images.count)")
print("Annotations: \(coco.annotations.count)")
for (_, ann) in coco.annotations {
    if ann["caption"] == nil {
        let _ = coco.annotationToRLE(ann)
        let _ = coco.annotationToMask(ann)
    }
}
