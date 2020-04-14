import Datasets
import Foundation
import SwiftProtobuf

let fileNames = [
    "captions_train2017.json",
    "captions_val2017.json",
    "ILSVRC2014_train.json",
    "ILSVRC2014_val.json",
    "image_info_test2017.json",
    "image_info_test-dev2017.json",
    "image_info_unlabeled2017.json",
    "instances_train2017.json",
    "instances_val2017.json",
    "pascal_test2007.json",
    "pascal_train2007.json",
    "pascal_train2012.json",
    "pascal_val2007.json",
    "pascal_val2012.json",
    "person_keypoints_train2017.json",
    "person_keypoints_val2017.json",
    "stuff_train2017.json",
    "stuff_val2017.json",
]

for fileName in fileNames {
    print("---")
    print(fileName)
    let fileURL = URL(fileURLWithPath: "pycocotools/annotations/\(fileName)")
    let coco = try! COCO(fromFile: fileURL)
    print("Loaded COCO dataset from \(fileName)")
    print("Info:")
    for (k, v) in coco.info {
        print("  \(k): \(v)")
    }
    print("Categories: \(coco.categories.count)")
    print("Images: \(coco.images.count)")
    print("Annotations: \(coco.annotations.count)")
    for (_, ann) in coco.annotations {
        if ann["caption"] == nil {
            let rle = coco.annotationToRLE(ann)
        }
    }
}
