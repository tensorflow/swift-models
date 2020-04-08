import Datasets
import Foundation
import SwiftProtobuf

let fileNames = [
    //"captions_train2017.json",
    //"captions_val2017.json",
    //"ILSVRC2014_train.json",
    //"ILSVRC2014_val.json",
    //"image_info_test2017.json",
    //"image_info_test-dev2017.json",
    //"image_info_unlabeled2017.json",
    //"instances_train2017.json",
    //"instances_val2017.json",
    //"pascal_test2007.json",
    //"pascal_train2007.json",
    //"pascal_train2012.json",
    //"pascal_val2007.json",
    //"pascal_val2012.json",
    //"person_keypoints_train2017.json",
    //"person_keypoints_val2017.json",
    // "stuff_train2017.json",
    "stuff_val2017.json"
]

for fileName in fileNames {
    let fileURL = URL(fileURLWithPath: "pycocotools/annotations/\(fileName)")
    let coco = try! COCO(fromFile: fileURL)
    print("Loaded COCO dataset from \(fileName)")
    print("Categories: \(coco.cats.count)")
    print("Images: \(coco.imgs.count)")
    print("Annotations: \(coco.anns.count)")
}
