import Datasets
import Foundation

let fileURL = URL(fileURLWithPath: "pycocotools/annotations/pascal_val2012.json")
let metadata = COCO(fromFile: fileURL)
