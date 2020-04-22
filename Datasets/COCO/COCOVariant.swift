import Foundation
import ModelSupport

public struct COCOVariant {
    static let trainAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-train2017.zip"
        )!
    static let valAnnotationsURL =
        URL(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-val2017.zip"
        )!
    static let trainImagesURL =
        URL(string: "http://images.cocodataset.org/zips/train2017.zip")!
    static let valImagesURL =
        URL(string: "http://images.cocodataset.org/zips/val2017.zip")!

    static func downloadIfNotPresent(
        from location: URL,
        to directory: URL,
        filename: String
    ) {
        let downloadPath = directory.appendingPathComponent(filename).path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        let _ = DatasetUtilities.downloadResource(
            filename: filename, fileExtension: "zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
    }

    static func loadJSON(_ directory: URL, annotations: String, images: String) -> COCO {
        let jsonPath = directory.appendingPathComponent(annotations).path
        let jsonURL = URL(string: jsonPath)!
        let imagesDirectory = directory.appendingPathComponent(images)
        let coco = try! COCO(fromFile: jsonURL, imagesDirectory: imagesDirectory)
        return coco
    }

    public static func defaultDirectory() -> URL {
        return DatasetUtilities.defaultDirectory
            .appendingPathComponent("COCO", isDirectory: true)
    }

    public static func loadTrain(to directory: URL = defaultDirectory()) -> COCO {
        downloadIfNotPresent(
            from: trainAnnotationsURL, to: directory,
            filename: "annotations-train2017")
        downloadIfNotPresent(
            from: trainImagesURL, to: directory,
            filename: "train2017")
        return loadJSON(
            directory,
            annotations: "annotations-train2017/instances_train2017.json",
            images: "train2017")
    }

    public static func loadVal(to directory: URL = defaultDirectory()) -> COCO {
        downloadIfNotPresent(
            from: valAnnotationsURL, to: directory,
            filename: "annotations-val2017")
        downloadIfNotPresent(
            from: valImagesURL, to: directory,
            filename: "val2017")
        return loadJSON(
            directory,
            annotations: "annotations-val2017/instances_val2017.json",
            images: "val2017")
    }
}
