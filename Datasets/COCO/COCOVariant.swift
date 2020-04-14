import Foundation
import ModelSupport

public struct COCOVariant {
    static let annotationsURL =
        URL(string: "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")!

    static func downloadIfNotPresent(from location: URL, to directory: URL) {
        let downloadPath = directory.path
        let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return }

        let _ = DatasetUtilities.downloadResource(
            filename: "annotations_trainval2017", fileExtension: "zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
    }

    static func loadVariant(from url: URL, to directory: URL, nested file: String) -> COCO {
        downloadIfNotPresent(from: url, to: directory)
        let jsonFile = URL(string: directory.appendingPathComponent(file).path)!
        let coco = try! COCO(fromFile: jsonFile)
        return coco
    }

    public static func defaultDirectory() -> URL {
        return DatasetUtilities.defaultDirectory
            .appendingPathComponent("COCO-annotations_trainval2017", isDirectory: true)
    }

    public static func loadCaptionsTrain2017(to directory: URL = defaultDirectory()) -> COCO {
        return loadVariant(
            from: annotationsURL, to: directory, nested: "annotations/captions_train2017.json")
    }

    public static func loadCaptionsVal2017(to directory: URL = defaultDirectory()) -> COCO {
        return loadVariant(
            from: annotationsURL, to: directory, nested: "annotations/captions_val2017.json")
    }

    public static func loadInstancesTrain2017(to directory: URL = defaultDirectory()) -> COCO {
        return loadVariant(
            from: annotationsURL, to: directory, nested: "annotations/instances_train2017.json")
    }

    public static func loadInstancesVal2017(to directory: URL = defaultDirectory()) -> COCO {
        return loadVariant(
            from: annotationsURL, to: directory, nested: "annotations/instances_val2017.json")
    }

    public static func loadPersonKeypointsTrain2017(to directory: URL = defaultDirectory()) -> COCO
    {
        return loadVariant(
            from: annotationsURL, to: directory,
            nested: "annotations/person_keypoints_train2017.json")
    }

    public static func loadPersonKeypointsVal2017(to directory: URL = defaultDirectory()) -> COCO {
        return loadVariant(
            from: annotationsURL, to: directory, nested: "annotations/person_keypoints_val2017.json"
        )
    }
}
