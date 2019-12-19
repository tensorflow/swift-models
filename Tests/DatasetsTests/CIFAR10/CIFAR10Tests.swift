import TensorFlow
import XCTest
import Datasets

final class CIFAR10Tests: XCTestCase {
    override class func setUp() {
        super.setUp()

        // prepare data
        let from = FileManager.default.currentDirectoryPath + "/Tests/DatasetsTests/CIFAR10/cifar-10-binary.tar.gz"
        let toDir = FileManager.default.temporaryDirectory.path + "/CIFAR10"
        do {
            if !FileManager.default.fileExists(atPath: toDir) {
                try FileManager.default.createDirectory(atPath: toDir, withIntermediateDirectories: false)
            }

            try FileManager.default.copyItem(atPath: from, toPath: toDir + "/cifar-10-binary.tar.gz")
        } catch {
            print("Could not copy over CIFAR archive, error: \(error)")
            exit(-1)
        }
        print("CIFAR archive copied to \(toDir)")
    }

    func testCreateCIFAR10() {
        let dataset = CIFAR10()

        var totalCount = 0
        for example in dataset.trainingDataset {
            XCTAssertTrue((0..<10).contains(example.label.scalar!))
            XCTAssertEqual(example.data.shape, [32, 32, 3])
            totalCount += 1
        }
        XCTAssertEqual(totalCount, 50000)
    }

    override class func tearDown() {
        super.tearDown()

        // clean up archive
        let archivePath = FileManager.default.temporaryDirectory.path + "/CIFAR10/cifar-10-binary.tar.gz"
        do {
            if FileManager.default.fileExists(atPath: archivePath) {
                try FileManager.default.removeItem(atPath: archivePath)
            }
        } catch {
            print("Could not remove archive, error: \(error)")
            exit(-1)
        }
        print("CIFAR archive \(archivePath) cleaned")
    }
}

extension CIFAR10Tests {
    static var allTests = [
        ("testCreateCIFAR10", testCreateCIFAR10),
    ]
}

