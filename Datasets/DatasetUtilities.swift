// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import ModelSupport

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public enum DatasetUtilities {
    public static let currentWorkingDirectoryURL = URL(
        fileURLWithPath: FileManager.default.currentDirectoryPath)

    @discardableResult
    public static func downloadResource(
        filename: String,
        fileExtension: String,
        remoteRoot: URL,
        localStorageDirectory: URL = currentWorkingDirectoryURL
    ) -> URL {
        printError("Loading resource: \(filename)")

        let resource = ResourceDefinition(
            filename: filename,
            fileExtension: fileExtension,
            remoteRoot: remoteRoot,
            localStorageDirectory: localStorageDirectory)

        let localURL = resource.localURL

        if !FileManager.default.fileExists(atPath: localURL.path) {
            printError(
                "File does not exist locally at expected path: \(localURL.path) and must be fetched"
            )
            fetchFromRemoteAndSave(resource)
        }

        return localURL
    }

    @discardableResult
    public static func fetchResource(
        filename: String,
        fileExtension: String,
        remoteRoot: URL,
        localStorageDirectory: URL = currentWorkingDirectoryURL
    ) -> Data {
        let localURL = DatasetUtilities.downloadResource(
            filename: filename, fileExtension: fileExtension, remoteRoot: remoteRoot,
            localStorageDirectory: localStorageDirectory)

        do {
            let data = try Data(contentsOf: localURL)
            return data
        } catch {
            fatalError("Failed to contents of resource: \(localURL)")
        }
    }

    struct ResourceDefinition {
        let filename: String
        let fileExtension: String
        let remoteRoot: URL
        let localStorageDirectory: URL

        var localURL: URL {
            localStorageDirectory.appendingPathComponent(filename)
        }

        var remoteURL: URL {
            remoteRoot.appendingPathComponent(filename).appendingPathExtension(fileExtension)
        }

        var archiveURL: URL {
            localURL.appendingPathExtension(fileExtension)
        }
    }

    static func fetchFromRemoteAndSave(_ resource: ResourceDefinition) {
        let remoteLocation = resource.remoteURL
        let archiveLocation = resource.localStorageDirectory

        do {
            printError("Fetching URL: \(remoteLocation)...")
            try download(from: remoteLocation, to: archiveLocation)
        } catch {
            fatalError("Failed to fetch and save resource with error: \(error)")
        }
        printError("Archive saved to: \(archiveLocation.path)")

        extractArchive(for: resource)
    }

    static func extractArchive(for resource: ResourceDefinition) {
        printError("Extracting archive...")

        let archivePath = resource.archiveURL.path

        #if os(macOS)
            let binaryLocation = "/usr/bin/"
        #else
            let binaryLocation = "/bin/"
        #endif

        let toolName: String
        let arguments: [String]
        switch resource.fileExtension {
        case "gz":
            toolName = "gunzip"
            arguments = [archivePath]
        case "tar.gz", "tgz":
            toolName = "tar"
            arguments = ["xzf", archivePath, "-C", resource.localStorageDirectory.path]
        case "zip":
            toolName = "unzip"
            arguments = [archivePath, "-d", resource.localStorageDirectory.path]
        default:
            printError("Unable to find archiver for extension \(resource.fileExtension).")
            exit(-1)
        }
        let toolLocation = "\(binaryLocation)\(toolName)"

        let task = Process()
        task.executableURL = URL(fileURLWithPath: toolLocation)
        task.arguments = arguments
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            printError("Failed to extract \(archivePath) with error: \(error)")
            exit(-1)
        }

        if FileManager.default.fileExists(atPath: archivePath) {
            do {
                try FileManager.default.removeItem(atPath: archivePath)
            } catch {
                printError("Could not remove archive, error: \(error)")
                exit(-1)
            }
        }
    }
}
