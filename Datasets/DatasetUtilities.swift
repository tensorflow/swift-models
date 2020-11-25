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
        
    public static let defaultDirectory = try! FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("swift-models").appendingPathComponent("datasets")

    @discardableResult
    public static func downloadResource(
        filename: String,
        fileExtension: String,
        remoteRoot: URL,
        localStorageDirectory: URL = currentWorkingDirectoryURL,
        extract: Bool = true
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
            fetchFromRemoteAndSave(resource, extract: extract)
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

    static func fetchFromRemoteAndSave(_ resource: ResourceDefinition, extract: Bool) {
        let remoteLocation = resource.remoteURL
        let archiveLocation = resource.localStorageDirectory

        do {
            printError("Fetching URL: \(remoteLocation)...")
            try download(from: remoteLocation, to: archiveLocation)
        } catch {
            fatalError("Failed to fetch and save resource with error: \(error)")
        }
        printError("Archive saved to: \(archiveLocation.path)")

        if extract {
            extractArchive(
                at: resource.archiveURL, to: resource.localStorageDirectory,
                fileExtension: resource.fileExtension, deleteArchiveWhenDone: true)
        }
    }
}
