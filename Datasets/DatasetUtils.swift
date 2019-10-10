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
import FoundationNetworking

public struct DatasetUtils {
    
    public static let curentWorkingDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    public static func fetchResource(filename: String,
                                     remoteRoot: URL,
                                     localStorageDirectory: URL = curentWorkingDirectoryURL) -> Data {
        
        print("Loading resource: \(filename)")
        
        let resource = ResourceDefinition(filename: filename,
                                          remoteRoot: remoteRoot,
                                          localStorageDirectory: localStorageDirectory)
        
        let localURL = resource.localURL
        
        // Fetch from remote if the file is not available locally
        if !FileManager.default.fileExists(atPath: localURL.path) {
            print("File does not exist locally at expected path: \(localURL.path) and must be fetched")
            fetchFromRemoteAndSave(resource: resource)
        }
        
        do {
            print("Loading local data at: \(localURL.path)")
            let data = try Data(contentsOf: localURL)
            print("Succesfully loaded resource: \(filename)")
            return data
        } catch {
            fatalError("Failed to contents of resource: \(localURL)")
        }
        
    }
    
    struct ResourceDefinition {
        
        let filename: String
        let remoteRoot: URL
        let localStorageDirectory: URL
        
        var localURL: URL {
            return localStorageDirectory.appendingPathComponent(filename)
        }
        
        var remoteURL: URL {
            remoteRoot.appendingPathComponent(filename).appendingPathExtension("gz")
        }
        
        var archiveURL: URL {
            localURL.appendingPathExtension("gz")
        }

    }

        
    static func fetchFromRemoteAndSave(resource: ResourceDefinition) {
        let remoteLocation = resource.remoteURL
        let archiveLocation = resource.archiveURL
        
        do {
            print("Fetching URL: \(remoteLocation)...")
            let archiveData = try Data(contentsOf: remoteLocation)
            print("Writing fetched archive to: \(archiveLocation.path)")
            try archiveData.write(to: archiveLocation)
        } catch {
            fatalError("Failed to fetch and save resource with error: \(error)")
        }
        print("Archive saved to: \(archiveLocation.path)")
        
        extractArchive(for: resource)
    }

    
    static func extractArchive(for resource: ResourceDefinition) {
        
        print("Extracting archive...")
        
        let archivePath = resource.archiveURL.path
        
        #if os(macOS)
        let gunzipLocation = "/usr/bin/gunzip"
        #else
        let gunzipLocation = "/bin/gunzip"
        #endif
        
        let task = Process()
        task.executableURL = URL(fileURLWithPath: gunzipLocation)
        task.arguments = [archivePath]
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            fatalError("Failed to extract \(archivePath) with error: \(error)")
        }
        
    }
    
}
