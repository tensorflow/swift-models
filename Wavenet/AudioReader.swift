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
import TensorFlow
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

public struct AudioReader {
    let rootDir: URL
    let audioSampleRate: Int
    let sampleSize: Int

    public init(rootDir: URL, audioSampleRate: Int,
                sampleSize: Int) {
        self.rootDir = rootDir
        self.audioSampleRate = audioSampleRate
        self.sampleSize = sampleSize
    }

    public static func loadSamplesFromFile(
        from url: URL, receptiveField: Int, audioSampleRate: Int,
        sampleSize: Int
    ) -> [Tensor<Float>] {
        let np = Python.import("numpy")
        let pydub = Python.import("pydub")
        let audio = pydub.AudioSegment.from_file(url.path, format: "wav")
            .set_frame_rate(audioSampleRate)
        let audioBuffer = audio.get_array_of_samples()

        // Convert buffer to float32 using NumPy
        let audio_as_np_int16 = np.frombuffer(audioBuffer, dtype: np.int16)
        let audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
        var features = Tensor<Float>(numpy: audio_as_np_float32)!
        features = features.padded(
            forSizes: [(receptiveField, 0)])
        return sliceOrPadSample(
            features, sampleSize: sampleSize, receptiveFieldWidth: receptiveField
        )
    }

    // TODO: This function is temporary and will be superseded
    // by the PaddingFIFOQueue
    private static func sliceOrPadSample(_ sample: Tensor<Float>,
                                         sampleSize: Int, receptiveFieldWidth: Int) -> [Tensor<Float>] {
        let sampleWidth = sample.shape[0]
        // If the sample is smaller than sampleSize
        // pad the sample to sampleSize
        if sampleWidth < sampleSize {
            let padSize = sampleSize - sampleWidth + receptiveFieldWidth
            return [sample.padded(
                forSizes: [(0, padSize)])]
        }
        // If the sample is larger than sampleSize
        // slice it into pieces of size = sampleSize
        // and pad the last piece
        else {
            let (quotient, remainder) = sampleWidth.quotientAndRemainder(dividingBy: sampleSize)
            let padSize = sampleSize - remainder + receptiveFieldWidth
            let result = sample
                .padded(
                    forSizes: [(0, padSize)])
            return result
                .split(count: quotient + 1, alongAxis: 0)
        }
    }

    public func loadDataFileNames() throws -> [URL] {
        guard
            let directoryEnumerator = FileManager.default.enumerator(
                at: rootDir, includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            )
        else {
            return []
        }

        var samples: [URL] = []
        for case let location as URL in directoryEnumerator {
            let resourceValues = try location.resourceValues(forKeys: [.isDirectoryKey])
            if !(resourceValues.isDirectory ?? false),
                location.path.hasSuffix(".wav") {
                let url = URL(
                    fileURLWithPath: String(location.path))
                samples.append(url)
            }
        }
        return samples
    }
}
