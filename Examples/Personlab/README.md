# PersonLab

Personlab human pose estimator, inference only version.

Had to build slightly custom mobilenet backbone as the checkpoint I used does not fit into the mobilenet versions available in this repo.

CLI demo with option to run inference on a local image file and on live video from a local webcam using SwiftCV.

## Demo set up (only tested in Ubuntu)
Installation of a [slightly improved version of SwiftCV](https://github.com/joaqo/SwiftCV) is required. I'll upstream the features I had to add to it for this demo to the official SwiftCV repo soon.

The installation of SwiftCV should be handled automatically by the SPM, but openCV has to be installed manually. OpenCV instsallation is simple, just run the `install/install_cv4.sh` [script](https://github.com/joaqo/SwiftCV/blob/master/install/install_cv4.sh) in the SwiftCV repo.

Finally download the [checkpoint](https://github.com/joaqo/swift-models/releases/download/PersonlabDemo/personlabCheckpoint.tar.gz) from the releases page in this repo, whose path you'll have to provide to the CLI demo.

## Running
```
swift run Personlab --help
```
