# PersonLab

Personlab human pose estimator, inference only version.

Had to build slightly custom mobilenet backbone as the checkpoint I used does not fit into the mobilenet versions available in this repo.

CLI demo with option to run inference on a local image file and on live video from a local webcam using SwiftCV.

## Installation (Only tested on Ubuntu)
Installation of a [slightly improved version of SwiftCV](https://github.com/joaqo/SwiftCV) is required. I'll upstream the features I had to add to it for this demo to the official SwiftCV repo soon.

First install OpenCV by running the `install/install_cv4.sh` [script](https://github.com/joaqo/SwiftCV/blob/master/install/install_cv4.sh) in SwiftCV repo, and then just add the following to Package.swift.

```Swift
.package(url: "https://github.com/joaqo/SwiftCV.git", .branch("master"))
```

Finally download the checkpoint from the releases page in this repo.

## Running
```
swift run Personlab --help
```
