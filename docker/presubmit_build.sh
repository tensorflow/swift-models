#!/bin/bash

set -exuo pipefail

sudo apt-get install -y docker.io

# Sets 'swift_tf_url' to the public url corresponding to
# 'swift_tf_bigstore_gfile', if it exists.
if [[ ! -z ${swift_tf_bigstore_gfile+x} ]]; then
  export swift_tf_url="${swift_tf_bigstore_gfile/\/bigstore/https://storage.googleapis.com}"
  case "$swift_tf_url" in
    *stock*) STOCK_TOOLCHAIN=YES ;;
    *) ;;
  esac
fi

# Help debug the job's disk space.
df -h

# Move docker images into /tmpfs, where there is more space.
sudo /etc/init.d/docker stop
sudo mv /var/lib/docker /tmpfs/
sudo ln -s /tmpfs/docker /var/lib/docker
sudo /etc/init.d/docker start

# Help debug the job's disk space.
df -h

cd github/swift-models
sudo -E docker build -t built-img -f docker/Dockerfile --build-arg swift_tf_url .

# SwiftPM-based build.
docker run built-img /bin/bash -c "
swift build ${STOCK_TOOLCHAIN:+"-Xswiftc -D -Xswiftc TENSORFLOW_USE_STANDARD_TOOLCHAIN"} ;
swift test ${STOCK_TOOLCHAIN:+"-Xswiftc -D -Xswiftc TENSORFLOW_USE_STANDARD_TOOLCHAIN"} ;
"

# CMake-based build.
sudo docker run built-img /bin/bash -c "
set -e;
cmake -B /BinaryCache/tensorflow-swift-models -D CMAKE_BUILD_TYPE=Release -D CMAKE_Swift_COMPILER=/swift-tensorflow-toolchain/usr/bin/swiftc -G Ninja -S /swift-models;
cmake --build /BinaryCache/tensorflow-swift-models --verbose;
"
