
# XCodeGen - generate Tensorflow for Swift xcode project #               
**brew install xcodegen**    
**xcodegen**    
open project    
**File > Project Settings > Build System > Legacy Build System** (otherwise you'll see TensorFlow not found error)    



This uses the project.yml file included to handle these steps. 

  In your target's Build Settings:
   * Go to `Swift Compiler > Code Generation > Optimization Level` and select `Optimize for Speed [-O]`.
   * Add `libtensorflow.so` and `libtensorflow_framework.so` to `Linked Frameworks and Libraries` and change `Runtime Search Paths`.
     See [this comment](https://github.com/tensorflow/swift/issues/10#issuecomment-385167803) for specific instructions with screenshots.
   * Go to `Linking > Other Linker Flags` and add `-lpython` to the list of flags.
