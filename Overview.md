Swift for TensorFlow Design Overview

Swift for TensorFlow provides a new programming model for TensorFlow - one that combines the performance of graphs with the flexibility and expressivity of Eager execution, while keeping a strong focus on improved usability at every level of the stack.  While most existing machine learning frameworks are designed within the constraint of being a library for a language like Python, we open the door to (carefully considered) language and compiler enhancements.

For users, we’ve designed Swift for TensorFlow to feel like a simple and obvious tool for writing machine learning libraries and models that “just work”.  However, if you look under the hood, the implementation of this user experience is a bunch of mostly independent features and subsystems that compose and feel natural together, but which can also be used in isolation.  

This document provides a high level view of these subcomponents and describe how they interact and fit together.  The goal of this document is to describe the big picture without requiring extensive subject matter expertise.  Technical deep-dive white papers are linked to provide a deeper perspective where it makes sense, as are links to the code itself.

We go describe these pieces of the project:

Swift   // TODO: These will be markdown links to the sections below
TensorFlow  
Code Partitioning
The TensorFlow module
Automatic Differentiation
Python Interoperability
Future Directions
Swift
Swift is an open source general-purpose programming language, which has a large and growing user base.  We chose Swift because it has an open language design process and for specific technical reasons described in the Compiler Code Partitioning whitepaper.  We assume that most readers are unfamiliar with it, so we’ll briefly touch on some additional important things about it here.

The development of Swift started in 2010, and aimed to bring the best practices in programming language design together into one system - rather than trying for academic novelty or to religiously propagate one specific programming methodology.  As a result, it supports multi-paradigm development (e.g. functional, OOP, generic, procedural, etc) all in one system,  and brings many well-known concepts from academic languages (e.g. pattern matching, algebraic data types, and type classes) into the forefront.  Instead of strongly encouraging developers to rewrite all their code in Swift, it pragmatically focuses on interoperability with other languages, e.g., allowing you to directly import C header files and use them without an FFI and (now) the ability to use Python APIs without wrappers.

Swift has the audacious goal of spanning all the way from low-level systems programming to high-level scripting, with a focus on being easy to learn and use.  Because Swift needs to be easy to learn and use but also powerful, it relies on the principle of progressive disclosure of complexity, which aggressively factors the cost of complexity onto the people who benefit from the complexity.  The “scripting language feel” combined with high performance is very useful for machine learning.

A final pertinent aspect of the design of Swift is that much of the Swift language is actually implemented in its standard library.  “Builtin” types like “Int” and “Bool” are actually just structs defined in the standard library that wrap magic types and operations.  As such, sometimes we joke that Swift is just “syntactic sugar for LLVM”.   This capability is very important to our work because the Tensor type in the [TensorFlow module](TODO: Link to the section below) is just syntactic sugar for TensorFlow, and the `PyValue` type is just syntactic sugar for `PyObject*`!

There is a lot more that is cool about Swift and a ton of content available online.  If you are interested in learning more about general Swift programming concepts, here are a few links to get started:

A skimmable tour of the high level syntax and feel: A Swift Tour
Value semantics are powerful and play an important role in Swift code: Building Better Apps with Value Types in Swift
Swift supports classic OOP, but has a better model adopted from Haskell: Protocol-Oriented Programming in Swift
TODO: … other links.??

One warning: Swift evolved rapidly in its early years, so you should be careful trusting anything before Swift 3 (which was released in 2016).
TensorFlow
TensorFlow is a popular and widely-used machine learning framework, which hopefully needs no further introduction.  Though most users of Swift for TensorFlow will think of it as a new language binding for TensorFlow, it works completely differently than the other bindings.

The most common way to use TensorFlow today is to build a graph, and then evaluate it one or more times with the session APIs.  The Python, Java, Go, C++ and C bindings for TensorFlow provide wrappers for their languages that expose these graph building operations directly to the user-visible programming model: you write code that builds a graph, then more code to execute it.

In contrast, with Swift, you just write code that performs a tensor computation using types defined in the TensorFlow module, and Swift accelerates the tensor operations transparently with TensorFlow.  The magic behind this is a [code partitioning](TODO: LINK BELOW) compiler transformation that extracts the graph out of your code, and inserts calls to the TensorFlow runtime for you.  The nice thing about this is that TensorFlow “just works”: the user doesn’t have to think about graphs at all.  They just write simple “eager execution style” code against the Tensor APIs, and they get all the benefits of graphs, such as holistic graph-based performance optimization in TensorFlow.

For the purposes of our work, the useful thing to know about TensorFlow is how it represents graphs.  We use TF_Function’s to represent tensor computation, which are a function that takes some number of tensor inputs and produces some number of tensor results.  Each “op”/”node” in a TensorFlow graph is defined by a string op name, a list of input values, a list of attributes (which are guaranteed to be constants), and produces some number of tensor results.  Each input and result value has a “dtype” associated with it that describes the element type (specified by the `TF_DataType` enum), and attributes also have their own simple type system (integer, string, float, shape, etc).  The details of this are described in the TensorFlow documentation.  

Swift for TensorFlow has a low-level syntax that gives you direct access to any op, using a magic `#tfop` syntax (which is a placeholder that is likely to be revised).  For example, here are a few methods defined on the Tensor type (simplified slightly for presentation), you can see their full definition in [Ops.swift](TODO: LINK TO GITHUB).

```swift
struct Tensor<Scalar> {
 …
  // Implement the infix `+` operator on Tensor in terms of the TensorFlow `Add` op, 
  // which takes two input tensors and returns one result.
  static func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    return #tfop("Add", lhs, rhs)
  }


  // Another example that implements a method in terms of the TensorFlow `Conv2D` op, 
  // which takes two input tensors, as well as a `strides` and `padding` attribute.
  func convolved2D(withFilter filter: Tensor,
                   strides: (Int32, Int32, Int32, Int32),
                   padding: Padding) -> Tensor {
    return #tfop("Conv2D", handle, filter,
                 strides: [strides.0, strides.1, strides.2, strides.3],
                 padding: padding.cName)
  }
}
```

While the `+` example is very simple, the convolution example shows another important role that these functions play: they are adaptors that handle bridging between the “Swift way of thinking about things” and the “TensorFlow way of thinking about things”.  For example, Swift programmers get to think about paddings as a Swift enum, even though TensorFlow takes strings.  Similarly, strides can be passed as a strongly-typed 4-ary tuple and this code handles erasing that type information when passing it to TensorFlow as an array.
Compiler Code Partitioning
The code partitioning transformation is the key technique that allows TensorFlow integration to work in a seamless way.  Partitioning acts like an additional stage in the compiler, which uses static analysis to find tensor operations and split them out to a TensorFlow graph.  At a high level, the enhanced Swift compiler looks like this:



First, the compiler finds the tensor operations in the code (which is trivial due to the low-level `#tfop` syntax described above).  Next, it desugars high-level abstractions (like structs, tuples, generics, functions, variables, etc) that connect tensor operations through a process called “deabstraction”.  After deabstraction, the tensor operations are directly connected to each other through SSA dataflow edges and are embedded in a control flow graph represented in the Swift Intermediate Language (SIL).  The code for this is primarily implemented in [TFDeabstraction.cpp](Link to Github).

Once the tensor operations are desugared, a transformation called “partitioning” extracts the operations from the program and builds a new SIL function to represent the tensor code.  In addition to removing the tensor operations from the host code, new calls are injected that call into [our new runtime library](TODO: Link below) to start up TensorFlow, rendezvous to collect any results, and send/receive values between the host and the tensor program as it runs.  The bulk of the partitioning transformation itself lives in [TFPartition.cpp](TODO: LINK TO GITHUB)

Once the tensor function is formed, it has some transformations applied to it, and is eventually emitted to a TensorFlow graph using the code in [TFLowerGraph.cpp](TODO: LINK TO GITHUB). After the TensorFlow graph is formed, we serialize it to a protobuf and encode the bits directly into the executable, making it easy to load at program runtime.

We aren’t aware of any other system using this approach, but our implementation draws on a lot of related conceptual work, including program slicing, abstract interpretation, and is implemented as a static compiler analysis.  Please see our detailed code partitioning whitepaper for more information on how all of this works.

Finally, while TensorFlow is the reason we built this infrastructure, its algorithms are independent of TensorFlow itself: the same compiler transformation can extract any computation that executes asynchronously from the host program while communicating through sends and receives.  This is useful and can be applied to anything that represents computation as a graph, including other ML frameworks, other kinds of accelerators (for cryptography, graphics, transcoding, etc), and general distributed systems programming models based on graph abstractions.  We’re interested in exploring generalizations in the future.
The TensorFlow module
The TensorFlow module is the library of code you get as a result of `import TensorFlow` in a Swift program.  It is written in Swift and lives in the [stdlib/public/TensorFlow](TODO: github link) directory.  It implements a few different things:
User APIs: Tensor, ShapedArray, etc.
As we described in the [section about Swift](TODO: LINK), a lot of the Swift experience is actually defined in the standard library, not the compiler itself.  Similarly, because our code partitioning approach is so general and flexible, the TensorFlow module defines most of the user experience and feel of working with TensorFlow - it isn’t baked into the language or compiler.  As such, we have a lot of latitude to experiment with different approaches in the TensorFlow library. 

Our most significant design constraint is that we don’t want users of Swift for TensorFlow to write code that accidentally causes unnecessary copies back and forth between the host and the accelerator.  Because of this, we chose to implement a user model that provides two primary concepts: “arrays” and “tensors”.  Both of these represent n-dimensional tensors of values, but the “arrays” in our system should be thought of as data in the host program, whereas “tensors” are values that are primarily managed by TensorFlow.  Among other things, this means that “arrays” conform to `MutableCollection` and `RangeReplaceableCollection` and thus have normal collection APIs, but `Tensor` has methods and operators that correspond to TensorFlow ops.

Both “arrays” and “tensors” have dynamically ranked n-dimensional versions, named `ShapedArray` and `Tensor` respectively.  We are also experimenting with statically ranked versions (`Array2D`, `Array3D`, etc which compose on top of `Swift.Array`) and (`Tensor1D`, `Tensor2D`, `Tensor3D`, etc).  [[TODO: a bunch of links to our API docs]].  Here are a couple of simple examples showing `Tensor` and `ShapedArray`:

```
// `Tensor` examples.
var matrix = Tensor<Float>(shape: [2, 2], scalars: [1, 2, 3, 4])
var matrix: Tensor<Float> = [[1, 2], [3, 4]]
// `matrix` represents [[1, 2], [3, 4]].

// Arithmetic operations, using TensorFlow.
let sum = matrix + matrix
let sqrt = sqrt(matrix)
let matrixProduct = matrix.dot(matrix)
// `sum` represents [[2.0, 4.0], [6.0, 8.0]].
// `sqrt` represents [[1.0, 1.41421], [1.73205, 2.0]].
// `matrixProduct` represents [[7.0, 10.0], [15.0, 22.0]].

// Convert `Tensor` to `ShapedArray`.
let array2D = ShapedArray(matrix)
// `array2D` is stored on the host.
```

```
// `ShapedArray` examples.
var matrix = ShapedArray(shape: [3, 2], scalars: [1, 2, 0, 0, 5, 6])
// `matrix` represents [[1, 2], [0, 0], [5, 6]].

let element = matrix[0]
// `element` is a `ShapedArraySlice` with shape [2], representing [1, 2].

matrix[1] = ShapedArraySlice(shape: [2], scalars: [3, 4])
// The second element in `matrix` has been mutated.
// `matrix` now represents [[1, 2], [3, 4], [5, 6]].

let zeros = ShapedArray(shape: [3, 2], repeating: 0)
let subarray = matrix.prefix(2)
// `subarray` is a `ShapedArraySlice` with shape [2, 2], representing [[1, 2], [3, 4]].
matrix[0..<2] = zeros.prefix(2)
// The first 2 elements in `matrix` have been modified.
// `matrix` now represents [[0, 0], [0, 0], [5, 6]].

// Convert `ShapedArray` to `Tensor`.
let tensor2D = Tensor(matrix)
// It’s now possible to perform TensorFlow operations on `tensor2D`.
```

The implementation of `Tensor` builds on the `#tfop` magic syntax that builds TensorFlow graph nodes, and is defined in [Tensor.swift](TODO), [Ops.swift](TODO), [RankedTensor.swift.gyb](TODO), and [TensorProtocol.swift](TODO).  The implementation of `ShapedArray` follows standard techniques used when implementing Swift collections and is defined primarily in [ShapedArray.swift](TODO: GitHub URL) and [RankedArray.swift.gyb](TODO).  In addition to the `Tensor` family of types, we are experimenting with building abstractions on top of the TensorFlow graph nodes for data pipelines, resources, variants, and other things representable as graph nodes.
Runtime Entry Points for Partitioning
The [code partitioning transformation](TODO: Link to section above) extracts the tensor operations out to a TensorFlow graph and serializes it to a protobuf that is encoded into the program’s executable.  It rewrites the host code to insert calls to “start tensor program”, “finish tensor program”, and “terminate tensor program” runtime entry points, which are implemented in the [CompilerRuntime.swift](TODO: URL) file in terms of TensorFlow APIs.

Our runtime currently has several supported paths for driving TensorFlow, including paths that enable XLA, paths that go through classic Executor, paths that uses the “eager execution” runtime entry points, and some specialized support for CloudTPU configurations.  This is still rapidly evolving and subject to continuous change.

The most significant unimplemented piece of our compiler and runtime model is support for sending and receiving data between co-executing asynchronous host and TensorFlow programs.  This is an incredibly important part of our model that allows you to transparently use host calls (e.g. `print` or Xcode Playground value logging) on Tensor values, and intermix host and accelerator code freely.  This is a top priority to implement in the coming weeks.  In the meantime, we have full support for arguments and results that are passed and received at the start and end of the tensor program.
Automatic Differentiation
Automatic differentiation (AD) is a powerful technique that all machine learning frameworks are expected to implement, because gradients are so important for this work (e.g. with SGD).  TensorFlow implements automatic differentiation as a TensorFlow graph transformation, but we would like to deploy more powerful techniques to enable custom data structures, recursion, and higher-order differentiation.  As such, we built a stand-alone AD feature for Swift: one that is completely independent of the standard TensorFlow implementation of AD, and also completely independent of TensorFlow support in Swift.  

The way this works is by having Swift AD support arbitrary user-defined types.  Swift for TensorFlow builds on this by making its Tensor types conform to the AD system, allowing them to participate as you’d expect.  A nice thing about this is that Swift programmers interested in non-Tensor numerical analysis can use AD for any other types that are important for their work.


Automatic differentiation in Swift is a compiler IR transformation, based upon static analysis. When differentiating a function in reverse mode, the compiler produces separate functions that contain the corresponding “primal code” and “adjoint code”, which in turn compute the partial derivatives of the model output with respect to the input parameters. Since we want AD in Swift to be completely general across all use cases and allow custom data structures and arbitrary functions, the compiler makes no assumption on individual math operations.  Instead, the developer specifies the adjoint code to use for a function, and how two back-propagated adjoint values should combine - all in pure Swift code. The compiler will then differentiatie and chain any usage of these functions.

We use the `@differentiable` attribute on a function to specify the custom adjoint for the function. The first parameter to `@differentiable` specifies whether the function is differentiable using forward-mode (not supported yet) or reverse-mode AD. The second argument specifies an adjoint function that takes the original arguments, the original result and a seed (back-propagated adjoint value from another function) and computes the gradient.

```swift
@differentiable(reverse, adjoint: adjointLog)
func log(_ x: Float) -> Float {
  ...
}

func adjointLog(_ x: Float, originalResult: Float, seed: Float) -> Float {
  return seed / x
}
```

In addition to making the operator differentiable, the compiler needs to know how to combine two derivatives in the backward pass (usually a `+`, but sometimes broadcasting), and how to create a default seed by broadcasting a number to match the original result’s shape. Each type has different behavior, e.g. `Float` vs. `Tensor<Float>`.  This is implemented with a conformance to the `Differentiable` protocol, and we’ve made all `FloatingPoint` types conform, and `Tensor` also conforms when its scalar type is `Differentiable`. With this foundation, the user can request the gradient of any function so long as the parameter types, the return type, and the functions called along the data flow are differentiable. When any operation along the data flow is not differentiable, e.g. a call to a non-differentiable function or an assignment to a global variable, the compiler will produce a compile time error and point to the relevant location.

We provide two compiler-intrinsic operators for requesting the gradient of a function: `#gradient(of:)` and `#valueAndGradient(of:)`. The former takes a function and returns another function that computes the gradient of the input function. The latter takes a function and returns another function that computes both the result and the gradient of the original function. An optional variadic argument `wrt:` specifies the indices of parameters (of `self`) to differentiate with respect to. The following example demonstrates how to request the gradient of a differentiable function with respect to certain arguments.

```swift
func cube<T : FloatingPoint>(_ x: T, _ str: String) -> T {
  print(str)
  return x * x * x
}

let dCube = #gradient(of: cube, wrt: .0)

cube(5, “hi”)  // prints “hi” and returns 125
dCube(5, “hi”) // prints “hi” and returns 75
```

Today, we have basic support for reverse-mode AD on straight-line code, but we plan to complete support for full control flow and discuss the need for forward-mode AD with the community.  To learn more about Swift automatic differentiation, see [Automatic Differentiation Deep Dive](TODO: Link to the deep dive document).
Python Interoperability
A large part of the machine learning community uses Python, and we heavily leverage the massive data science, visualization, and other random packages that Python provides to get our jobs done.  Swift for TensorFlow has a lot of technical and usability advantages over Python for TensorFlow, and Swift fixes many suboptimal things about Python (e.g. by providing much higher performance out of the box and by not having a GIL).  However, it is perfectly clear that being “not Python” will be the biggest impediment to getting users to try and use Swift for TensorFlow in practice.

There are a few things that we can do to reduce the burden of moving from programming in Python to programming in Swift for TensorFlow.  For example, Swift already supports a command line interpreter, and `#!` script workflows.  Adding Jupyter Notebook integration will make it easy to try out Swift for TensorFlow and is a popular part of many people’s workflows.  

To further smooth the transition, we made it possible to directly call Python APIs from Swift, which allows ML programmers to continue using data science and other useful APIs while also getting the benefits of Swift for their TensorFlow code.  Here is an example of what this looks like in practice, with commented out code that shows the pure-Python syntax for comparison:

```swift
// NumPy example:
let np = Python.import("numpy")             // import numpy as np
let a = np.arange(15).reshape(3, 5)         // a = np.arange(15).reshape(3, 5)
let b = np.array([6, 7, 8])                 // b = np.array([6, 7, 8])

// Pickle example:
let gzip = Python.import("gzip")            // import gzip as gzip
let pickle = Python.import("pickle")        // import pickle as pickle
let file = gzip.open("mnist.pkl.gz", "rb")  // file = gzip.open("mnist.pkl.gz", "rb")
                                       // (images, labels) = pickle.load(file)
let (images, labels) = pickle.load(file).tuple2
print(images.shape) // (50000, 784)            print(images.shape)
```

As you can see, the syntax here is very close: the major differences are that Swift requires values to be declared before use, and that we decided to put Python builtins functions like `import`, `type`, `slice`, etc under a `Python.` namespace (to avoid cluttering the global scope).  This doesn’t require SWIG or any other wrappers, so it is super easy to use.

This feature is accomplished without making Python specific changes to the compiler or language - it is completely implemented in the [Python.swift file](link to Python.swift on GitHub).  This means that we can use the same techniques to directly integrate with other dynamic language runtimes (e.g. Javascript, Ruby, etc) if it becomes important in the future.  Python support is also completely independent of the other TensorFlow and automatic differentiation logic we’re building in the rest of the project.  It is a generally useful extension to the Swift ecosystem that can stand alone, useful for server side development or anything else that wants to interoperate with existing Python APIs.

To find out more about how this works, please check out the Python Interoperability Deep Dive, or browse the implementation in [Python.swift on GitHub](link to Python.swift on GitHub).
Future Directions
We’re focusing on finishing the basic Swift for TensorFlow model, gaining more experience using it, and start building a developer community.  Despite that, we have tons of ideas for how to push things forward - and welcome additional ideas of course!  For example:

Availability Checking: Swift has a powerful model for working with conditionally available functionality known as “availability checking”, and the TensorFlow ecosystem has many similar challenges: many ops are only available on certain devices, some ops only work with certain dtypes, and some deployment targets like XLA and TFLite have additional restrictions.  We’d like to consider extending availability checking or building a similar system to allow us to statically diagnose misuse of Tensor ops for the hardware and configuration you’re deploying for.  We should be able to directly point to the problematic line of code and give a detailed error message about problems we detect.

Deployment Support: We’d like to explore a model where deployed models are explicitly declared in code, including the device(s) they are intended to support.  This enables improved availability checking (described above), allows better management of the interfaces used for inference, eliminates certain classes of bugs, and should directly support deployment workflows that want to update weight values without recompiling and changing code.  We have an initial plan of how to pursue this but need to develop the ideas out more.

Shape Checking: Shape errors are an ongoing problem for machine learning research and productivity.  We already have some basic shape checks from the TensorFlow graph building APIs, but we need to invest in first class support for this to produce better diagnostics and diagnose more errors at high level API boundaries instead of inside the implementation of those APIs.  We have initial ideas of how this could work, but need to explore this much further.

Named Dimensions: A frequently requested feature is to be able to use symbolic dimension names in Tensors.  There are several different possible models that could be explored here.

Differentiating Opaque Closures: Statically differentiating a function requires the body of the function to be visible to the compiler. However, this limits the expressiveness of the gradient operator, e.g. users can’t apply the gradient operator to a function argument that has a function type because the compiler can’t always see into the declaration of the original function. We will discuss the possibility to introduce a new function convention - when a differentiable function is passed around, a pointer to its primal and adjoint gets passed along. This enables the compiler to directly call the primal and the adjoint, without the need to see into the function declaration.  This is important for class and protocol methods.

Quantization Support: We believe we can get a much better user experience for fixed-point quanitization tools if we integrate them into the compiler, and this should help with integrating quanitization into the training process.  We haven’t started thinking about this though.

