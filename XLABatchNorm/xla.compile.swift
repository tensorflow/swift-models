import TensorFlow

struct PullbackArgs<T : TensorGroup, U : TensorGroup> : TensorGroup {
  let input: T
  let cotangent: U
}

func xlaCompiled<T : Differentiable & TensorGroup, U : Differentiable & TensorGroup>(
  _ fn: @escaping @differentiable (T) -> U) -> @differentiable (T) -> U
where T.CotangentVector : TensorGroup, U.CotangentVector : TensorGroup
{
  let xlaCompiledFn: (T) -> U = _graph(fn, useXla: true)
  let xlaCompiledPullback = _graph(
    { (pbArgs: PullbackArgs<T, U.CotangentVector>) in
      pullback(at: pbArgs.input, in: fn)(pbArgs.cotangent)
    },
    useXla: true)
  return  differentiableFunction { x in
      (value: xlaCompiledFn(x),
      pullback: {
        v in
        xlaCompiledPullback(PullbackArgs(input: x, cotangent: v))}
      )
  }
}
  


@differentiable 
func square(_ a: Tensor<Float>) -> Tensor<Float>{
  return a * a 
}

let blah = xlaCompiled(square)
let b = blah(Tensor<Float>(5.0))
print("b \(b)")

withDevice(.cpu) {
  let input = Tensor<Float>(10.0)
  let computation = xlaCompiled(square)
  let res = computation(input)
  let diff = gradient(at: input, in: computation)
  print("in: \(input), sq: \(res) grad: \(diff) ")
}
