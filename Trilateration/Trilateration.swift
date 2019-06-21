import TensorFlow

struct Coordinate: Differentiable {
    var x: Tensor<Float>
    var y: Tensor<Float>
    
    @differentiable
    func euclideanDistance(to other: Coordinate) -> Tensor<Float> {
        let squaredDiffX = (x - other.x).squared()
        let squaredDiffY = (y - other.y).squared()
        let sumSquaredDiff = squaredDiffX + squaredDiffY
        return sqrt(sumSquaredDiff)
    }
}

extension Coordinate {
    init(x: Float, y: Float) {
        self.x = Tensor(x)
        self.y = Tensor(y)
    }
}

struct TrilaterationReferences: Differentiable {
    struct ReferenceCoordinate: Differentiable {
        var location: Coordinate
        var expectedDistance: Tensor<Float>
        
        init(location: Coordinate, expectedDistance: Float) {
            self.location = location
            self.expectedDistance = Tensor(expectedDistance)
        }
    }
    
    var ref1: ReferenceCoordinate
    var ref2: ReferenceCoordinate
    var ref3: ReferenceCoordinate
    
    @differentiable
    func error(for guess: Coordinate) -> Tensor<Float> {
        let error1 = (ref1.expectedDistance - guess.euclideanDistance(to: ref1.location)).squared()
        let error2 = (ref2.expectedDistance - guess.euclideanDistance(to: ref2.location)).squared()
        let error3 = (ref3.expectedDistance - guess.euclideanDistance(to: ref3.location)).squared()
        return error1 + error2 + error3
    }
}

let references = TrilaterationReferences(
    ref1: .init(
        location: .init(x: 2, y: -2),
        expectedDistance: 4
    ),
    ref2: .init(
        location: .init(x: 10, y: 8),
        expectedDistance: 10
    ),
    ref3: .init(
        location: .init(x: -1, y: 6),
        expectedDistance: 5
    )
)

var guess = Coordinate(x: 1, y: 1)

for _ in 1...100 {
    let grad = guess.gradient { guess -> Tensor<Float> in
        return references.error(for: guess)
    }
    guess.x += grad.x * -0.1
    guess.y += grad.y * -0.1
}

print(guess) // Expected to be [2, 2]
