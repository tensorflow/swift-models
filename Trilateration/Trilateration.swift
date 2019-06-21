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
    
    var reference1: ReferenceCoordinate
    var reference2: ReferenceCoordinate
    var reference3: ReferenceCoordinate
    
    @differentiable
    func error(for guess: Coordinate) -> Tensor<Float> {
        let distance1 = guess.euclideanDistance(to: reference1.location)
        let distance2 = guess.euclideanDistance(to: reference2.location)
        let distance3 = guess.euclideanDistance(to: reference3.location)
        let error1 = (reference1.expectedDistance - distance1).squared()
        let error2 = (reference2.expectedDistance - distance2).squared()
        let error3 = (reference3.expectedDistance - distance3).squared()
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
        references.error(for: guess)
    }
    guess.x += grad.x * -0.1
    guess.y += grad.y * -0.1
}

print(guess) // Expected to be [2, 2]
