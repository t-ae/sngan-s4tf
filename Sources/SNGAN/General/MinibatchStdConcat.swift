import TensorFlow

public struct MinibatchStdConcat<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative
    public let groupSize: Int
    
    public init(groupSize: Int) {
        self.groupSize = groupSize
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let (b, h, w, c) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        precondition(b.isMultiple(of: groupSize), "Not divisible by `groupSize`: \(b) / \(groupSize)")
        
        var x = input.reshaped(to: [groupSize, b/groupSize, h, w, c])
        let mean = x.mean(alongAxes: 0)
        let variance = squaredDifference(x, mean).mean(alongAxes: 0)
        let std = sqrt(variance + 1e-8) // [1, b/groupSize, h, w, c]
        x = std.mean(alongAxes: 2, 3, 4) // [1, b/groupSize, 1, 1, 1]
        x = x.tiled(multiples: Tensor<Int32>([Int32(groupSize), 1, Int32(h), Int32(w), 1]))
        x = x.reshaped(to: [b, h, w, 1])
        
        return input.concatenated(with: x, alongAxis: 3)
    }
}
