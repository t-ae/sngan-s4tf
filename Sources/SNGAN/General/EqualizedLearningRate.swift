import TensorFlow

// https://arxiv.org/abs/1710.10196

struct EqualizedDense<Scalar: TensorFlowFloatingPoint>: Layer {
    var dense: Dense<Scalar>
    
    @noDerivative
    var scale: Scalar
    
    init(_ dense: Dense<Scalar>, enableScaling: Bool = true) {
        self.dense = dense
        precondition(dense.weight.rank == 2, "batched dense is not supported.")
        
        let std = dense.weight.standardDeviation().scalarized()
        
        if enableScaling {
            self.scale = std
            self.dense.weight /= std
        } else {
            self.scale = 1
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dense.activation(matmul(input, dense.weight * scale) + dense.bias)
    }
}

struct EqualizedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    var conv: Conv2D<Scalar>
    
    @noDerivative
    var scale: Scalar
    
    init(_ conv: Conv2D<Scalar>, enableScaling: Bool = true) {
        self.conv = conv
        
        let std = conv.filter.standardDeviation().scalarized()
        
        if enableScaling {
            self.scale = std
            self.conv.filter /= std
        } else {
            self.scale = 1
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return conv.activation(conv2D(
            input,
            filter: conv.filter * scale,
            strides: (1, conv.strides.0, conv.strides.1, 1),
            padding: conv.padding,
            dilations: (1, conv.dilations.0, conv.dilations.1, 1)) + conv.bias)
    }
}
