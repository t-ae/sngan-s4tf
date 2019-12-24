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

// Currently unavailable because `conv2DBackpropInput` is not public.
//
//struct EqualizedTransposedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
//    var conv: TransposedConv2D<Scalar>
//
//    @noDerivative
//    var scale: Scalar
//
//    init(_ conv: TransposedConv2D<Scalar>, enableScaling: Bool = true) {
//        self.conv = conv
//
//        let std = conv.filter.standardDeviation().scalarized()
//
//        if enableScaling {
//            self.scale = std
//            self.conv.filter /= std
//        } else {
//            self.scale = 1
//        }
//    }
//
//    @differentiable
//    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
//        let paddingIndex = conv.paddingIndex
//        let strides = conv.strides
//        let filter = conv.filter
//        let bias = conv.bias
//        let padding = conv.padding
//        let activation = conv.activation
//
//        let batchSize = input.shape[0]
//        let h = (input.shape[1] - (1 * paddingIndex)) *
//          strides.0 + (filter.shape[0] * paddingIndex)
//        let w = (input.shape[2] - (1 * paddingIndex)) *
//          strides.1 + (filter.shape[1] * paddingIndex)
//        let c = filter.shape[2]
//        let newShape = Tensor<Int32>([Int32(batchSize), Int32(h), Int32(w), Int32(c)])
//        return activation(conv2DBackpropInput(
//            input,
//            shape: newShape,
//            filter: filter,
//            strides: (1, strides.0, strides.1, 1),
//            padding: padding) + bias)
//    }
//}
