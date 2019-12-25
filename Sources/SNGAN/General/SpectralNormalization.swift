import TensorFlow

// https://arxiv.org/abs/1802.05957

// MARK: - Based on PyTorch implementation
// https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html
public typealias SN = SpectralNorm

public struct SpectralNorm<L: Layer>: Layer {
    public var layer: L
    
    @noDerivative
    public let keyPath: WritableKeyPath<L, Tensor<Float>>
    @noDerivative
    public var v: Tensor<Float>
    @noDerivative
    public let numPowerIterations = 1
    @noDerivative
    public let outputAxis: Int
    
    public init(layer: L, keyPath: WritableKeyPath<L, Tensor<Float>>, outputAxis: Int) {
        self.layer = layer
        self.keyPath = keyPath
        self.outputAxis = outputAxis
        let weight = layer[keyPath: keyPath]
        v = Tensor<Float>(randomNormal: [1, weight.shape[outputAxis]])
    }
    
    public mutating func normalize() {
        let weight = layer[keyPath: keyPath]
        let mat = weight.reshaped(to: [-1, weight.shape[outputAxis]]) // [rows, cols]
        
        var u = Tensor<Float>(0)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v, mat.transposed())) // [1, rows]
            v = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.transposed()) // [1, 1]
        layer[keyPath: keyPath] /= sigma.squeezingShape()
    }
    
    @differentiable
    public func callAsFunction(_ input: L.Input) -> L.Output {
        return layer(input)
    }
}

public func spectralNormalize<L: Layer>(_ layer: inout L) {
    for kp in layer.recursivelyAllWritableKeyPaths(to: SpectralNorm<Dense<Float>>.self) {
        layer[keyPath: kp].normalize()
    }
    for kp in layer.recursivelyAllWritableKeyPaths(to: SpectralNorm<Conv2D<Float>>.self) {
        layer[keyPath: kp].normalize()
    }
    for kp in layer.recursivelyAllWritableKeyPaths(to: SpectralNorm<TransposedConv2D<Float>>.self) {
        layer[keyPath: kp].normalize()
    }
}

extension SpectralNorm where L == Dense<Float> {
    public init(_ layer: Dense<Float>) {
        self.init(layer: layer, keyPath: \Dense<Float>.weight, outputAxis: 1)
    }
    
    public var weight: Tensor<Float> {
        layer.weight
    }
    public var bias: Tensor<Float> {
        layer.bias
    }
}

extension SpectralNorm where L == Conv2D<Float> {
    public init(_ layer: Conv2D<Float>) {
        self.init(layer: layer, keyPath: \Conv2D<Float>.filter, outputAxis: 3)
    }
    
    public var filter: Tensor<Float> {
        layer.filter
    }
    public var bias: Tensor<Float> {
        layer.bias
    }
}

extension SpectralNorm where L == TransposedConv2D<Float> {
    public init(_ layer: TransposedConv2D<Float>) {
        self.init(layer: layer, keyPath: \TransposedConv2D<Float>.filter, outputAxis: 3)
    }
    
    public var filter: Tensor<Float> {
        layer.filter
    }
    public var bias: Tensor<Float> {
        layer.bias
    }
}

private func l2normalize<Scalar: TensorFlowFloatingPoint>(_ tensor: Tensor<Scalar>) -> Tensor<Scalar> {
    tensor * rsqrt(tensor.squared().sum() + 1e-8)
}

// MARK: - Based on original chainer implementation
// https://github.com/pfnet-research/sngan_projection/blob/master/source/links/sn_convolution_2d.py
public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var conv: Conv2D<Scalar>
    
    @noDerivative
    public var enabled: Bool
    
    @noDerivative
    public let numPowerIterations = 1
    
    @noDerivative
    public var v: Parameter<Scalar>
    
    public init(_ conv: Conv2D<Scalar>, enabled: Bool) {
        self.conv = conv
        self.enabled = enabled
        v = Parameter(Tensor(randomNormal: [1, conv.filter.shape[3]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard enabled else {
            return conv.filter
        }
        let outputDim = conv.filter.shape[3]
        let mat = conv.filter.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v.value, mat.transposed())) // [1, rows]
            v.value = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.value.transposed()) // [1, 1]
        
        return conv.filter / sigma
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        conv.activation(conv2D(
            input,
            filter: wBar(),
            strides: (1, conv.strides.0, conv.strides.1, 1),
            padding: conv.padding,
            dilations: (1, conv.dilations.0, conv.dilations.1, 1)) + conv.bias)
    }
}
