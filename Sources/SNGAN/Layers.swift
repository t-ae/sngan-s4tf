import TensorFlow

typealias SN = SpectralNorm

// MARK: - Spectral normalization
struct SpectralNorm<L: Layer>: Layer {
    var layer: L
    
    @noDerivative
    let keyPath: WritableKeyPath<L, Tensor<Float>>
    @noDerivative
    var v: Tensor<Float>
    @noDerivative
    let numPowerIterations = 1
    @noDerivative
    let outputAxis: Int
    
    init(layer: L, keyPath: WritableKeyPath<L, Tensor<Float>>, outputAxis: Int) {
        self.layer = layer
        self.keyPath = keyPath
        self.outputAxis = outputAxis
        let weight = layer[keyPath: keyPath]
        v = Tensor<Float>(randomNormal: [1, weight.shape[outputAxis]])
    }
    
    mutating func normalize() {
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
    func callAsFunction(_ input: L.Input) -> L.Output {
        return layer(input)
    }
}

func spectralNormalize<L: Layer>(_ layer: inout L) {
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
    init(_ layer: Dense<Float>) {
        self.init(layer: layer, keyPath: \Dense<Float>.weight, outputAxis: 1)
    }
    
    var weight: Tensor<Float> {
        layer.weight
    }
}

extension SpectralNorm where L == Conv2D<Float> {
    init(_ layer: Conv2D<Float>) {
        self.init(layer: layer, keyPath: \Conv2D<Float>.filter, outputAxis: 3)
    }
    
    var filter: Tensor<Float> {
        layer.filter
    }
}

extension SpectralNorm where L == TransposedConv2D<Float> {
    init(_ layer: TransposedConv2D<Float>) {
        self.init(layer: layer, keyPath: \TransposedConv2D<Float>.filter, outputAxis: 3)
    }
    
    var filter: Tensor<Float> {
        layer.filter
    }
}

// MARK: - Conditional batch normalization
struct ConditionalBN: Layer {
    struct Input: Differentiable {
        var feature: Tensor<Float>
        @noDerivative
        var label: Tensor<Int32>
        
        @differentiable
        init(feature: Tensor<Float>, label: Tensor<Int32>) {
            self.feature = feature
            self.label = label
        }
    }
    
    @noDerivative
    let featureCount: Int
    
    var bn: BatchNorm<Float>
    
    var gammaEmb: Embedding<Float>
    var betaEmb: Embedding<Float>
    
    init(featureCount: Int) {
        self.featureCount = featureCount
        self.bn = BatchNorm(featureCount: featureCount)
        self.gammaEmb = Embedding(embeddings: Tensor<Float>(ones: [10, featureCount]))
        self.betaEmb = Embedding(embeddings: Tensor<Float>(zeros: [10, featureCount]))
    }
    
    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        let x = bn(input.feature)
        
        let gamma = gammaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        let beta = betaEmb(input.label).reshaped(to: [-1, 1, 1, featureCount])
        
        return x  * gamma + beta
    }
}

// Mark: - MinibatchStdConcat
struct MinibatchStdConcat: ParameterlessLayer {
    @noDerivative
    var groupSize: Int
    
    init(groupSize: Int) {
        self.groupSize = groupSize
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
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

// Mark: - Configurable wrapper
struct Configurable<L: Layer>: Layer where L.Input == L.Output {
    var layer: L
    @noDerivative
    var enabled: Bool
    
    init(_ layer: L, enabled: Bool = true) {
        self.layer = layer
        self.enabled = enabled
    }
    
    @differentiable
    func callAsFunction(_ input: L.Input) -> L.Output {
        if enabled {
            return layer(input)
        } else {
            return input
        }
    }
}
