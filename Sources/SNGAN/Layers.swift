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

// MARK: - Upsampling conv
struct UpSamplingConv2D: Layer {
    enum Method: String, Codable {
        case convStride, nearestNeighbor, bilinear, depthToSpace
    }
    
    @noDerivative
    var method: Method
    var conv: SN<TransposedConv2D<Float>>
    
    init(inputDim: Int, outputDim: Int, kernelSize: Int, method: Method) {
        self.method = method
        let depthFactor = method == .depthToSpace ? 4 : 1
        let strides = method == .convStride ? (2, 2) : (1, 1)
        let filterShape = (kernelSize, kernelSize, outputDim*depthFactor, inputDim)
        conv = SN(TransposedConv2D(filterShape: filterShape, strides: strides,
                                   padding: .same, filterInitializer: heNormal))
    }
    
    var upsampling = UpSampling2D<Float>(size: 2)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        
        switch method {
        case .convStride:
            break
        case .nearestNeighbor:
            x = upsampling(x)
        case .bilinear:
            x = resize2xBilinear(images: x)
        case .depthToSpace:
            x = depthToSpace(x, blockSize: 2)
        }
        
        return x
    }
    
    var filter: Tensor<Float> {
        conv.filter
    }
}

// MARK: - Downsample conv
struct DownSamplingConv2D: Layer {
    enum Method: String, Codable {
        case convStride, avgPool
    }
    
    @noDerivative
    var method: Method
    var conv: SN<Conv2D<Float>>
    
    init(inputDIm: Int, outputDim: Int, kernelSize: Int, method: Method) {
        self.method = method
        let strides = method == .convStride ? (2, 2) : (1, 1)
        conv = SN(Conv2D(filterShape: (kernelSize, kernelSize, inputDIm, outputDim), strides: strides,
                         padding: .same, filterInitializer: heNormal))
    }
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = conv(input)
        
        switch method {
        case .convStride:
            break
        case .avgPool:
            x = avgPool(x)
        }
        
        return x
    }
    
    var filter: Tensor<Float> {
        conv.filter
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

// MARK: - Instance normalization
struct InstanceNorm2D<F: TensorFlowFloatingPoint>: ParameterlessLayer {
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        precondition(input.rank == 4)
        
        let mean = input.mean(alongAxes: 1, 2)
        let variance = squaredDifference(input, mean).mean(alongAxes: 1, 2)
        return input * rsqrt(variance + 1e-8)
    }
}

// MARK: - Normalization selector
struct XNorm: Layer {
    enum Method: String, Codable {
        case none, instanceNorm, batchNorm
    }
    
    @noDerivative
    var method: Method
    
    var batchNorm: BatchNorm<Float>
    var instanceNorm: InstanceNorm2D<Float>
    
    init(method: Method, dim: Int) {
        self.method = method
        
        batchNorm = BatchNorm(featureCount: dim)
        instanceNorm = InstanceNorm2D()
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        switch method {
        case .none:
            return input
        case .instanceNorm:
            return instanceNorm(input)
        case .batchNorm:
            return batchNorm(input)
        }
    }
}
