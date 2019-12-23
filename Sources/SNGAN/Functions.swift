import TensorFlow

func resize(images: Tensor<Float>, width: Int, height: Int) -> Tensor<Float> {
    _Raw.resizeBilinear(images: images, size: Tensor<Int32>([Int32(height), Int32(width)]))
}

@differentiable
func lrelu(_ tensor: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(tensor)
}

func l2normalize(_ tensor: Tensor<Float>) -> Tensor<Float> {
    return tensor / sqrt(pow(tensor, 2).sum() + 1e-8)
}

func heNormal(shape: TensorShape) -> Tensor<Float> {
    let out = shape.dimensions.dropLast().reduce(1, *)
    return Tensor<Float>(randomNormal: shape) * sqrt(2 / Float(out))
}

typealias SN = SpectralNorm

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

@differentiable(wrt: tensor)
func depthToSpace(_ tensor: Tensor<Float>, blockSize: Int) -> Tensor<Float> {
    // Currently _Raw.depthToSpace is not differentiable
//    _Raw.depthToSpace(tensor, blockSize: Int64(blockSize))
    
    let (b, h, w, c) = (tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
    let newHeight = h * blockSize
    let newWidth = w * blockSize
    let newDepth = c / (blockSize*blockSize)
    
    precondition(newDepth*blockSize*blockSize == c)
    
    var x = tensor.reshaped(to: [b, h, w, blockSize, blockSize, newDepth])
    x = x.transposed(permutation: 0, 1, 3, 2, 4, 5)
    x = x.reshaped(to: [b, newHeight, newWidth, newDepth])
    return x
}

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

@differentiable(vjp: vjpResize2xBilinear)
public func resize2xBilinear(images: Tensor<Float>) -> Tensor<Float> {
    let newHeight = images.shape[1] * 2
    let newWidth = images.shape[2] * 2
    return _Raw.resizeBilinear(images: images,
                               size: Tensor([Int32(newHeight), Int32(newWidth)]),
                               alignCorners: true)
}

public func vjpResize2xBilinear(images: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>)->Tensor<Float>) {
    let resized = resize2xBilinear(images: images)
    return (resized, { v in
        _Raw.resizeBilinearGrad(grads: v, originalImage: images, alignCorners: true)
    })
}
