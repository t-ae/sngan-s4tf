import TensorFlow

public func heNormal<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    return { shape in
        let out = shape.dimensions.dropLast().reduce(1, *)
        return Tensor(randomNormal: shape) * sqrt(2 / Scalar(out))
    }
}

public func glorotUniform<Scalar: TensorFlowFloatingPoint>(
    scale: Scalar,
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    return { shape in
        Tensor(glorotUniform: shape, seed: seed) * scale
    }
}

@differentiable(wrt: tensor)
public func depthToSpace<Scalar: TensorFlowFloatingPoint>(
    _ tensor: Tensor<Scalar>,
    blockSize: Int
) -> Tensor<Scalar> {
    // Currently _Raw.depthToSpace has no grad function
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
