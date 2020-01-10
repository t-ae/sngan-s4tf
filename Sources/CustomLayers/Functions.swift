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

//@derivative(of: resize2xBilinear)
public func vjpResize2xBilinear(images: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>)->Tensor<Float>) {
    let resized = resize2xBilinear(images: images)
    return (resized, { v in
        _Raw.resizeBilinearGrad(grads: v, originalImage: images, alignCorners: true)
    })
}

@differentiable(wrt: images, vjp: vjpResizeNN)
public func resizeNN(images: Tensor<Float>,
                     width: Int,
                     height: Int,
                     alignCorners: Bool) -> Tensor<Float> {
    _Raw.resizeNearestNeighbor(images: images,
                               size: Tensor([Int32(height), Int32(width)]),
                               alignCorners: true)
}

//@derivative(of: resizeNN)
@usableFromInline
func vjpResizeNN(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resize2xBilinear(images: images)
    return (resized, { v in
        _Raw.resizeNearestNeighborGrad(grads: v,
                                       size: Tensor([Int32(height), Int32(width)]),
                                       alignCorners: alignCorners)
//        (grads: v, originalImage: images, alignCorners: true)
    })
}


@differentiable(wrt: images, vjp: vjpResizeBL)
public func resizeBL(images: Tensor<Float>,
                     width: Int,
                     height: Int,
                     alignCorners: Bool) -> Tensor<Float> {
    _Raw.resizeBilinear(images: images,
                        size: Tensor([Int32(height), Int32(width)]),
                        alignCorners: true)
}

//@derivative(of: resizeBL)
@usableFromInline
func vjpResizeBL(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBL(images: images,
                           width: width,
                           height: height,
                           alignCorners: alignCorners)
    return (resized, { v in
        _Raw.resizeBilinearGrad(grads: v,
                                originalImage: images,
                                alignCorners: alignCorners)
    })
}

@differentiable(wrt: images, vjp: vjpResizeBC)
public func resizeBC(images: Tensor<Float>,
                     width: Int,
                     height: Int,
                     alignCorners: Bool = false,
                     halfPixelCenters: Bool = false) -> Tensor<Float> {
    _Raw.resizeBicubic(images: images,
                       size: Tensor([Int32(height), Int32(width)]),
                       alignCorners: alignCorners,
                       halfPixelCenters: halfPixelCenters)
}

@usableFromInline
func vjpResizeBC(
    images: Tensor<Float>,
    width: Int,
    height: Int,
    alignCorners: Bool,
    halfPixelCenters: Bool
) -> (value: Tensor<Float>, pullback: (Tensor<Float>)->Tensor<Float>) {
    let resized = resizeBC(images: images,
                           width: width,
                           height: height,
                           alignCorners: alignCorners,
                           halfPixelCenters: halfPixelCenters)
    return (resized, { v in
        _Raw.resizeBicubicGrad(grads: v,
                               originalImage: images,
                               alignCorners: alignCorners,
                               halfPixelCenters: halfPixelCenters)
    })
}
