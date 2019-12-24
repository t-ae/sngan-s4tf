import TensorFlow

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

@differentiable(wrt: tensor)
func depthToSpace(_ tensor: Tensor<Float>, blockSize: Int) -> Tensor<Float> {
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
