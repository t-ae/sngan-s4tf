import XCTest
import class Foundation.Bundle
import TensorFlow
import CustomLayers

final class CustomLayersTests: XCTestCase {
    func testMinibatchStdConcat() throws {
        let layer = MinibatchStdConcat<Float>(groupSize: 4)
        
        var batch = Tensor<Float>((0..<8).map { Float($0) }).reshaped(to: [8, 1, 1, 1])
        batch[0] += 100
        
        let out = layer(batch)
        
        XCTAssertEqual(out.shape, [8, 1, 1, 2])
        
        XCTAssertEqual(out[2, 0, 0, 1], out[0, 0, 0, 1])
        XCTAssertEqual(out[4, 0, 0, 1], out[0, 0, 0, 1])
        XCTAssertEqual(out[6, 0, 0, 1], out[0, 0, 0, 1])
        
        XCTAssertEqual(out[3, 0, 0, 1], out[1, 0, 0, 1])
        XCTAssertEqual(out[5, 0, 0, 1], out[1, 0, 0, 1])
        XCTAssertEqual(out[7, 0, 0, 1], out[1, 0, 0, 1])
    }
    
    func testActivation() {
        let act = Activation(method: .leakyRelu)
        let input = Tensor<Float>(randomNormal: [100, 100])
        
        XCTAssertEqual(act(input), leakyRelu(input))
        
        let g = gradient(at: input) { input in
            act(input).sum()
        }
        let g2 = gradient(at: input) { input in
            leakyRelu(input).sum()
        }
        XCTAssertEqual(g, g2)
    }
    
    func testDepthToSpace() {
        let input = Tensor<Float>([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43])
            .reshaped(to: [1, 2, 2, 4])
        let output = depthToSpace(input, blockSize: 2)
        XCTAssertEqual(output, Tensor([[10, 11, 20, 21],
                                       [12, 13, 22, 23],
                                       [30, 31, 40, 41],
                                       [32, 33, 42, 43]]).reshaped(to: [1, 4, 4, 1]))
    }
    
    static var allTests = [
        ("testMinibatchStdConcat", testMinibatchStdConcat),
    ]
}
