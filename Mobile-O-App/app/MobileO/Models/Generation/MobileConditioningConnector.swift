import Foundation
import CoreML
import MLX
import MLXFast

import func MLX.stacked
import func MLX.concatenated

/// Transforms VLM hidden states into DiT conditioning tensors (encoder hidden states + attention mask).
protocol ConditioningConnector {
    func forward(_ hiddenStates: [MLXArray]) throws -> (MLMultiArray, MLMultiArray)
}

/// CoreML-based conditioning connector that fuses the last N VLM hidden layers into DiT conditioning.
class MobileConditioningConnector: ConditioningConnector {

    private let numLayers: Int
    private let inputDim: Int
    private let outputDim: Int
    private let minSeqLen: Int
    private let maxSeqLen: Int

    private var coreMLModel: MLModel?
    private var outputKey: String = "var_1279"

    init(
        modelURL: URL,
        numLayers: Int = 4,
        inputDim: Int = 896,
        outputDim: Int = 2304,
        minSeqLen: Int = 77,
        maxSeqLen: Int = 512
    ) throws {
        self.numLayers = numLayers
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.minSeqLen = minSeqLen
        self.maxSeqLen = maxSeqLen

        let config = MLModelConfiguration()
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true

        self.coreMLModel = try MLModel(contentsOf: modelURL, configuration: config)

        if let firstOutput = coreMLModel!.modelDescription.outputDescriptionsByName.first {
            self.outputKey = firstOutput.key
        }
    }

    /// Stack the last `numLayers` hidden states, pad to min sequence length, and run through CoreML.
    func forward(_ hiddenStates: [MLXArray]) throws -> (MLMultiArray, MLMultiArray) {
        guard hiddenStates.count >= numLayers else {
            throw NSError(domain: "MobileConditioningConnector", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Expected at least \(numLayers) hidden states, got \(hiddenStates.count)"])
        }

        let lastNLayers = Array(hiddenStates.suffix(numLayers))
        let firstLayer = lastNLayers[0]

        guard firstLayer.ndim == 3 else {
            throw NSError(domain: "MobileConditioningConnector", code: -2,
                         userInfo: [NSLocalizedDescriptionKey: "Expected 3D hidden states, got shape \(firstLayer.shape)"])
        }

        let batchSize = firstLayer.dim(0)
        let originalSeqLen = firstLayer.dim(1)
        let hiddenDim = firstLayer.dim(2)

        guard hiddenDim == inputDim else {
            throw NSError(domain: "MobileConditioningConnector", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "Expected hidden_dim=\(inputDim), got \(hiddenDim)"])
        }

        guard originalSeqLen <= maxSeqLen else {
            throw NSError(domain: "MobileConditioningConnector", code: -4,
                         userInfo: [NSLocalizedDescriptionKey: "Sequence length \(originalSeqLen) exceeds maximum \(maxSeqLen)"])
        }

        let stacked = stacked(lastNLayers, axis: 1)
        let effectiveSeqLen = max(originalSeqLen, minSeqLen)

        let (processedStacked, attentionMask) = try prepareSequence(
            stacked,
            originalSeqLen: originalSeqLen,
            effectiveSeqLen: effectiveSeqLen
        )

        let stackedFloat32 = processedStacked.asType(DType.float32)
        MLX.eval(stackedFloat32)

        let inputArray = try mlxToMLMultiArray(
            stackedFloat32,
            shape: [batchSize, numLayers, effectiveSeqLen, hiddenDim]
        )

        guard let model = coreMLModel else {
            throw NSError(domain: "MobileConditioningConnector", code: -5,
                         userInfo: [NSLocalizedDescriptionKey: "CoreML model not loaded"])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "stacked_hidden_states": MLFeatureValue(multiArray: inputArray)
        ])
        let output = try model.prediction(from: input)

        guard let encoderHiddenStates = output.featureValue(for: outputKey)?.multiArrayValue else {
            throw NSError(domain: "MobileConditioningConnector", code: -6,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get output from CoreML model"])
        }

        let outputShape = encoderHiddenStates.shape.map { $0.intValue }
        guard outputShape.count == 3,
              outputShape[0] == batchSize,
              outputShape[1] == effectiveSeqLen,
              outputShape[2] == outputDim else {
            throw NSError(domain: "MobileConditioningConnector", code: -7,
                         userInfo: [NSLocalizedDescriptionKey: "Unexpected output shape: \(outputShape)"])
        }

        return (encoderHiddenStates, attentionMask)
    }

    // MARK: - Helpers

    /// Pad or pass through the stacked hidden states to `effectiveSeqLen`, returning the attention mask.
    private func prepareSequence(
        _ stacked: MLXArray,
        originalSeqLen: Int,
        effectiveSeqLen: Int
    ) throws -> (MLXArray, MLMultiArray) {
        let batchSize = stacked.dim(0)
        let numLayers = stacked.dim(1)
        let hiddenDim = stacked.dim(3)

        var result = stacked
        var maskArray: [Float]

        if originalSeqLen < effectiveSeqLen {
            let padSize = effectiveSeqLen - originalSeqLen
            let padding = MLXArray.zeros(
                [batchSize, numLayers, padSize, hiddenDim],
                dtype: stacked.dtype
            )
            result = concatenated([stacked, padding], axis: 2)
            maskArray = Array(repeating: 1.0, count: originalSeqLen) +
                        Array(repeating: 0.0, count: padSize)
        } else {
            maskArray = Array(repeating: 1.0, count: effectiveSeqLen)
        }

        let attentionMask = try MLMultiArray(
            shape: [batchSize as NSNumber, effectiveSeqLen as NSNumber],
            dataType: .float32
        )

        attentionMask.dataPointer.withMemoryRebound(to: Float.self, capacity: batchSize * effectiveSeqLen) { ptr in
            for b in 0..<batchSize {
                for i in 0..<effectiveSeqLen {
                    ptr[b * effectiveSeqLen + i] = maskArray[i]
                }
            }
        }

        return (result, attentionMask)
    }

    /// Copy an MLXArray's Float32 data into a new MLMultiArray with the given shape.
    private func mlxToMLMultiArray(_ array: MLXArray, shape: [Int]) throws -> MLMultiArray {
        let totalElements = shape.reduce(1, *)
        let data = array.asArray(Float.self)

        let mlMultiArray = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: .float32
        )

        mlMultiArray.dataPointer.withMemoryRebound(to: Float.self, capacity: totalElements) { dstPtr in
            data.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                memcpy(dstPtr, srcBase, totalElements * MemoryLayout<Float>.stride)
            }
        }

        return mlMultiArray
    }
}
