import CoreML
import Foundation

// MARK: - Protocol-Based Model Wrappers

extension MobileOGenerator {

    // MARK: - Transformer Protocol

    /// Unified API for SANA DiT transformer noise prediction
    protocol TransformerModel {
        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray
    }

    // MARK: - VAE Protocol

    /// Unified API for SANA VAE latent-to-image decoding
    protocol VAEModel {
        func decode(latent: MLMultiArray) async throws -> MLMultiArray
    }

    // MARK: - FP32 Transformer Wrapper

    /// FP32 SANA transformer with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32Transformer: TransformerModel {
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            self.model = try MLModel(contentsOf: modelURL, configuration: configuration)

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in transformer model"])
            }
            self.outputKey = firstOutput.key
        }

        func predict(
            latent: MLMultiArray,
            timestep: MLMultiArray,
            encoderHiddenStates: MLMultiArray,
            encoderAttentionMask: MLMultiArray
        ) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent),
                "timestep": MLFeatureValue(multiArray: timestep),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
                "encoder_attention_mask": MLFeatureValue(multiArray: encoderAttentionMask)
            ])

            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from transformer"])
            }

            return result
        }
    }

    // MARK: - FP32 VAE Wrapper

    /// FP32 SANA VAE decoder with dynamic output key detection.
    /// Loads the model directly from a compiled `.mlmodelc` URL.
    class FP32VAE: VAEModel {
        private let model: MLModel
        private let outputKey: String

        init(modelURL: URL, configuration: MLModelConfiguration) throws {
            self.model = try MLModel(contentsOf: modelURL, configuration: configuration)

            let outputs = model.modelDescription.outputDescriptionsByName
            guard let firstOutput = outputs.first else {
                throw NSError(domain: "MobileOGenerator", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No output found in vae_decoder model"])
            }
            self.outputKey = firstOutput.key
        }

        func decode(latent: MLMultiArray) async throws -> MLMultiArray {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "latent": MLFeatureValue(multiArray: latent)
            ])
            let output = try await model.prediction(from: input)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                throw NSError(domain: "MobileOGenerator", code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to get output '\(outputKey)' from VAE decoder"])
            }

            return result
        }
    }

    // MARK: - Model Factory

    /// Factory for creating model wrappers based on variant
    struct ModelFactory {

        static func createTransformer(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> TransformerModel {
            let modelURL = modelDirectory.appendingPathComponent(variant.fileName)
            switch variant {
            case .fp32:
                return try FP32Transformer(modelURL: modelURL, configuration: configuration)
            }
        }

        static func createVAE(
            variant: ModelVariant,
            configuration: MLModelConfiguration,
            modelDirectory: URL
        ) throws -> VAEModel {
            let modelURL = modelDirectory.appendingPathComponent(variant.vaeFileName)
            switch variant {
            case .fp32:
                return try FP32VAE(modelURL: modelURL, configuration: configuration)
            }
        }
    }
}
