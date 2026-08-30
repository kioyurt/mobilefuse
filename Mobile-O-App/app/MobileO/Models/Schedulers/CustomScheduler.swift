import Foundation
import CoreML
import Accelerate

/// Adams-Bashforth-3 scheduler with progressive order startup and update clamping.
///
/// Order ramps up: Euler (steps 1-2) → AB-2 (step 3) → AB-3 (step 4+).
/// Update clamping limits step magnitudes to prevent artifacts.
class CustomScheduler: DiffusionScheduler {

    struct Config: Codable {
        let num_train_timesteps: Int
        let beta_start: Float
        let beta_end: Float
        let beta_schedule: String
        let solver_order: Int
        let prediction_type: String
        let algorithm_type: String
        let use_flow_sigmas: Bool
        let flow_shift: Float
        let lower_order_final: Bool

        let lambda_min_clipped: Float?
        let timestep_spacing: String?

        enum CodingKeys: String, CodingKey {
            case num_train_timesteps, beta_start, beta_end, beta_schedule
            case solver_order, prediction_type, algorithm_type
            case use_flow_sigmas, flow_shift, lower_order_final
            case lambda_min_clipped, timestep_spacing
        }
    }

    let config: Config
    private(set) var timesteps: [Float] = []
    private(set) var sigmas: [Float] = []

    private var velocities: [MLMultiArray] = []
    private var stepIndex: Int = 0
    private var updateMagnitudes: [Float] = []

    /// Clamp updates exceeding this factor times the median magnitude
    private let updateClampFactor: Float = 2.0

    // Reusable buffers for vectorized operations
    private var tempBuffer1: [Float] = []
    private var tempBuffer2: [Float] = []
    private var tempBuffer3: [Float] = []
    private var bufferSize: Int = 0
    private var cachedResultArray: MLMultiArray?

    init(config: Config) {
        self.config = config
    }

    static func fromPretrained(path: String) throws -> CustomScheduler {
        let configURL = URL(fileURLWithPath: path).appendingPathComponent("scheduler_config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Config.self, from: data)
        return CustomScheduler(config: config)
    }

    static func fromConfigFile(path: String) throws -> CustomScheduler {
        let configURL = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Config.self, from: data)
        return CustomScheduler(config: config)
    }

    /// Set timesteps with linear schedule matching the DPM solver approach
    func setTimesteps(numInferenceSteps: Int) {
        let numTrainTimesteps = config.num_train_timesteps

        timesteps = (0..<numInferenceSteps).map { i in
            let progress = Float(i) / Float(numInferenceSteps - 1)
            return Float(numTrainTimesteps - 1) * (1.0 - progress)
        }

        if config.use_flow_sigmas {
            sigmas = timesteps.map { t in t / Float(numTrainTimesteps) }
            sigmas.append(0.0)
        } else {
            fatalError("Non-flow sigmas not implemented")
        }

        velocities.removeAll()
        updateMagnitudes.removeAll()
        stepIndex = 0
        tempBuffer1.removeAll(keepingCapacity: true)
        tempBuffer2.removeAll(keepingCapacity: true)
        tempBuffer3.removeAll(keepingCapacity: true)
        bufferSize = 0
    }

    /// Perform one denoising step using progressive Adams-Bashforth integration
    func step(modelOutput: MLMultiArray, timestep: Float, sample: MLMultiArray) throws -> MLMultiArray {
        defer { stepIndex += 1 }

        let velocity = try convertToVelocity(modelOutput: modelOutput, timestep: timestep, sample: sample)

        velocities.append(velocity)
        if velocities.count > 4 {
            velocities.removeFirst()
        }

        // Progressive order: Euler -> AB-2 -> AB-3
        let result: MLMultiArray

        if velocities.count <= 2 {
            result = try eulerStep(sample: sample, velocity: velocity, tIndex: stepIndex)
        } else if velocities.count == 3 {
            let (stepped, _) = try variableStepAB2(sample: sample, tIndex: stepIndex)
            result = stepped
        } else {
            // Fourth+ step: AB-3 (full power)
            let (stepped, _) = try variableStepAB3WithErrorEstimation(sample: sample, tIndex: stepIndex)
            result = stepped
        }

        return try clampUpdateMagnitude(sample: sample, result: result)
    }

    // MARK: - Prediction Type Conversion

    /// Convert model output to velocity based on `prediction_type` config
    private func convertToVelocity(modelOutput: MLMultiArray, timestep: Float, sample: MLMultiArray) throws -> MLMultiArray {
        let predType = config.prediction_type.lowercased()

        if predType == "velocity" || predType == "v_prediction" || predType == "flow_prediction" {
            return modelOutput
        }

        // v = (x - epsilon) / (1 - sigma)
        if predType == "epsilon" {
            let sigma = timestep / Float(config.num_train_timesteps)
            let oneMinusSigma = max(1.0 - sigma, 1e-8)

            let count = sample.shape.map { $0.intValue }.reduce(1, *)

            if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
                cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
            }
            let result = cachedResultArray!

            sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
                modelOutput.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { epsilonPtr in
                    result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                        var divisor = oneMinusSigma
                        vDSP_vsub(epsilonPtr, 1, samplePtr, 1, resultPtr, 1, vDSP_Length(count))
                        vDSP_vsdiv(resultPtr, 1, &divisor, resultPtr, 1, vDSP_Length(count))
                    }
                }
            }

            return result
        }

        // v = (x_0 - x_t) / sigma
        if predType == "sample" {
            let sigma = max(timestep / Float(config.num_train_timesteps), 1e-8)
            let count = sample.shape.map { $0.intValue }.reduce(1, *)

            if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
                cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
            }
            let result = cachedResultArray!

            sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
                modelOutput.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { x0Ptr in
                    result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                        vDSP_vsub(samplePtr, 1, x0Ptr, 1, resultPtr, 1, vDSP_Length(count))
                        var sigmaInv = 1.0 / sigma
                        vDSP_vsdiv(resultPtr, 1, &sigmaInv, resultPtr, 1, vDSP_Length(count))
                    }
                }
            }

            return result
        }

        return modelOutput
    }

    // MARK: - Integration Methods

    /// Euler step: x_{n+1} = x_n + h·v_n
    private func eulerStep(sample: MLMultiArray, velocity: MLMultiArray, tIndex: Int) throws -> MLMultiArray {
        let sigma_t = sigmas[tIndex]
        let sigma_next = (tIndex < sigmas.count - 1) ? sigmas[tIndex + 1] : 0.0
        var h = sigma_next - sigma_t

        let count = sample.shape.map { $0.intValue }.reduce(1, *)

        if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
            cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
        }
        let result = cachedResultArray!

        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            velocity.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { velPtr in
                result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                    vDSP_vsma(velPtr, 1, &h, samplePtr, 1, resultPtr, 1, vDSP_Length(count))
                }
            }
        }

        return result
    }

    /// Adams-Bashforth-2 (constant coefficients - robust for moderate non-uniformity)
    /// AB-2: x_{n+1} = x_n + h·(3/2·v_n - 1/2·v_{n-1})
    private func variableStepAB2(sample: MLMultiArray, tIndex: Int) throws -> (MLMultiArray, Float) {
        guard velocities.count >= 2 else {
            return try (eulerStep(sample: sample, velocity: velocities.last!, tIndex: tIndex), Float.nan)
        }

        let v_curr = velocities[velocities.count - 1]
        let v_prev = velocities[velocities.count - 2]

        let sigma_t = sigmas[tIndex]
        let sigma_next = (tIndex < sigmas.count - 1) ? sigmas[tIndex + 1] : 0.0

        let h_n = sigma_next - sigma_t

        let alpha: Float = 3.0 / 2.0
        let beta: Float = -1.0 / 2.0

        let count = sample.shape.map { $0.intValue }.reduce(1, *)

        if bufferSize != count {
            tempBuffer1 = [Float](repeating: 0, count: count)
            bufferSize = count
        }

        if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
            cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
        }
        let result = cachedResultArray!

        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            v_curr.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { currPtr in
                v_prev.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { prevPtr in
                    result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                        var alphaVar = alpha
                        var betaVar = beta
                        var hVar = h_n

                        vDSP_vsmul(currPtr, 1, &alphaVar, &tempBuffer1, 1, vDSP_Length(count))
                        vDSP_vsma(prevPtr, 1, &betaVar, &tempBuffer1, 1, &tempBuffer1, 1, vDSP_Length(count))
                        vDSP_vsma(&tempBuffer1, 1, &hVar, samplePtr, 1, resultPtr, 1, vDSP_Length(count))
                    }
                }
            }
        }

        return (result, Float.nan)
    }

    /// Adams-Bashforth-3 with embedded error estimation (constant coefficients)
    /// AB-3: x_{n+1} = x_n + h·(23/12·v_n - 16/12·v_{n-1} + 5/12·v_{n-2})
    /// Error estimate: ||x_AB3 - x_AB2|| / ||x_AB3||
    private func variableStepAB3WithErrorEstimation(sample: MLMultiArray, tIndex: Int) throws -> (MLMultiArray, Float) {
        guard velocities.count >= 3 else {
            return try variableStepAB2(sample: sample, tIndex: tIndex)
        }

        let v_curr = velocities[velocities.count - 1]
        let v_prev1 = velocities[velocities.count - 2]
        let v_prev2 = velocities[velocities.count - 3]

        let sigma_t = sigmas[tIndex]
        let sigma_next = (tIndex < sigmas.count - 1) ? sigmas[tIndex + 1] : 0.0

        let h_n = sigma_next - sigma_t

        let alpha3: Float = 23.0 / 12.0
        let beta3: Float = -16.0 / 12.0
        let gamma3: Float = 5.0 / 12.0

        // AB-2 coefficients for error estimation
        let alpha2: Float = 3.0 / 2.0
        let beta2: Float = -1.0 / 2.0

        let count = sample.shape.map { $0.intValue }.reduce(1, *)

        if bufferSize != count || tempBuffer2.isEmpty || tempBuffer3.isEmpty {
            tempBuffer1 = [Float](repeating: 0, count: count)
            tempBuffer2 = [Float](repeating: 0, count: count)
            tempBuffer3 = [Float](repeating: 0, count: count)
            bufferSize = count
        }

        if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
            cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
        }
        let result = cachedResultArray!

        var estimatedError: Float = 0.0

        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            v_curr.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { currPtr in
                v_prev1.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { prev1Ptr in
                    v_prev2.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { prev2Ptr in
                        result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                            var alpha3Var = alpha3
                            var beta3Var = beta3
                            var gamma3Var = gamma3
                            var hVar = h_n

                            // AB-3 velocity combination
                            vDSP_vsmul(currPtr, 1, &alpha3Var, &tempBuffer1, 1, vDSP_Length(count))
                            vDSP_vsma(prev1Ptr, 1, &beta3Var, &tempBuffer1, 1, &tempBuffer1, 1, vDSP_Length(count))
                            vDSP_vsma(prev2Ptr, 1, &gamma3Var, &tempBuffer1, 1, &tempBuffer1, 1, vDSP_Length(count))
                            vDSP_vsma(&tempBuffer1, 1, &hVar, samplePtr, 1, resultPtr, 1, vDSP_Length(count))

                            // AB-2 for error estimation
                            var alpha2Var = alpha2
                            var beta2Var = beta2
                            vDSP_vsmul(currPtr, 1, &alpha2Var, &tempBuffer2, 1, vDSP_Length(count))
                            vDSP_vsma(prev1Ptr, 1, &beta2Var, &tempBuffer2, 1, &tempBuffer2, 1, vDSP_Length(count))
                            vDSP_vsma(&tempBuffer2, 1, &hVar, samplePtr, 1, &tempBuffer3, 1, vDSP_Length(count))

                            // Relative error: ||x_AB3 - x_AB2|| / ||x_AB3||
                            var diff_norm: Float = 0.0
                            var result_norm: Float = 0.0

                            vDSP_vsub(&tempBuffer3, 1, resultPtr, 1, &tempBuffer2, 1, vDSP_Length(count))
                            vDSP_svesq(&tempBuffer2, 1, &diff_norm, vDSP_Length(count))
                            vDSP_svesq(resultPtr, 1, &result_norm, vDSP_Length(count))

                            diff_norm = sqrt(diff_norm)
                            result_norm = sqrt(result_norm)

                            estimatedError = diff_norm / max(result_norm, 1e-8)
                        }
                    }
                }
            }
        }

        return (result, estimatedError)
    }

    // MARK: - Update Clamping

    /// Clamp update magnitude: if infinity-norm exceeds `updateClampFactor * median`, scale down
    private func clampUpdateMagnitude(sample: MLMultiArray, result: MLMultiArray) throws -> MLMultiArray {
        let count = sample.shape.map { $0.intValue }.reduce(1, *)

        var updateMagnitude: Float = 0.0
        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                if tempBuffer1.count != count {
                    tempBuffer1 = [Float](repeating: 0, count: count)
                }
                vDSP_vsub(samplePtr, 1, resultPtr, 1, &tempBuffer1, 1, vDSP_Length(count))
                vDSP_maxmgv(&tempBuffer1, 1, &updateMagnitude, vDSP_Length(count))
            }
        }

        updateMagnitudes.append(updateMagnitude)
        if updateMagnitudes.count > 10 { updateMagnitudes.removeFirst() }

        guard updateMagnitudes.count >= 3 else { return result }

        let sortedMagnitudes = updateMagnitudes.sorted()
        let median = sortedMagnitudes[sortedMagnitudes.count / 2]

        let threshold = updateClampFactor * median
        if updateMagnitude > threshold && median > 1e-6 {
            let scale = threshold / updateMagnitude
            if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
                cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
            }
            let clampedResult = cachedResultArray!

            sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
                result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                    clampedResult.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { clampedPtr in
                        var scaleVar = scale
                        vDSP_vsub(samplePtr, 1, resultPtr, 1, &tempBuffer1, 1, vDSP_Length(count))
                        vDSP_vsma(&tempBuffer1, 1, &scaleVar, samplePtr, 1, clampedPtr, 1, vDSP_Length(count))
                    }
                }
            }

            return clampedResult
        }

        return result
    }
}
