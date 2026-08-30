import Foundation
import CoreML
import Accelerate

/// DPM-Solver++ multistep scheduler for flow-matching diffusion.
///
/// Supports first and second order steps with precomputed coefficients.
/// Uses Accelerate (vDSP) for vectorized array operations.
class DPMSolverScheduler: DiffusionScheduler {

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
        let apply_flow_shift: Bool?
        let lambda_min_clipped: Float?
        let timestep_spacing: String?
    }

    let config: Config
    private(set) var timesteps: [Float] = []
    private(set) var sigmas: [Float] = []
    private var modelOutputs: [MLMultiArray] = []
    private var stepIndex: Int = 0

    // Precomputed step coefficients (calculated once in setTimesteps)
    private var precomputedDt: [Float] = []
    private var precomputedH: [Float] = []
    private var precomputedR: [Float] = []
    private var precomputedHalfR: [Float] = []

    // Reusable buffers for vectorized operations
    private var tempBuffer1: [Float] = []
    private var tempBuffer2: [Float] = []
    private var bufferSize: Int = 0
    private var cachedResultArray: MLMultiArray?

    init(config: Config) {
        self.config = config
    }

    /// Load scheduler from config file
    static func fromPretrained(path: String) throws -> DPMSolverScheduler {
        let configURL = URL(fileURLWithPath: path).appendingPathComponent("scheduler_config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Config.self, from: data)
        return DPMSolverScheduler(config: config)
    }

    /// Load scheduler from direct config file path
    static func fromConfigFile(path: String) throws -> DPMSolverScheduler {
        let configURL = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Config.self, from: data)
        return DPMSolverScheduler(config: config)
    }

    /// Configure timesteps and sigmas for the given number of inference steps
    func setTimesteps(numInferenceSteps: Int) {
        let numTrainTimesteps = config.num_train_timesteps

        timesteps = (0..<numInferenceSteps).map { i in
            let progress = Float(i) / Float(numInferenceSteps - 1)
            return Float(numTrainTimesteps - 1) * (1.0 - progress)
        }

        if config.use_flow_sigmas {
            sigmas = timesteps.map { t in t / Float(numTrainTimesteps) }

            // Apply flow_shift transformation if configured
            let shouldApplyShift = config.apply_flow_shift ?? false
            if shouldApplyShift && config.flow_shift != 1.0 {
                let shift = config.flow_shift
                sigmas = sigmas.map { sigma in
                    // Transform: σ' = shift·σ / (1 + (shift-1)·σ)
                    return shift * sigma / (1.0 + (shift - 1.0) * sigma)
                }
            }

            sigmas.append(0.0)
        } else {
            fatalError("Non-flow sigmas not implemented")
        }

        precomputeStepCoefficients(numSteps: numInferenceSteps)

        modelOutputs.removeAll()
        stepIndex = 0
        tempBuffer1.removeAll(keepingCapacity: true)
        tempBuffer2.removeAll(keepingCapacity: true)
        bufferSize = 0
    }

    /// Precompute dt, h, r, and halfR for each step to avoid per-step arithmetic
    private func precomputeStepCoefficients(numSteps: Int) {
        precomputedDt = []
        precomputedH = []
        precomputedR = []
        precomputedHalfR = []

        precomputedDt.reserveCapacity(numSteps)
        precomputedH.reserveCapacity(numSteps)
        precomputedR.reserveCapacity(numSteps)
        precomputedHalfR.reserveCapacity(numSteps)

        for i in 0..<numSteps {
            let sigma_t = sigmas[i]
            let sigma_next = (i < sigmas.count - 1) ? sigmas[i + 1] : 0.0
            let dt = sigma_next - sigma_t
            precomputedDt.append(dt)

            if i > 0 {
                let sigma_prev = sigmas[i - 1]
                let h = sigma_t - sigma_prev
                let h_next = sigma_next - sigma_t
                let r = h_next / max(abs(h), 1e-8)
                let r_clamped = max(-5.0, min(5.0, r))
                precomputedH.append(h)
                precomputedR.append(r_clamped)
                precomputedHalfR.append(0.5 * r_clamped)
            } else {
                precomputedH.append(0.0)
                precomputedR.append(0.0)
                precomputedHalfR.append(0.0)
            }
        }
    }

    /// Perform one denoising step, choosing first or second order based on history
    func step(modelOutput: MLMultiArray, timestep: Float, sample: MLMultiArray) throws -> MLMultiArray {
        defer { stepIndex += 1 }

        modelOutputs.append(modelOutput)
        if modelOutputs.count > config.solver_order {
            modelOutputs.removeFirst()
        }

        let tIndex = stepIndex
        let order = min(config.solver_order, modelOutputs.count)
        let useLowerOrderFinal = config.lower_order_final && (tIndex >= timesteps.count - 2)
        let actualOrder = useLowerOrderFinal ? 1 : order

        if actualOrder == 1 || modelOutputs.count < 2 {
            return try firstOrderStep(sample: sample, modelOutput: modelOutput, tIndex: tIndex)
        } else {
            return try secondOrderStep(sample: sample, tIndex: tIndex)
        }
    }

    /// First-order Euler step: x_{t+1} = x_t + dt * v_t
    private func firstOrderStep(sample: MLMultiArray, modelOutput: MLMultiArray,
                               tIndex: Int) throws -> MLMultiArray {
        var dt = precomputedDt[tIndex]
        let count = sample.shape.map { $0.intValue }.reduce(1, *)

        if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
            cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
        }
        let result = cachedResultArray!

        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            modelOutput.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { outputPtr in
                result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                    vDSP_vsma(outputPtr, 1, &dt, samplePtr, 1, resultPtr, 1, vDSP_Length(count))
                }
            }
        }

        return result
    }

    /// Second-order multistep using linear extrapolation with precomputed coefficients
    private func secondOrderStep(sample: MLMultiArray, tIndex: Int) throws -> MLMultiArray {
        guard modelOutputs.count >= 2 else {
            fatalError("Need at least 2 model outputs for second-order")
        }

        let modelOutput_curr = modelOutputs[modelOutputs.count - 1]
        let modelOutput_prev = modelOutputs[modelOutputs.count - 2]

        var dt = precomputedDt[tIndex]
        var halfR = precomputedHalfR[tIndex]
        let count = sample.shape.map { $0.intValue }.reduce(1, *)
        if cachedResultArray == nil || cachedResultArray!.shape != sample.shape {
            cachedResultArray = try MLMultiArray(shape: sample.shape, dataType: .float32)
        }
        let result = cachedResultArray!

        if bufferSize != count {
            tempBuffer1 = [Float](repeating: 0, count: count)
            tempBuffer2 = [Float](repeating: 0, count: count)
            bufferSize = count
        }

        // D = v_curr + 0.5*r*(v_curr - v_prev), then x_next = x_t + dt * D
        sample.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { samplePtr in
            modelOutput_curr.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { currPtr in
                modelOutput_prev.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { prevPtr in
                    result.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { resultPtr in
                        vDSP_vsub(prevPtr, 1, currPtr, 1, &tempBuffer1, 1, vDSP_Length(count))
                        vDSP_vsma(&tempBuffer1, 1, &halfR, currPtr, 1, &tempBuffer2, 1, vDSP_Length(count))
                        vDSP_vsma(&tempBuffer2, 1, &dt, samplePtr, 1, resultPtr, 1, vDSP_Length(count))
                    }
                }
            }
        }

        return result
    }
}
