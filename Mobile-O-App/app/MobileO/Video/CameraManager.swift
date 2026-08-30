import AVFoundation
import CoreImage

import UIKit

/// Simplified camera manager that uses native preview and captures frames on-demand
@Observable
@MainActor
class CameraManager: NSObject {

    // Public properties
    private(set) var session = AVCaptureSession()
    private(set) var isCameraAvailable = true

    private(set) var isRunning = false
    private(set) var isPaused = false

    var useBackCamera = true {
        didSet {
            if useBackCamera != oldValue {
                Task { await restartCamera() }
            }
        }
    }

    // Private properties
    private var videoOutput: AVCaptureVideoDataOutput?
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var currentFrameContinuation: CheckedContinuation<CVImageBuffer?, Never>?

    // MARK: - Session Management

    func start() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async { [weak self] in
                guard let self = self else {
                    continuation.resume()
                    return
                }

                self.checkPermission { granted in
                    guard granted else {
                            continuation.resume()
                        return
                    }

                    self.setupCaptureSession()
                    self.session.startRunning()

                    Task { @MainActor in
                        self.isRunning = true
                        self.isPaused = false
                        continuation.resume()
                    }
                }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.session.stopRunning()

            Task { @MainActor in
                self.isRunning = false
                self.isPaused = false
            }
        }
    }

    func pause() {
        sessionQueue.async { [weak self] in
            guard let self = self, self.isRunning else { return }
            self.session.stopRunning()

            Task { @MainActor in
                self.isPaused = true
            }
        }
    }

    func resume() {
        sessionQueue.async { [weak self] in
            guard let self = self, self.isPaused else { return }
            self.session.startRunning()

            Task { @MainActor in
                self.isPaused = false
            }
        }
    }

    private func restartCamera() async {
        stop()
        try? await Task.sleep(for: .milliseconds(100))
        await start()
    }

    // MARK: - Frame Capture

    /// Wait for camera to be ready and capture the first valid frame
    func waitForFirstFrame() async -> CVImageBuffer? {
        try? await Task.sleep(for: .milliseconds(300))

        for _ in 0..<10 {
            if let frame = await captureFrame(),
               CVPixelBufferGetWidth(frame) > 0,
               CVPixelBufferGetHeight(frame) > 0 {
                return frame
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        return nil
    }

    /// Capture a single frame from the camera on-demand
    func captureFrame() async -> CVImageBuffer? {
        return await withCheckedContinuation { continuation in
            sessionQueue.async { [weak self] in
                self?.currentFrameContinuation = continuation
            }
        }
    }

    // MARK: - Setup

    private func checkPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video, completionHandler: completion)
        default:
            completion(false)
        }
    }

    private func setupCaptureSession() {
        session.beginConfiguration()

        // Remove existing inputs/outputs
        for input in session.inputs {
            session.removeInput(input)
        }
        for output in session.outputs {
            session.removeOutput(output)
        }

        // Setup camera input
        let position: AVCaptureDevice.Position = useBackCamera ? .back : .front
        guard let device = getCameraDevice(position: position),
              let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        // Setup video output for frame capture
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.queue"))
        output.alwaysDiscardsLateVideoFrames = true

        if session.canAddOutput(output) {
            session.addOutput(output)
            videoOutput = output
        }

        // Set preset
        if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        }

        session.commitConfiguration()

        // Setup rotation
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }
    }

    private func getCameraDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video,
            position: position
        ).devices.first
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let imageBuffer = sampleBuffer.imageBuffer else { return }

        // Only capture if someone is waiting for a frame
        sessionQueue.async { [weak self] in
            if let continuation = self?.currentFrameContinuation {
                self?.currentFrameContinuation = nil
                continuation.resume(returning: imageBuffer)
            }
        }
    }
}
