import AVFoundation
import SwiftUI
import CoreImage
import UIKit

/// Full-screen camera view with real-time or single-shot image understanding.
///
/// Supports two modes:
/// - **Live** (`continuous`) — captures and analyzes frames in a loop.
/// - **Photo** (`single`) — freezes a frame, pauses the camera, and analyzes once.
///
/// A toolbar menu lets the user pick prompt presets or write a custom prompt.
struct CameraUnderstandingView: View {
    @State private var cameraManager = CameraManager()
    @Binding var understandingModel: ImageUnderstandingModel?

    @State private var prompt = "Describe the image in English."
    @State private var promptSuffix = "Output should be brief, about 15 words or less."

    @State private var selectedCameraType: CameraType = .continuous
    @State private var evaluationState: EvaluationState = .idle
    @State private var selectedPreset: PromptPreset? = .describe
    @State private var isEditingPrompt = false

    @State private var displayedResponse = ""
    @State private var displayedMetrics: ResponseMetrics?

    @State private var frozenFrame: PlatformImage?
    @State private var continuousAnalysisTask: Task<Void, Never>?

    // MARK: - Body

    var body: some View {
        GeometryReader { mainGeometry in
            ZStack {
                cameraLayer
                if selectedCameraType == .continuous { statusIndicator }
                bottomControls(safeAreaBottom: mainGeometry.safeAreaInsets.bottom)
                responseSheet
            }
        }
        .background(Color.black)
        .animation(.spring(response: 0.35, dampingFraction: 0.75), value: selectedCameraType)
        .animation(.spring(response: 0.35, dampingFraction: 0.75), value: frozenFrame != nil)
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: displayedResponse)
        .animation(.spring(response: 0.35, dampingFraction: 0.7), value: displayedMetrics != nil)
        .task { await startCamera() }
        .onDisappear {
            continuousAnalysisTask?.cancel()
            cameraManager.stop()
        }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .navigationTitle("Live Camera")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { promptToolbar }
        .sheet(isPresented: $isEditingPrompt) { promptEditor }
    }

    // MARK: - Camera Layer

    @ViewBuilder
    private var cameraLayer: some View {
        if frozenFrame == nil {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()
                .transition(.opacity)
        } else if let frame = frozenFrame, frame.size.width > 0 && frame.size.height > 0 {
            Image(uiImage: frame)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .ignoresSafeArea()
                .transition(.opacity)
        }
    }

    // MARK: - Status Indicator

    private var statusIndicator: some View {
        HStack(spacing: 6) {
            Group {
                switch evaluationState {
                case .processingPrompt:
                    ProgressView().tint(.white).controlSize(.small)
                case .generatingResponse:
                    Image(systemName: "sparkles").font(.caption2)
                case .idle:
                    Image(systemName: "eye.fill").font(.caption2)
                }
            }
            Text(evaluationState.rawValue)
                .font(.caption)
                .fontWeight(.semibold)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.black.opacity(0.5))
        .clipShape(Capsule())
        .transition(.scale(scale: 0.8).combined(with: .opacity))
        .animation(.spring(response: 0.35, dampingFraction: 0.7), value: evaluationState)
    }

    // MARK: - Bottom Controls

    private func bottomControls(safeAreaBottom: CGFloat) -> some View {
        VStack(spacing: 0) {
            Spacer()
            HStack(spacing: 12) {
                modeToggle
                Spacer()
                flipButton
            }
            .padding(.horizontal, 16)
            .padding(.bottom, max(safeAreaBottom + 8, 16))
        }
    }

    private var modeToggle: some View {
        HStack(spacing: 2) {
            ForEach([CameraType.continuous, CameraType.single], id: \.self) { type in
                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    switchMode(to: type)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: type == .continuous ? "video.fill" : "camera.shutter.button.fill")
                            .font(.caption)
                        Text(type == .continuous ? "Live" : "Photo")
                            .font(.caption2)
                            .fontWeight(.medium)
                    }
                    .foregroundStyle(selectedCameraType == type ? .white : .white.opacity(0.6))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(selectedCameraType == type ? Color.white.opacity(0.2) : Color.clear)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
        }
        .background(.ultraThinMaterial)
        .cornerRadius(10)
    }

    private var flipButton: some View {
        Button {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            cameraManager.useBackCamera.toggle()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                    .font(.caption)
                Text("Flip")
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Response Sheet

    @ViewBuilder
    private var responseSheet: some View {
        if !displayedResponse.isEmpty || selectedCameraType == .single {
            VStack(spacing: 0) {
                Spacer()
                VStack(spacing: 0) {
                    if !displayedResponse.isEmpty {
                        responseContent
                        if selectedCameraType == .single {
                            Divider().padding(.horizontal, 20).transition(.opacity)
                        }
                    }
                    if selectedCameraType == .single {
                        CaptureButton(isFrozen: frozenFrame != nil, onTap: handleCaptureButton)
                            .padding(.horizontal, 20)
                            .padding(.vertical, displayedResponse.isEmpty ? 20 : 16)
                    }
                }
                .background(.ultraThinMaterial)
                .cornerRadius(24)
                .shadow(color: .black.opacity(0.25), radius: 20, y: -5)
                .padding(.horizontal, 16)
                .padding(.bottom, 90)
                .transition(.asymmetric(
                    insertion: .move(edge: .bottom).combined(with: .opacity),
                    removal: .move(edge: .bottom).combined(with: .opacity)
                ))
            }
        }
    }

    private var responseContent: some View {
        VStack(spacing: 12) {
            if let metrics = displayedMetrics {
                HStack(spacing: 10) {
                    MetricPill(label: "TTFT", value: String(format: "%.0fms", metrics.ttft * 1000))
                    MetricPill(label: "Total", value: String(format: "%.1fs", metrics.totalTime))
                    MetricPill(label: "Speed", value: String(format: "%.1f tok/s", metrics.tokensPerSecond))
                }
                .font(.caption2)
                .transition(.scale(scale: 0.9).combined(with: .opacity))
            }

            ScrollView {
                Text(displayedResponse)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 150)
        }
        .padding(.horizontal, 20)
        .padding(.top, 20)
        .padding(.bottom, selectedCameraType == .single ? 16 : 20)
        .transition(.opacity.combined(with: .move(edge: .top)))
    }

    // MARK: - Prompt Toolbar & Editor

    @ToolbarContentBuilder
    private var promptToolbar: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Menu {
                Section("Quick Prompts") {
                    ForEach(PromptPreset.allCases.filter { $0 != .custom }, id: \.self) { preset in
                        Button {
                            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                selectedPreset = preset
                                prompt = preset.prompt
                                promptSuffix = preset.suffix
                            }
                        } label: {
                            Label(preset.displayName, systemImage: preset.icon)
                        }
                    }
                }
                Divider()
                Button { isEditingPrompt = true } label: {
                    Label("Custom Prompt", systemImage: "pencil.circle")
                }
            } label: {
                Image(systemName: "text.bubble").imageScale(.large)
            }
        }
    }

    private var promptEditor: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("What should I analyze?", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                } header: {
                    Text("Main Prompt")
                } footer: {
                    Text("Describe what you want the camera to analyze")
                }

                Section {
                    TextField("Additional instructions...", text: $promptSuffix, axis: .vertical)
                        .lineLimit(1...3)
                } header: {
                    Text("Additional Instructions (Optional)")
                } footer: {
                    Text("Add constraints like output length or format")
                }

                Section("Quick Presets") {
                    Button {
                        prompt = ""
                        promptSuffix = ""
                        selectedPreset = .custom
                    } label: {
                        presetRow(icon: "pencil.circle", name: "Custom", isSelected: selectedPreset == .custom)
                    }

                    ForEach(PromptPreset.allCases.filter { $0 != .custom }, id: \.self) { preset in
                        Button {
                            prompt = preset.prompt
                            promptSuffix = preset.suffix
                            selectedPreset = preset
                        } label: {
                            presetRow(icon: preset.icon, name: preset.displayName, isSelected: selectedPreset == preset)
                        }
                    }
                }
            }
            .navigationTitle("Edit Prompt")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { isEditingPrompt = false }
                }
            }
        }
    }

    private func presetRow(icon: String, name: String, isSelected: Bool) -> some View {
        HStack {
            Image(systemName: icon)
            Text(name)
            Spacer()
            if isSelected {
                Image(systemName: "checkmark").foregroundStyle(.blue)
            }
        }
    }

    // MARK: - Camera Lifecycle

    private func startCamera() async {
        await cameraManager.start()
        if selectedCameraType == .continuous {
            _ = await cameraManager.waitForFirstFrame()
            startContinuousAnalysis()
        }
    }

    // MARK: - Mode Switching

    private func switchMode(to newMode: CameraType) {
        continuousAnalysisTask?.cancel()
        understandingModel?.cancel()

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            frozenFrame = nil
            if newMode == .continuous {
                displayedResponse = ""
                displayedMetrics = nil
            }
            selectedCameraType = newMode
        }

        if cameraManager.isPaused { cameraManager.resume() }

        if newMode == .continuous {
            Task {
                _ = await cameraManager.waitForFirstFrame()
                startContinuousAnalysis()
            }
        }
    }

    // MARK: - Continuous Analysis

    private func startContinuousAnalysis() {
        continuousAnalysisTask?.cancel()
        continuousAnalysisTask = Task {
            while !Task.isCancelled {
                await analyzeCurrentFrame()
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    // MARK: - Frame Capture & Analysis

    private func handleCaptureButton() {
        if frozenFrame != nil {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                frozenFrame = nil
                displayedResponse = ""
                displayedMetrics = nil
            }
            cameraManager.resume()
        } else {
            Task { await captureAndAnalyzeFrame() }
        }
    }

    private func captureAndAnalyzeFrame() async {
        guard let frame = await cameraManager.captureFrame(),
              CVPixelBufferGetWidth(frame) > 0, CVPixelBufferGetHeight(frame) > 0 else { return }

        let platformImage = convertToPlatformImage(frame)
        guard platformImage.size.width > 0 && platformImage.size.height > 0 else { return }

        cameraManager.pause()

        await MainActor.run {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                frozenFrame = platformImage
                displayedResponse = ""
                displayedMetrics = nil
            }
            evaluationState = .processingPrompt
        }

        await analyzeImage(platformImage)
    }

    private func analyzeCurrentFrame() async {
        guard let frame = await cameraManager.captureFrame(),
              CVPixelBufferGetWidth(frame) > 0, CVPixelBufferGetHeight(frame) > 0 else { return }

        let platformImage = convertToPlatformImage(frame)
        guard platformImage.size.width > 0 && platformImage.size.height > 0 else { return }

        await analyzeImage(platformImage)
    }

    private func analyzeImage(_ image: PlatformImage) async {
        guard let model = understandingModel else { return }

        await MainActor.run { evaluationState = .processingPrompt }

        _ = await model.understand(image: image, prompt: "\(prompt) \(promptSuffix)")

        await MainActor.run {
            displayedResponse = model.response
            if model.totalTime > 0 {
                displayedMetrics = ResponseMetrics(
                    ttft: model.timeToFirstToken,
                    totalTime: model.totalTime,
                    tokensPerSecond: Double(model.tokensGenerated) / model.totalTime
                )
            }
            evaluationState = .idle
        }
    }

    private func convertToPlatformImage(_ frame: CVImageBuffer) -> PlatformImage {
        let ciImage = CIImage(cvPixelBuffer: frame)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            return UIImage(cgImage: cgImage)
        }
        return UIImage()
    }
}

// MARK: - Metric Pill

/// Small label+value chip for displaying inference metrics.
struct MetricPill: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 3) {
            Text(label).foregroundStyle(.tertiary)
            Text(value).foregroundStyle(.secondary).fontWeight(.medium)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(Color.primary.opacity(0.05))
        .cornerRadius(6)
    }
}

// MARK: - Capture Button

/// Circular shutter button that toggles between capture and continue states.
struct CaptureButton: View {
    let isFrozen: Bool
    let onTap: () -> Void

    var body: some View {
        Button {
            UIImpactFeedbackGenerator(style: isFrozen ? .light : .medium).impactOccurred()
            onTap()
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle().stroke(.white, lineWidth: 3).frame(width: 60, height: 60)
                    if isFrozen {
                        Image(systemName: "play.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(.white)
                    } else {
                        Circle().fill(.white).frame(width: 50, height: 50)
                    }
                }
                Text(isFrozen ? "Continue" : "Capture")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Supporting Types

/// Camera capture mode: continuous loop or single-shot.
private enum CameraType {
    case continuous
    case single
}

/// Timing metrics for a single understanding response.
struct ResponseMetrics {
    let ttft: TimeInterval
    let totalTime: TimeInterval
    let tokensPerSecond: Double
}

/// Camera analysis state shown in the status indicator pill.
enum EvaluationState: String, CaseIterable {
    case idle = "Ready"
    case processingPrompt = "Analyzing"
    case generatingResponse = "Responding"
}

/// Built-in prompt templates for camera understanding.
enum PromptPreset: String, CaseIterable {
    case describe, expression, readText, countObjects, custom

    var displayName: String {
        switch self {
        case .describe: "Describe"
        case .expression: "Expression"
        case .readText: "Read Text"
        case .countObjects: "Count"
        case .custom: "Custom"
        }
    }

    var icon: String {
        switch self {
        case .describe: "eye"
        case .expression: "face.smiling"
        case .readText: "text.viewfinder"
        case .countObjects: "number.circle"
        case .custom: "pencil.circle"
        }
    }

    var prompt: String {
        switch self {
        case .describe: "Describe the image in English."
        case .expression: "What is this person's facial expression?"
        case .readText: "What is written in this image?"
        case .countObjects: "How many objects are in this image?"
        case .custom: ""
        }
    }

    var suffix: String {
        switch self {
        case .describe: "Output should be brief, about 15 words or less."
        case .expression: "Output only one or two words."
        case .readText: "Output only the text in the image."
        case .countObjects: "Output just the count."
        case .custom: ""
        }
    }
}
