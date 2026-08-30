import SwiftUI

/// Progress screen showing per-file and overall download progress, speed, ETA,
/// compilation phase, and completion state.
struct DownloadProgressView: View {
    @Bindable var downloadManager: ModelDownloadManager

    private let components = [
        ("connector", "Connector", "link"),
        ("transformer", "Transformer", "cpu"),
        ("vae_decoder", "VAE Decoder", "wand.and.rays"),
        ("vision_encoder", "Vision Encoder", "eye"),
        ("llm", "Language Model", "text.bubble")
    ]

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            if downloadManager.state == .completed {
                completionView
            } else if downloadManager.state == .compiling {
                compilingView
            } else {
                downloadingView
            }

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }

    // MARK: - Downloading View

    private var downloadingView: some View {
        VStack(spacing: 24) {
            // Progress ring
            ZStack {
                Circle()
                    .stroke(.purple.opacity(0.15), lineWidth: 8)
                    .frame(width: 120, height: 120)
                Circle()
                    .trim(from: 0, to: downloadManager.overallProgress)
                    .stroke(.purple, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: downloadManager.overallProgress)

                Text("\(Int(downloadManager.overallProgress * 100))%")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(.purple)
            }

            // Status text
            VStack(spacing: 4) {
                Text(downloadManager.state == .paused ? "Paused" : "Downloading...")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                Text("\(formatBytes(downloadManager.downloadedBytes)) / \(formatBytes(downloadManager.totalBytes))")
                    .font(.system(size: 15, design: .rounded))
                    .foregroundStyle(.secondary)
            }

            // Component checklist
            componentList

            // Speed + ETA
            if downloadManager.state == .downloading {
                HStack(spacing: 16) {
                    Label(formatSpeed(downloadManager.downloadSpeed), systemImage: "speedometer")
                    if downloadManager.eta > 0 && downloadManager.eta < 86400 {
                        Label(formatETA(downloadManager.eta), systemImage: "clock")
                    }
                }
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            }

            // Pause/Resume button
            Button(action: {
                if downloadManager.state == .paused {
                    downloadManager.resume()
                } else {
                    downloadManager.pause()
                }
            }) {
                Label(
                    downloadManager.state == .paused ? "Resume" : "Pause",
                    systemImage: downloadManager.state == .paused ? "play.fill" : "pause.fill"
                )
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(.purple)
                .padding(.horizontal, 24)
                .padding(.vertical, 10)
                .background(
                    Capsule()
                        .fill(.purple.opacity(0.1))
                )
            }
        }
        .padding(.horizontal, 24)
    }

    // MARK: - Compiling View

    private var compilingView: some View {
        VStack(spacing: 24) {
            ProgressView()
                .scaleEffect(1.5)
                .tint(.purple)
                .padding(.bottom, 8)

            Text("Preparing Models")
                .font(.system(size: 20, weight: .semibold, design: .rounded))

            Text(downloadManager.compilationProgress)
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Text("This may take a few minutes...")
                .font(.system(size: 13))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 24)
    }

    // MARK: - Completion View

    private var completionView: some View {
        VStack(spacing: 24) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 64))
                .foregroundStyle(.green)
                .transition(.scale.combined(with: .opacity))

            Text("You're All Set!")
                .font(.system(size: 24, weight: .bold, design: .rounded))

            Text("All AI models are ready. Everything runs\non-device — no internet needed.")
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button(action: { downloadManager.checkModelsReady() }) {
                Text("Get Started")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(.purple)
                    )
            }
            .padding(.horizontal, 32)
        }
    }

    // MARK: - Component List

    private var componentList: some View {
        VStack(spacing: 8) {
            ForEach(components, id: \.0) { (id, name, icon) in
                HStack(spacing: 12) {
                    Image(systemName: icon)
                        .font(.system(size: 14))
                        .foregroundStyle(.secondary)
                        .frame(width: 24)

                    Text(name)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.primary)

                    Spacer()

                    if downloadManager.completedComponents.contains(id) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 16))
                            .foregroundStyle(.green)
                    } else if downloadManager.currentFileName.hasPrefix(id) {
                        ProgressView()
                            .scaleEffect(0.7)
                    } else {
                        Image(systemName: "circle")
                            .font(.system(size: 16))
                            .foregroundStyle(.quaternary)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
        }
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.regularMaterial)
                .shadow(color: .black.opacity(0.03), radius: 4, y: 2)
        )
        .padding(.horizontal, 8)
    }

    // MARK: - Formatting

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_000_000_000
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / 1_000_000
        return String(format: "%.0f MB", mb)
    }

    private func formatSpeed(_ bytesPerSec: Double) -> String {
        let mbps = bytesPerSec / 1_000_000
        if mbps >= 1 { return String(format: "%.1f MB/s", mbps) }
        let kbps = bytesPerSec / 1_000
        return String(format: "%.0f KB/s", kbps)
    }

    private func formatETA(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        if mins > 0 { return "\(mins)m \(secs)s left" }
        return "\(secs)s left"
    }
}
