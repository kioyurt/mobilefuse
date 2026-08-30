import SwiftUI

/// Full-screen permission view shown before downloading models.
/// Displays download size, network status, and cellular warning.
struct DownloadPermissionView: View {
    let downloadManager: ModelDownloadManager
    let networkMonitor: NetworkMonitor

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            // Icon
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 56))
                .foregroundStyle(.purple.opacity(0.7))
                .padding(.bottom, 20)

            // Title
            Text("Set Up Mobile-O")
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .multilineTextAlignment(.center)
                .padding(.bottom, 8)

            Text("Mobile-O runs entirely on your device for\nmaximum privacy and speed. A one-time download\nis needed to get the AI models — after that,\neverything works fully offline.")
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(2)
                .padding(.horizontal, 24)
                .padding(.bottom, 28)

            // Info card
            VStack(spacing: 16) {
                infoRow(icon: "arrow.down.circle.fill", color: .purple,
                        title: "One-Time Download", detail: "~3.6 GB")
                infoRow(icon: "internaldrive.fill", color: .blue,
                        title: "Storage Required", detail: "~4 GB")
                infoRow(icon: "iphone.badge.checkmark", color: .green,
                        title: "After Setup", detail: "Fully offline")

                Divider()

                networkStatusRow
            }
            .padding(20)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.05), radius: 8, y: 3)
            )
            .padding(.horizontal, 24)

            // Cellular warning
            if networkMonitor.isExpensive {
                cellularWarning
                    .padding(.top, 12)
                    .padding(.horizontal, 24)
            }

            Spacer()

            // Download button
            Button(action: { downloadManager.startDownload() }) {
                Text("Download Models")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(networkMonitor.isConnected ? .purple : .gray)
                    )
            }
            .disabled(!networkMonitor.isConnected)
            .padding(.horizontal, 32)
            .padding(.bottom, 16)

            if !networkMonitor.isConnected {
                Text("No internet connection")
                    .font(.system(size: 13))
                    .foregroundStyle(.red)
                    .padding(.bottom, 8)
            }

            Spacer().frame(height: 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }

    // MARK: - Components

    private func infoRow(icon: String, color: Color, title: String, detail: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundStyle(color)
                .frame(width: 28)

            Text(title)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.primary)

            Spacer()

            Text(detail)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
        }
    }

    private var networkStatusRow: some View {
        HStack(spacing: 12) {
            Image(systemName: networkIcon)
                .font(.system(size: 20))
                .foregroundStyle(networkColor)
                .frame(width: 28)

            Text("Network")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.primary)

            Spacer()

            Text(networkLabel)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(networkColor)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(networkColor.opacity(0.12))
                )
        }
    }

    private var cellularWarning: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 16))
                .foregroundStyle(.orange)

            Text("You're on cellular data. This download is ~3.6 GB. WiFi is recommended.")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.orange.opacity(0.08))
                .strokeBorder(.orange.opacity(0.2), lineWidth: 1)
        )
    }

    // MARK: - Helpers

    private var networkIcon: String {
        if !networkMonitor.isConnected { return "wifi.slash" }
        switch networkMonitor.connectionType {
        case .wifi: return "wifi"
        case .cellular: return "antenna.radiowaves.left.and.right"
        default: return "network"
        }
    }

    private var networkColor: Color {
        if !networkMonitor.isConnected { return .red }
        return networkMonitor.isExpensive ? .orange : .green
    }

    private var networkLabel: String {
        if !networkMonitor.isConnected { return "Disconnected" }
        switch networkMonitor.connectionType {
        case .wifi: return "WiFi"
        case .cellular: return "Cellular"
        default: return "Connected"
        }
    }
}
