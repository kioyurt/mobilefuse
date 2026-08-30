import SwiftUI

/// Settings screen for image generation parameters (scheduler, steps, CFG).
struct SettingsView: View {
    @Bindable var model: MobileOModel
    @Bindable var settings: SettingsViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                SettingsSection(title: "Scheduler", icon: "waveform.path.ecg") {
                    SchedulerPicker(model: model)
                }
                SettingsSection(title: "Generation Parameters", icon: "slider.horizontal.3") {
                    VStack(alignment: .leading, spacing: 14) {
                        StepsSlider(numSteps: $settings.numSteps)
                        Divider()
                        CFGSettings(enableCFG: $settings.enableCFG, guidanceScale: $settings.guidanceScale)
                    }
                }
            }
            .padding(20)
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.large)
    }
}

// MARK: - Section Container

private struct SettingsSection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 14) {
                content()
            }
            .padding(18)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .shadow(color: .black.opacity(0.04), radius: 6, y: 2)
        }
    }
}

// MARK: - Scheduler Picker

private struct SchedulerPicker: View {
    @Bindable var model: MobileOModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(MobileOGenerator.SchedulerType.allCases, id: \.self) { schedulerType in
                Button {
                    guard model.schedulerType != schedulerType else { return }
                    model.schedulerType = schedulerType
                    Task { await model.loadWithScheduler(schedulerType) }
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: model.schedulerType == schedulerType ? "circle.inset.filled" : "circle")
                            .font(.system(size: 16))
                            .foregroundStyle(model.schedulerType == schedulerType ? .purple : .secondary)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(schedulerType.rawValue)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.primary)
                            Text(schedulerType.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 10)
                    .background(model.schedulerType == schedulerType ? Color.purple.opacity(0.1) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Steps Slider

private struct StepsSlider: View {
    @Binding var numSteps: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Inference Steps")
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text("\(Int(numSteps))")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.thinMaterial)
                    .clipShape(Capsule())
            }
            Slider(value: $numSteps, in: 1...50, step: 1)
                .tint(.blue)
        }
    }
}

// MARK: - CFG Settings

private struct CFGSettings: View {
    @Binding var enableCFG: Bool
    @Binding var guidanceScale: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $enableCFG) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Classifier-Free Guidance (CFG)")
                        .font(.subheadline.weight(.medium))
                    Text("2× slower but significantly better quality and prompt following")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .tint(.orange)

            if enableCFG {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Guidance Scale")
                            .font(.subheadline.weight(.medium))
                        Spacer()
                        Text(String(format: "%.1f", guidanceScale))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(.thinMaterial)
                            .clipShape(Capsule())
                    }
                    Slider(value: $guidanceScale, in: 1.0...2.0, step: 0.1)
                        .tint(.purple)
                    Text("Higher = stronger prompt following")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
