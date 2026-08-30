import SwiftUI

/// Observable settings for image generation parameters, persisted via UserDefaults.
@Observable
class SettingsViewModel {
    // Generation parameters (persisted via UserDefaults with manual observation)
    var numSteps: Double {
        didSet {
            UserDefaults.standard.set(numSteps, forKey: "numSteps")
        }
    }

    var guidanceScale: Double {
        didSet {
            UserDefaults.standard.set(guidanceScale, forKey: "guidanceScale")
        }
    }

    var enableCFG: Bool {
        didSet {
            UserDefaults.standard.set(enableCFG, forKey: "enableCFG")
        }
    }

    // Seed is transient (not persisted - user sets per generation)
    var seed: UInt64? = nil

    init() {
        // Load persisted values from UserDefaults
        self.numSteps = UserDefaults.standard.object(forKey: "numSteps") as? Double ?? 15.0
        self.guidanceScale = UserDefaults.standard.object(forKey: "guidanceScale") as? Double ?? 1.3
        self.enableCFG = UserDefaults.standard.object(forKey: "enableCFG") as? Bool ?? true
    }
}
