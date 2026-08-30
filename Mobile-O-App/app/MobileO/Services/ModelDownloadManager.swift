import Foundation
import CoreML

/// Orchestrates downloading model weights from HuggingFace, compiling `.mlpackage` → `.mlmodelc`,
/// and managing download lifecycle (pause/resume/retry).
@Observable
@MainActor
final class ModelDownloadManager: NSObject {

    // MARK: - State Machine

    enum State: Equatable {
        case idle
        case downloading
        case paused
        case compiling
        case completed
        case failed(String)
    }

    // MARK: - Observable Properties

    private(set) var state: State = .idle
    private(set) var overallProgress: Double = 0
    private(set) var currentFileProgress: Double = 0
    private(set) var currentFileName: String = ""
    private(set) var downloadedBytes: Int64 = 0
    private(set) var totalBytes: Int64 = 0
    private(set) var downloadSpeed: Double = 0  // bytes/sec
    private(set) var eta: TimeInterval = 0
    private(set) var completedComponents: Set<String> = []
    private(set) var compilationProgress: String = ""

    /// True when all models are on disk and compiled — gates access to the main app.
    private(set) var modelsReady = false

    /// Check the filesystem for all required model files.
    func checkModelsReady() {
        let fm = FileManager.default
        let dir = modelsDirectory
        let coreMLReady = Self.coreMLComponents.allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent("\($0).mlmodelc").path)
        }
        let llmReady = fm.fileExists(atPath: dir.appendingPathComponent("llm/model.safetensors").path)
        modelsReady = coreMLReady && llmReady
    }

    /// Root directory where downloaded models live.
    let modelsDirectory: URL = {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    // MARK: - Manifest

    private static let repo = "Amshaker/Mobile-O-0.5B-iOS"
    private static let baseURL = "https://huggingface.co/\(repo)/resolve/main"
    static let coreMLComponents = ["connector", "transformer", "vae_decoder", "vision_encoder"]

    /// Each downloadable file: (relative path inside the repo, relative destination on disk).
    private static let fileManifest: [(remotePath: String, localPath: String, component: String)] = {
        var files: [(String, String, String)] = []

        // CoreML .mlpackage files
        for name in coreMLComponents {
            files.append(("\(name).mlpackage/Manifest.json",
                          "\(name).mlpackage/Manifest.json", name))
            files.append(("\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel",
                          "\(name).mlpackage/Data/com.apple.CoreML/model.mlmodel", name))
            files.append(("\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                          "\(name).mlpackage/Data/com.apple.CoreML/weights/weight.bin", name))
        }

        // LLM files
        let llmFiles = [
            "added_tokens.json", "config.json", "merges.txt",
            "model.safetensors", "model.safetensors.index.json",
            "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json"
        ]
        for f in llmFiles {
            files.append(("llm/\(f)", "llm/\(f)", "llm"))
        }

        return files
    }()

    // MARK: - Private State

    private var session: URLSession!
    private var currentTask: URLSessionDownloadTask?
    private var currentFileIndex = 0
    private var resumeData: Data?
    private var speedSamples: [(Date, Int64)] = []
    private var downloadStartTime: Date?
    /// Cumulative bytes from files that finished downloading (not counting current in-progress file).
    private var bytesFromCompletedFiles: Int64 = 0
    /// Current file's totalBytesWritten (reset per file).
    private var currentFileBytesWritten: Int64 = 0

    // Persistent resume data path
    private var resumeDataPath: URL {
        modelsDirectory.appendingPathComponent(".resume_data")
    }
    private var progressFilePath: URL {
        modelsDirectory.appendingPathComponent(".download_progress")
    }

    // Download continuation
    private var downloadContinuation: CheckedContinuation<URL, Error>?

    // MARK: - Init

    override init() {
        super.init()

        let config = URLSessionConfiguration.background(withIdentifier: "com.ival.mobileo.modeldownload")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsCellularAccess = true
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: .main)

        // Restore progress if we were mid-download
        restoreProgress()
        checkModelsReady()
    }

    // MARK: - Public Actions

    /// Check available disk space. Returns true if there's enough room (~5 GB buffer).
    func hasSufficientDiskSpace() -> Bool {
        let requiredBytes: Int64 = 5_000_000_000 // ~5 GB for download + compilation
        let resourceValues = try? modelsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let available = resourceValues?.volumeAvailableCapacityForImportantUsage ?? 0
        return available >= requiredBytes
    }

    /// Start downloading all model files sequentially.
    func startDownload() {
        guard state == .idle || state == .paused || state != .downloading else { return }

        if case .failed = state {
            // Retry: keep currentFileIndex where it was
        }

        if !hasSufficientDiskSpace() {
            state = .failed("Not enough disk space. Please free at least 5 GB and try again.")
            return
        }

        state = .downloading
        downloadStartTime = Date()
        speedSamples.removeAll()

        // Estimate total bytes (actual will be refined by Content-Length headers)
        if totalBytes == 0 {
            totalBytes = 3_630_000_000 // ~3.63 GB estimate
        }

        Task { await downloadNextFile() }
    }

    func pause() {
        guard state == .downloading else { return }
        currentTask?.cancel(byProducingResumeData: { [weak self] data in
            Task { @MainActor in
                self?.resumeData = data
                self?.saveResumeData(data)
            }
        })
        state = .paused
        saveProgress()
    }

    func resume() {
        guard state == .paused else { return }
        startDownload()
    }

    func cancel() {
        currentTask?.cancel()
        currentTask = nil
        state = .idle
        currentFileIndex = 0
        downloadedBytes = 0
        resumeData = nil
        cleanupProgressFiles()
    }

    func retry() {
        guard case .failed = state else { return }
        startDownload()
    }

    // MARK: - Sequential Download Engine

    private func downloadNextFile() async {
        guard currentFileIndex < Self.fileManifest.count else {
            // All files downloaded — start compilation
            await compileModels()
            return
        }

        let entry = Self.fileManifest[currentFileIndex]
        currentFileName = entry.localPath

        // Skip if file already exists
        let destURL = modelsDirectory.appendingPathComponent(entry.localPath)
        if FileManager.default.fileExists(atPath: destURL.path) {
            markComponentIfComplete(entry.component)
            currentFileIndex += 1
            saveProgress()
            await downloadNextFile()
            return
        }

        // Create parent directories
        let parentDir = destURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        let remoteURL = URL(string: "\(Self.baseURL)/\(entry.remotePath)")!

        do {
            let localURL = try await downloadFile(from: remoteURL)

            // Move to final destination
            try? FileManager.default.removeItem(at: destURL) // remove if exists
            try FileManager.default.moveItem(at: localURL, to: destURL)

            // Accumulate completed bytes and reset per-file counter
            bytesFromCompletedFiles += currentFileBytesWritten
            currentFileBytesWritten = 0
            currentFileProgress = 0

            markComponentIfComplete(entry.component)
            currentFileIndex += 1
            resumeData = nil
            cleanupResumeData()
            saveProgress()

            // Continue to next file if still downloading
            guard state == .downloading else { return }
            await downloadNextFile()
        } catch {
            if (error as NSError).code == NSURLErrorCancelled {
                // User paused — don't treat as failure
                return
            }
            state = .failed("Download failed: \(error.localizedDescription)")
            saveProgress()
        }
    }

    private func downloadFile(from url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation

            if let data = resumeData {
                currentTask = session.downloadTask(withResumeData: data)
                resumeData = nil
            } else {
                currentTask = session.downloadTask(with: url)
            }
            currentTask?.resume()
        }
    }

    // MARK: - Compilation

    private func compileModels() async {
        state = .compiling

        for (index, name) in Self.coreMLComponents.enumerated() {
            compilationProgress = "Compiling \(name) (\(index + 1)/\(Self.coreMLComponents.count))..."

            let packageURL = modelsDirectory.appendingPathComponent("\(name).mlpackage")
            let compiledURL = modelsDirectory.appendingPathComponent("\(name).mlmodelc")

            // Skip if already compiled
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                completedComponents.insert("\(name)_compiled")
                continue
            }

            do {
                let tempCompiledURL = try await Task.detached(priority: .userInitiated) {
                    try MLModel.compileModel(at: packageURL)
                }.value

                // Move compiled model to final location
                try? FileManager.default.removeItem(at: compiledURL)
                try FileManager.default.moveItem(at: tempCompiledURL, to: compiledURL)

                // Delete the raw .mlpackage to save disk space
                try? FileManager.default.removeItem(at: packageURL)

                completedComponents.insert("\(name)_compiled")
            } catch {
                state = .failed("Failed to compile \(name): \(error.localizedDescription)")
                return
            }
        }

        compilationProgress = ""
        cleanupProgressFiles()
        state = .completed
    }

    // MARK: - Component Tracking

    private func markComponentIfComplete(_ component: String) {
        let componentFiles = Self.fileManifest.filter { $0.component == component }
        let allExist = componentFiles.allSatisfy { entry in
            FileManager.default.fileExists(atPath: modelsDirectory.appendingPathComponent(entry.localPath).path)
        }
        if allExist {
            completedComponents.insert(component)
        }
    }

    // MARK: - Speed / ETA Calculation

    private func updateSpeed(bytesWritten: Int64) {
        let now = Date()
        speedSamples.append((now, bytesWritten))

        // Keep a 5-second sliding window
        let cutoff = now.addingTimeInterval(-5)
        speedSamples.removeAll { $0.0 < cutoff }

        guard speedSamples.count >= 2,
              let first = speedSamples.first,
              let last = speedSamples.last else { return }

        let elapsed = last.0.timeIntervalSince(first.0)
        guard elapsed > 0 else { return }

        let totalBytesInWindow = speedSamples.reduce(Int64(0)) { $0 + $1.1 }
        downloadSpeed = Double(totalBytesInWindow) / elapsed

        let remaining = Double(totalBytes - downloadedBytes)
        eta = downloadSpeed > 0 ? remaining / downloadSpeed : 0
    }

    // MARK: - Progress Persistence

    private func saveProgress() {
        let info: [String: Any] = [
            "fileIndex": currentFileIndex,
            "bytesFromCompletedFiles": bytesFromCompletedFiles
        ]
        try? JSONSerialization.data(withJSONObject: info)
            .write(to: progressFilePath)
    }

    private func restoreProgress() {
        guard let data = try? Data(contentsOf: progressFilePath),
              let info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        currentFileIndex = info["fileIndex"] as? Int ?? 0
        bytesFromCompletedFiles = info["bytesFromCompletedFiles"] as? Int64 ?? 0
        downloadedBytes = bytesFromCompletedFiles
        resumeData = try? Data(contentsOf: resumeDataPath)

        if currentFileIndex > 0 && currentFileIndex < Self.fileManifest.count {
            // We have partial progress — set state to paused so user can resume
            state = .paused

            // Re-check completed components
            for entry in Self.fileManifest.prefix(currentFileIndex) {
                markComponentIfComplete(entry.component)
            }
        }
    }

    private func saveResumeData(_ data: Data?) {
        guard let data else { return }
        try? data.write(to: resumeDataPath)
    }

    private func cleanupResumeData() {
        try? FileManager.default.removeItem(at: resumeDataPath)
    }

    private func cleanupProgressFiles() {
        try? FileManager.default.removeItem(at: progressFilePath)
        cleanupResumeData()
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloadManager: URLSessionDownloadDelegate {

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Copy to a temp location before the system cleans up
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try? FileManager.default.copyItem(at: location, to: tempURL)

        MainActor.assumeIsolated {
            downloadContinuation?.resume(returning: tempURL)
            downloadContinuation = nil
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        MainActor.assumeIsolated {
            currentFileBytesWritten = totalBytesWritten
            if totalBytesExpectedToWrite > 0 {
                currentFileProgress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            }

            // Byte-based overall progress
            downloadedBytes = bytesFromCompletedFiles + totalBytesWritten
            overallProgress = totalBytes > 0 ? min(Double(downloadedBytes) / Double(totalBytes), 1.0) : 0

            updateSpeed(bytesWritten: bytesWritten)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        guard let error else { return }

        MainActor.assumeIsolated {
            // Extract resume data from the error if available
            let nsError = error as NSError
            if let data = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                resumeData = data
                saveResumeData(data)
            }

            if nsError.code == NSURLErrorCancelled {
                downloadContinuation?.resume(throwing: error)
                downloadContinuation = nil
                return
            }

            downloadContinuation?.resume(throwing: error)
            downloadContinuation = nil
        }
    }
}
