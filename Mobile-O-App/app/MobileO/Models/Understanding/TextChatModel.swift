import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Tokenizers
import MLXVLM

/// Observable wrapper for text-only chat inference using the shared VLM
@Observable
@MainActor
class TextChatModel {

    public var running = false
    public var modelInfo = ""
    public var currentResponse: String = ""
    public var timeToFirstToken: TimeInterval = 0
    public var totalTime: TimeInterval = 0
    public var tokensGenerated: Int = 0

    private var container: ModelContainer?
    private var currentTask: Task<String, Never>?
    private var startTime: Date?
    private var firstTokenTime: Date?

    public init(container: ModelContainer? = nil) {
        self.container = container
    }

    public func load() async {
        modelInfo = container != nil ? "Ready for text chat" : "Error: Container not provided"
    }

    nonisolated public func chat(messages: [[String: String]], temperature: Float = 0.6) async -> String {
        await MainActor.run {
            currentTask?.cancel()
            running = true
            currentResponse = ""
            timeToFirstToken = 0
            totalTime = 0
            tokensGenerated = 0
            startTime = Date()
            firstTokenTime = nil
        }

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return "" }
            let activity = ProcessInfo.processInfo.beginActivity(
                options: [.suddenTerminationDisabled, .userInitiated],
                reason: "ML Text Chat"
            )
            defer { ProcessInfo.processInfo.endActivity(activity) }

            do {
                let container = await self.container
                guard let container = container else {
                    throw NSError(domain: "TextChatModel", code: -1,
                                 userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
                }

                guard let content = messages.last?["content"] else {
                    throw NSError(domain: "TextChatModel", code: -2,
                                 userInfo: [NSLocalizedDescriptionKey: "No message content found"])
                }

                let userInput = UserInput(prompt: .text(content), images: [])

                var fullResponse = ""
                var processedTokenCount = 0

                try await container.perform { context in
                    guard let processor = context.processor as? UserInputProcessor else {
                        throw NSError(domain: "TextChatModel", code: -3,
                                     userInfo: [NSLocalizedDescriptionKey: "Invalid processor type"])
                    }

                    let preparedInput = try await processor.prepare(input: userInput)

                    let result = try MLXLMCommon.generate(
                        input: preparedInput,
                        parameters: GenerateParameters(temperature: temperature),
                        context: context
                    ) { tokens in
                        if Task.isCancelled { return .stop }

                        if tokens.count > processedTokenCount {
                            let newTokens = Array(tokens[processedTokenCount...])
                            let tokenString = context.tokenizer.decode(tokens: newTokens)
                            let currentTokenCount = tokens.count

                            Task { @MainActor in
                                if self.firstTokenTime == nil {
                                    self.firstTokenTime = Date()
                                    if let start = self.startTime {
                                        self.timeToFirstToken = self.firstTokenTime!.timeIntervalSince(start)
                                    }
                                }
                                self.currentResponse += tokenString
                                self.tokensGenerated = currentTokenCount
                            }

                            processedTokenCount = tokens.count
                        }
                        return .more
                    }

                    fullResponse = result.output
                }

                await MainActor.run {
                    if !Task.isCancelled { self.currentResponse = fullResponse }
                    if let start = self.startTime { self.totalTime = Date().timeIntervalSince(start) }
                    self.running = false
                }

                return Task.isCancelled ? await MainActor.run { self.currentResponse } : fullResponse

            } catch {
                if !Task.isCancelled {
                    await MainActor.run { self.modelInfo = "Error: \(error.localizedDescription)" }
                }
                await MainActor.run { self.running = false }
                return ""
            }
        }

        await MainActor.run { currentTask = task }
        return await task.value
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        timeToFirstToken = 0
        totalTime = 0
        tokensGenerated = 0
        startTime = nil
        firstTokenTime = nil
    }

    public func releaseModels() {
        container = nil
    }
}
