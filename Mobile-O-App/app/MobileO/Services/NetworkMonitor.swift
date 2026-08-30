import Foundation
import Network

/// Lightweight wrapper around `NWPathMonitor` exposing connectivity state for SwiftUI.
@Observable
final class NetworkMonitor {

    enum ConnectionType: String {
        case wifi, cellular, wired, unknown
    }

    private(set) var isConnected = false
    private(set) var isExpensive = false
    private(set) var connectionType: ConnectionType = .unknown

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor", qos: .utility)

    init() {
        start()
    }

    func start() {
        monitor.pathUpdateHandler = { [weak self] path in
            let connected = path.status == .satisfied
            let expensive = path.isExpensive
            let type: ConnectionType = {
                if path.usesInterfaceType(.wifi) { return .wifi }
                if path.usesInterfaceType(.cellular) { return .cellular }
                if path.usesInterfaceType(.wiredEthernet) { return .wired }
                return .unknown
            }()
            DispatchQueue.main.async {
                self?.isConnected = connected
                self?.isExpensive = expensive
                self?.connectionType = type
            }
        }
        monitor.start(queue: queue)
    }

    func stop() {
        monitor.cancel()
    }

    deinit {
        monitor.cancel()
    }
}
