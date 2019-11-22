import Foundation

var stderr = FileHandle.standardError

extension FileHandle: TextOutputStream {
    public func write(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        self.write(data)
    }
}

public func printerr(_ msg: String) {
    print(msg, to: &stderr)
}
