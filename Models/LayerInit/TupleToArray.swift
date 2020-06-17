public func intTupleToArray(tuple: Any) -> [Int] {
    if let tuple = tuple as? (Int, Int) {
        return [tuple.0, tuple.1]
    }
    if let tuple = tuple as? (Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2]
    }
    if let tuple = tuple as? (Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14, tuple.15]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14, tuple.15, tuple.16]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14, tuple.15, tuple.16, tuple.17]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14, tuple.15, tuple.16, tuple.17, tuple.18]
    }
    if let tuple = tuple as? (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) {
        return [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9, tuple.10, tuple.11, tuple.12, tuple.13, tuple.14, tuple.15, tuple.16, tuple.17, tuple.18, tuple.19]
    }
    
    fatalError("Could not extract out elements of shape")
}
