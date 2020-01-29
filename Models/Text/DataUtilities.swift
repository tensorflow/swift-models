// Adapted from: https://gist.github.com/eaplatanios/5004c5857ec3140651ccef6766123ac2

import Foundation

extension IteratorProtocol {
  /// Returns an iterator that maps elements of this iterator using the provided function.
  ///
  /// - Parameters:
  ///   - mapFn: Function used to map the iterator elements.
  public func map<MappedElement>(
    _ mapFn: @escaping (Element) -> MappedElement
  ) -> MapIterator<Self, MappedElement> {
    MapIterator(self, mapFn: mapFn)
  }

  /// Returns an iterator that repeats this iterator indefinitely.
  public func repeated() -> RepeatIterator<Self> {
    RepeatIterator(self)
  }

  /// Returns an iterator that shuffles this iterator using a temporary buffer.
  ///
  /// - Parameters:
  ///   - bufferSize: Size of the shuffle buffer.
  public func shuffled(bufferSize: Int) -> ShuffleIterator<Self> {
    ShuffleIterator(self, bufferSize: bufferSize)
  }

  // TODO: [DOC] Add documentation string.
  public func grouped(
    keyFn: @escaping (Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Element]) -> Element
  ) -> GroupedIterator<Self> {
    GroupedIterator(self, keyFn: keyFn, sizeFn: sizeFn, reduceFn: reduceFn)
  }

  // TODO: [DOC] Add documentation string.
  public func prefetched(count: Int) -> PrefetchIterator<Self> {
    PrefetchIterator(self, prefetchCount: count)
  }
}

extension IteratorProtocol where Element: KeyPathIterable {
  /// Returns an iterator that batches elements of this iterator.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  public func batched(batchSize: Int) -> BatchIterator<Self> {
    BatchIterator(self, batchSize: batchSize)
  }
}

/// Iterator that maps elements of another iterator using the provided function.
public struct MapIterator<Base: IteratorProtocol, MappedElement>: IteratorProtocol {
  private var iterator: Base
  private let mapFn: (Base.Element) -> MappedElement

  public init(_ iterator: Base, mapFn: @escaping (Base.Element) -> MappedElement) {
    self.iterator = iterator
    self.mapFn = mapFn
  }

  public mutating func next() -> MappedElement? {
    if let element = iterator.next() { return mapFn(element) }
    return nil
  }
}

/// Iterator that repeats another iterator indefinitely.
public struct RepeatIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let originalIterator: Base
  private var currentIterator: Base

  public init(_ iterator: Base) {
    self.originalIterator = iterator
    self.currentIterator = iterator
  }

  public mutating func next() -> Base.Element? {
    if let element = currentIterator.next() {
      return element
    }
    currentIterator = originalIterator
    return currentIterator.next()
  }
}

/// Iterator that shuffles another iterator using a temporary buffer.
public struct ShuffleIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let bufferSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]
  private var bufferIndex: Int

  public init(_ iterator: Base, bufferSize: Int) {
    self.bufferSize = bufferSize
    self.iterator = iterator
    self.buffer = []
    self.bufferIndex = 0
  }

  public mutating func next() -> Base.Element? {
    if buffer.isEmpty || (bufferIndex >= bufferSize && bufferSize != -1) { fillBuffer() }
    if buffer.isEmpty { return nil }
    bufferIndex += 1
    return buffer[bufferIndex - 1]
  }

  private mutating func fillBuffer() {
    buffer = []
    bufferIndex = 0
    while let element = iterator.next(), bufferIndex < bufferSize || bufferSize == -1 {
      buffer.append(element)
      bufferIndex += 1
    }
    bufferIndex = 0
  }
}

/// Iterator that batches elements from another iterator.
public struct BatchIterator<Base: IteratorProtocol>: IteratorProtocol
where Base.Element: KeyPathIterable {
  private let batchSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]

  public init(_ iterator: Base, batchSize: Int) {
    self.batchSize = batchSize
    self.iterator = iterator
    self.buffer = []
    self.buffer.reserveCapacity(batchSize)
  }

  public mutating func next() -> Base.Element? {
    while buffer.count < batchSize {
      if let element = iterator.next() {
        buffer.append(element)
      } else {
        break
      }
    }
    if buffer.isEmpty { return nil }
    let batch = Base.Element.batch(buffer)
    buffer = []
    buffer.reserveCapacity(batchSize)
    return batch
  }
}

/// Iterator that groups elements from another iterator.
public struct GroupedIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let keyFn: (Base.Element) -> Int
  private let sizeFn: (Int) -> Int
  private let reduceFn: ([Base.Element]) -> Base.Element
  private var iterator: Base
  private var groups: [Int: [Base.Element]]
  private var currentGroup: Dictionary<Int, [Base.Element]>.Index? = nil

  public init(
    _ iterator: Base,
    keyFn: @escaping (Base.Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Base.Element]) -> Base.Element
  ) {
    self.keyFn = keyFn
    self.sizeFn = sizeFn
    self.reduceFn = reduceFn
    self.iterator = iterator
    self.groups = [Int: [Base.Element]]()
  }

  public mutating func next() -> Base.Element? {
    var elements: [Base.Element]? = nil
    while elements == nil {
      if let element = iterator.next() {
        let key = keyFn(element)
        if !groups.keys.contains(key) {
          groups[key] = [element]
        } else {
          groups[key]!.append(element)
        }
        if groups[key]!.count >= sizeFn(key) {
          elements = groups.removeValue(forKey: key)!
        }
      } else {
        break
      }
    }
    guard let elementsToReduce = elements else {
      if currentGroup == nil { currentGroup = groups.values.startIndex }
      if currentGroup! >= groups.values.endIndex { return nil }
      while groups.values[currentGroup!].isEmpty {
        currentGroup = groups.values.index(after: currentGroup!)
      }
      let elementsToReduce = groups.values[currentGroup!]
      currentGroup = groups.values.index(after: currentGroup!)
      return reduceFn(elementsToReduce)
    }
    return reduceFn(elementsToReduce)
  }
}

/// Iterator that prefetches elements from another iterator asynchronously.
public struct PrefetchIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let iterator: Base
  private let prefetchCount: Int

  private var queue: BlockingQueue<Base.Element>

  public init(_ iterator: Base, prefetchCount: Int) {
    self.iterator = iterator
    self.prefetchCount = prefetchCount
    self.queue = BlockingQueue<Base.Element>(count: prefetchCount, iterator: iterator)
  }

  public mutating func next() -> Base.Element? {
    queue.read()
  }

  // TODO: !!! This is needed because `BlockingQueue` is a class. Figure out a better solution.
  public func copy() -> PrefetchIterator {
    PrefetchIterator(iterator, prefetchCount: prefetchCount)
  }
}

extension PrefetchIterator {
  internal class BlockingQueue<Element> {
    private let prefetchingDispatchQueue: DispatchQueue = DispatchQueue(label: "PrefetchIterator")
    private let writeSemaphore: DispatchSemaphore
    private let readSemaphore: DispatchSemaphore
    private let deletedSemaphore: DispatchSemaphore
    private let dispatchQueue: DispatchQueue
    private var array: [Element?]
    private var readIndex: Int
    private var writeIndex: Int
    private var depleted: Bool
    private var deleted: Bool

    internal init<Base: IteratorProtocol>(
      count: Int,
      iterator: Base
    ) where Base.Element == Element {
      self.writeSemaphore = DispatchSemaphore(value: count)
      self.readSemaphore = DispatchSemaphore(value: 0)
      self.deletedSemaphore = DispatchSemaphore(value: 0)
      self.dispatchQueue = DispatchQueue(label: "BlockingQueue")
      self.array = [Element?](repeating: nil, count: count)
      self.readIndex = 0
      self.writeIndex = 0
      self.depleted = false
      self.deleted = false
      var iterator = iterator
      prefetchingDispatchQueue.async { [unowned self] () in
        while !self.deleted {
          if let element = iterator.next() {
            self.write(element)
          } else {
            self.depleted = true
            self.readSemaphore.signal()
            self.deletedSemaphore.signal()
            break
          }
        }
        self.readSemaphore.signal()
        self.deletedSemaphore.signal()
      }
    }

    deinit {
      self.deleted = true

      // Signal the write semaphore to make sure it's not in use anymore. It's final value must be
      // greater or equal to its initial value.
      for _ in 0...array.count { writeSemaphore.signal() }

      // Wait for the delete semaphore to make sure the prefetching thread is done.
      deletedSemaphore.wait()
    }

    private func write(_ element: Element) {
      writeSemaphore.wait()
      dispatchQueue.sync {
        array[writeIndex % array.count] = element
        writeIndex += 1
      }
      readSemaphore.signal()
    }

    internal func read() -> Element? {
      if self.depleted { return nil }
      readSemaphore.wait()
      let element = dispatchQueue.sync { () -> Element? in
        let element = array[readIndex % array.count]
        array[readIndex % array.count] = nil
        readIndex += 1
        return element
      }
      writeSemaphore.signal()
      return element
    }
  }
}
