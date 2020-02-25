import TensorFlow

//Build a dataset suitable for language modeling from an array of texts
public struct LanguageModelDataset<Item>: Collection {
  public typealias Index = Int
  public typealias Element = TensorPair<Int32, Int32>

  //A function that reads Item to get an array of Int
  public let openItem: (Item) -> [Int]
  //The size of a batch
  public var batchSize: Int
  //The length of a sequence
  public var sequenceLength: Int
  //The array of raw items to use
  public let items: [Item]
  //The length of each processed item
  public let lengths: [Int]
  //The length of a contiguous chunk of text
  private var batchLength: Int
  //The number of batches
  private var batchCount: Int
  //The sequence length of the last batch
  private var lastLength: Int
  //Indices used to iterate through the dataset
  public var indices: [Int]
  //Cumulative lengths
  private var cumLengths: [Int]
  //To conform to Collection
  public var startIndex: Int { return 0 }
  public var endIndex: Int { return batchCount * batchSize }

  public init(
    openItem: @escaping (Item) -> [Int],
    batchSize: Int,
    sequenceLength: Int,
    items: [Item],
    lengths: [Int]
  ) {
    self.openItem = openItem
    self.batchSize = batchSize
    self.sequenceLength = sequenceLength
    self.items = items
    self.lengths = lengths
    cumLengths = lengths.reduce(into: []) { $0.append(($0.last ?? 0) + $1) }
    batchLength = (cumLengths.last! - 1) / batchSize
    batchCount = batchLength / sequenceLength + (batchLength % sequenceLength == 0 ? 0 : 1)
    lastLength = batchLength - (batchCount - 1) * sequenceLength
    indices = Array(0..<items.count)
  }

  public init(
    openItem: @escaping (Item) -> [Int],
    batchSize: Int,
    sequenceLength: Int,
    items: [Item]
  ) {
    self.init(
      openItem: openItem,
      batchSize: batchSize,
      sequenceLength: sequenceLength,
      items: items,
      lengths: items.map { openItem($0).count })
  }

  // Method that returns the next index when iterating
  public func index(after i: Int) -> Int { return i + 1 }

  // Required subscript for Collection
  public subscript(index: Int) -> TensorPair<Int32, Int32> {
    get {
      let sampleLength = index / batchSize == batchCount - 1 ? lastLength : sequenceLength
      let startIndex = (index % batchSize) * batchLength + (index / batchSize) * sequenceLength
      let sample = readItems(from: startIndex, to: startIndex + sampleLength + 1)
      let sample32 = sample.map { Int32($0) }
      return TensorPair(
        input: Tensor<Int32>(sample32.prefix(upTo: sampleLength)),
        target: Tensor<Int32>(sample32.suffix(from: 1)))
    }
  }

  //Read a contiguous chunk of texts from start to end (may go througyh several items)
  private func readItems(from start: Int, to end: Int) -> [Int] {
    var res: [Int] = []
    var index = cumLengths.firstIndex { $0 >= start }!
    var pos = start
    while pos < end {
      let x = openItem(items[indices[index]])
      let cumLen = ([0] + cumLengths)[index]
      let readFrom = pos - cumLen
      let readUntil = Swift.min(end - cumLen, x.count)
      res = res + Array(x[readFrom..<readUntil])
      pos = readUntil + cumLen
      index += 1
    }
    return res
  }

  //Shuflle the dataset
  public mutating func shuffle() {
    indices = indices.shuffled()
    cumLengths[0] = lengths[indices[0]]
    for (i, j) in indices.suffix(from: 1).enumerated() {
      cumLengths[i + 1] = cumLengths[i] + lengths[j]
    }
  }
}

//Extension when Item is [Int] and openItem is not needed
extension LanguageModelDataset where Item == [Int] {
  public init(batchSize: Int, sequenceLength: Int, items: [Item], lengths: [Int]) {
    self.init(
      openItem: { $0 },
      batchSize: batchSize,
      sequenceLength: sequenceLength,
      items: items,
      lengths: lengths)
  }

  public init(batchSize: Int, sequenceLength: Int, items: [Item]) {
    self.init(
      openItem: { $0 },
      batchSize: batchSize,
      sequenceLength: sequenceLength,
      items: items,
      lengths: items.map { $0.count })
  }
}

//sampleIndices function to use in conjunction with a LanguageModelDataset
public func languageModelSample<C>(on dataset: inout LanguageModelDataset<C>, shuffled: Bool)
  -> [Int]
{
  if shuffled { dataset.shuffle() }
  return Array(0..<dataset.count)
}
