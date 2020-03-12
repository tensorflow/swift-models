import TensorFlow

//Build a dataset suitable for language modeling from an array of texts
public struct LanguageModelDataset<Item> {
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
  //Drop the last batch if its length is less than sequenceLength
  public let dropLast: Bool
  //The length of a contiguous chunk of text
  private var batchLength: Int
  //The number of batches
  private var batchCount: Int
  //The sequence length of the last batch
  private var lastLength: Int
  //Indices used to iterate through the dataset
  public var indices: [Int]
  //Cumulative lengths
  private var cumulativeLengths: [Int]

  public init(
    openItem: @escaping (Item) -> [Int],
    batchSize: Int,
    sequenceLength: Int,
    items: [Item],
    lengths: [Int],
    dropLast: Bool = false
  ) {
    self.openItem = openItem
    self.batchSize = batchSize
    self.sequenceLength = sequenceLength
    self.items = items
    self.lengths = lengths
    self.dropLast = dropLast
    cumulativeLengths = lengths.reduce(into: []) { $0.append(($0.last ?? 0) + $1) }
    batchLength = (cumulativeLengths.last! - 1) / batchSize
    if dropLast {
        batchLength = (batchLength / sequenceLength) * sequenceLength
    }
    batchCount = batchLength / sequenceLength + (batchLength % sequenceLength == 0 ? 0 : 1)
    lastLength = batchLength - (batchCount - 1) * sequenceLength
    indices = Array(0..<items.count)
  }

  public init(
    openItem: @escaping (Item) -> [Int],
    batchSize: Int,
    sequenceLength: Int,
    items: [Item],
    dropLast: Bool = false
  ) {
    self.init(
      openItem: openItem,
      batchSize: batchSize,
      sequenceLength: sequenceLength,
      items: items,
      lengths: items.map { openItem($0).count },
      dropLast: dropLast)
  }

  //Shuflle the dataset
  public mutating func shuffle() {
    indices = indices.shuffled()
    cumulativeLengths[0] = lengths[indices[0]]
    for (i, j) in indices.suffix(from: 1).enumerated() {
      cumulativeLengths[i + 1] = cumulativeLengths[i] + lengths[j]
    }
  }
}

extension LanguageModelDataset: Collection {
  public var startIndex: Int { return 0 }
  public var endIndex: Int { return batchCount * batchSize }  
  
  public func index(after i: Int) -> Int { return i + 1 }
    
  public subscript(index: Int) -> TensorPair<Int32, Int32> {
    get {
      let sampleLength = index / batchSize == batchCount - 1 ? lastLength : sequenceLength
      let startIndex = (index % batchSize) * batchLength + (index / batchSize) * sequenceLength
      let sample = readItems(from: startIndex, to: startIndex + sampleLength + 1)
      let sample32 = sample.map { Int32($0) }
      return TensorPair(
        first: Tensor<Int32>(sample32.prefix(upTo: sampleLength)),
        second: Tensor<Int32>(sample32.suffix(from: 1)))
    }
  }  
  
  //Read a contiguous chunk of texts from start to end (may go through several items)
  private func readItems(from start: Int, to end: Int) -> [Int] {
    var text: [Int] = []
    var index = cumulativeLengths.firstIndex { $0 >= start }!
    var position = start
    while position < end {
      let x = openItem(items[indices[index]])
      let cumulativeLength = ([0] + cumulativeLengths)[index]
      let readFrom = position - cumulativeLength
      let readUntil = Swift.min(end - cumulativeLength, x.count)
      text = text + Array(x[readFrom..<readUntil])
      position = readUntil + cumulativeLength
      index += 1
    }
    return text
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
