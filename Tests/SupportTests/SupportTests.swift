import XCTest
import ModelSupport

final class SupportTests: XCTestCase {
    func testBijectiveDictionaryConstruct() {
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])

      let dictionary: [Int:String] = [1: "one", 2: "two"]
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary(dictionary)

      let array: [(Int, String)] = [(1, "one"), (2, "two")]
      let _: BijectiveDictionary<Int, String> =
          BijectiveDictionary(array)
    }

    func testBijectiveDictionaryCount() {
      let map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map.count, 2)
    }

    func testBijectiveDictionarySubscript() {
      let map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map[1], "one")
      XCTAssertEqual(map.key("one"), 1)
    }

    func testBijectiveDictionaryDeletion() {
      var map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "one", 2: "two"])
      XCTAssertEqual(map.count, 2)

      map[2] = nil

      XCTAssertEqual(map.count, 1)
      XCTAssertEqual(map[1], "one")
      XCTAssertEqual(map.key("one"), 1)
    }

    func testBijectiveDictionaryRemapping() {
      // 1 -> "two", 2 -> "four"
      var map: BijectiveDictionary<Int, String> =
          BijectiveDictionary([1: "two", 2: "four"])
      XCTAssertEqual(map.count, 2)

      // 1 -> "three", 2 -> "four"
      map[1] = "three"

      XCTAssertEqual(map.count, 2)
      XCTAssertEqual(map[1], "three")
      XCTAssertEqual(map[2], "four")

      // 2 -> "three"
      map[2] = "three"

      XCTAssertEqual(map.count, 1)
      XCTAssertEqual(map[2], "three")
    }

    static var allTests = [
        ("testBijectiveDictionaryConstruct", testBijectiveDictionaryConstruct),
        ("testBijectiveDictionaryCount", testBijectiveDictionaryCount),
        ("testBijectiveDictionarySubscript", testBijectiveDictionarySubscript),
        ("testBijectiveDictionaryDeletion", testBijectiveDictionaryDeletion),
        ("testBijectiveDictionaryRemapping", testBijectiveDictionaryRemapping),
    ]
}

