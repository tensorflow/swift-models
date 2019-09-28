import XCTest

import ImageClassificationTests
import MiniGoTests

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
XCTMain(tests)
