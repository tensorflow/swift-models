import XCTest

import ImageClassificationTests
import MiniGoTests
import DatasetsTests

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += DatasetsTests.allTests()
XCTMain(tests)
