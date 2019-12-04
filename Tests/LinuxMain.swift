import XCTest

import ImageClassificationTests
import MiniGoTests
import FastStyleTransferTests

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
XCTMain(tests)
