import XCTest

import ImageClassificationTests
import MiniGoTests
import FastStyleTransferTests
import DatasetsTests


var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
tests += DatasetsTests.allTests()
XCTMain(tests)
