import XCTest

import ImageClassificationTests
import MiniGoTests
import FastStyleTransferTests
import DatasetsTests
import CheckpointTests

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
tests += DatasetsTests.allTests()
tests += CheckpointTests.allTests()
XCTMain(tests)
