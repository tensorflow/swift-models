import XCTest

import ImageClassificationTests
import MiniGoTests
import FastStyleTransferTests
import DatasetsTests
import CheckpointTests
import RecommendationModelTests

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
tests += DatasetsTests.allTests()
tests += CheckpointTests.allTests()
tests += RecommendationModelTests.allTests()
XCTMain(tests)
