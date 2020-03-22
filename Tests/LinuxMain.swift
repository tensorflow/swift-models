import CheckpointTests
import DatasetsTests
import FastStyleTransferTests
import ImageClassificationTests
import MiniGoTests
import RecommendationModelTests
import SupportTests
import TextTests
import XCTest

var tests = [XCTestCaseEntry]()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
tests += DatasetsTests.allTests()
tests += CheckpointTests.allTests()
tests += RecommendationModelTests.allTests()
tests += SupportTests.allTests()
tests += TextTests.allTests()
XCTMain(tests)
