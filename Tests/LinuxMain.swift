import XCTest

import MiniGoTests
import FastStyleTransferTests

var tests = [XCTestCaseEntry]()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
XCTMain(tests)
