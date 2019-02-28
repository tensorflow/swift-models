import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
  return [
    testCase(BoardStateTests.allTests),
    testCase(GoModelTests.allTests),
    testCase(MCTSModelBasedPredictorTests.allTests),
    testCase(MCTSNodeTests.allTests),
    testCase(MCTSPolicyTests.allTests),
  ]
}
#endif
