import XCTest

import CustomLayersTests
import SNGANTests

var tests = [XCTestCaseEntry]()
tests += CustomLayersTests.__allTests()
tests += SNGANTests.__allTests()

XCTMain(tests)
