add_library(MiniGoTests
  GameLib/BoardStateTests.swift
  GameLib/XCTestManifests.swift
  Models/GoModelTests.swift
  Strategies/MCTS/MCTSModelBasedPredictorTests.swift
  Strategies/MCTS/MCTSNodeTests.swift
  Strategies/MCTS/MCTSPolicyTests.swift)
set_target_properties(MiniGoTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>
  LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:ModelTests>)
target_link_libraries(MiniGoTests PUBLIC
  MiniGo)
