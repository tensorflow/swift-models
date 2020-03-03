add_library(MiniGo
  GameLib/Board.swift
  GameLib/BoardState.swift
  GameLib/Color.swift
  GameLib/GameConfiguration.swift
  GameLib/LibertyTracker.swift
  GameLib/Position.swift

  Models/GoModel.swift
  Models/MiniGoCheckpointReader.swift

  Play/PlayOneGame.swift

  Strategies/HumanPolicy.swift
  Strategies/Policy.swift
  Strategies/RandomPolicy.swift

  Strategies/MCTS/MCTSConfiguration.swift
  Strategies/MCTS/MCTSModelBasePredictor.swift
  Strategies/MCTS/MCTSNode.swift
  Strategies/MCTS/MCTSPolicy.swift
  Strategies/MCTS/MCTSPredictor.swift
  Strategies/MCTS/MCTSRandomPredictor.swift
  Strategies/MCTS/MCTSTree.swift)
set_target_properties(MiniGo PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(MiniGo PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(MiniGo PUBLIC
  ModelSupport)

add_executable(MiniGoDemo
  main.swift)
target_link_libraries(MiniGoDemo PRIVATE
  MiniGo)


install(TARGETS MiniGo
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
install(TARGETS MiniGoDemo
  DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:MiniGo,Swift_MODULE_DIRECTORY>/MiniGo.swiftdoc
  $<TARGET_PROPERTY:MiniGo,Swift_MODULE_DIRECTORY>/MiniGo.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
