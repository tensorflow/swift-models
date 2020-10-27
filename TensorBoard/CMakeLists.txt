add_library(TensorBoard
  SummaryWriter.swift
  EventLogger.swift
  Support.swift
  Protobufs/summary.pb.swift
  Protobufs/event.pb.swift
  Protobufs/tensor.pb.swift
  Protobufs/tensor_shape.pb.swift
  Protobufs/types.pb.swift
  Protobufs/resource_handle.pb.swift)
set_target_properties(TensorBoard PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(TensorBoard PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(TensorBoard PUBLIC
  SwiftProtobuf
  ModelSupport)

install(TARGETS TensorBoard
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:TensorBoard,Swift_MODULE_DIRECTORY>/TensorBoard.swiftdoc
  $<TARGET_PROPERTY:TensorBoard,Swift_MODULE_DIRECTORY>/TensorBoard.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})