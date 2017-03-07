cmake_minimum_required(VERSION 2.8)

cpu_setup("exkurumi")

function(exkurumi_add_executable exe)
  cpu_add_executable(${exe} ${ARGN})
  cpu_target_link_common_libraries(${exe})
  cpu_target_link_libraries(${exe} puyoai_core_server_connector)
endfunction()

cpu_add_runner(run.sh)

exkurumi_add_executable(exkurumi main.cc puppet_ai.cc ai.cc core.h)
