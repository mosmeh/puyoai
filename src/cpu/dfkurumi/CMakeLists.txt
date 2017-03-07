cmake_minimum_required(VERSION 2.8)

cpu_setup("dfkurumi")

function(dfkurumi_add_executable exe)
  cpu_add_executable(${exe} ${ARGN})
  cpu_target_link_common_libraries(${exe})

  cpu_target_link_libraries(${exe} rt)
endfunction()


cpu_add_runner(run.sh)

dfkurumi_add_executable(dfkurumi main.cc ai.cc core.h feature_extractor.h stochastic_decision_tree.cc)

