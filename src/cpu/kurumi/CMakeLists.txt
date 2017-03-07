cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/opt/OpenBLAS/include)
include_directories(/home/yut/Desktop/caffe/include)
link_directories(/opt/OpenBLAS/lib)
link_directories(/home/yut/Desktop/caffe/.build_release/lib)
add_definitions(-DCPU_ONLY)

cpu_setup("kurumi")

function(kurumi_add_executable exe)
  cpu_add_executable(${exe} ${ARGN})
  cpu_target_link_common_libraries(${exe})
  cpu_target_link_libraries(${exe} puyoai_core_server_connector)
  cpu_target_link_libraries(${exe} rt)

  cpu_target_link_libraries(${exe} caffe)
  cpu_target_link_libraries(${exe} boost_system)
endfunction()

kurumi_add_executable(kurumi kurumi_ai_main.cc kurumi_ai.cc core.h stochastic_decision_tree.cc feature_extractor.h neural_decision_tree.cc)
kurumi_add_executable(duel_recorder duel_recorder_ai_main.cc puppet_ai.cc duel_recorder_ai.cc core.h)
kurumi_add_executable(supervised_learn supervised_learn.cc core.h stochastic_decision_tree.cc supervised_stochastic_decision_tree.cc feature_extractor.h)
kurumi_add_executable(reinforce_kurumi reinforce_kurumi_ai_main.cc reinforce_kurumi_ai.cc core.h stochastic_decision_tree.cc reinforce_stochastic_decision_tree.cc feature_extractor.h logistic_regressor.h)
kurumi_add_executable(supervised_learn_logistic supervised_learn_logistic.cc neural_decision_tree.cc supervised_stochastic_decision_tree.cc stochastic_decision_tree.cc core.h feature_extractor.h)

cpu_add_runner(run.sh)
