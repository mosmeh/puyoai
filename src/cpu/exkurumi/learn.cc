#include <iostream>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core.h"
#include "stochastic_decision_tree.h"

DEFINE_int32(seed, 1, "");
DEFINE_string(record, "", "");
DEFINE_int32(depth, 12, "");
DEFINE_double(eta, 0.01, "");
DEFINE_int32(batch_size, 5000, "");
DEFINE_int32(snapshot, 1000, "");

using namespace kurumi;

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    google::InstallFailureSignalHandler();
#endif

    std::mt19937 randomEngine(FLAGS_seed);

    std::cout << "loading record" << std::endl;
    std::vector<std::pair<State, int> > transitions;
    std::ifstream ifs(FLAGS_record, std::ios::in | std::ios::binary);
    while (!ifs.eof()) {
        char decisionId;
        State state;
        ifs.read(&decisionId, sizeof(char));
        ifs.read(reinterpret_cast<char*>(&state), sizeof(State));
        transitions.emplace_back(std::make_pair(state, static_cast<int>(decisionId)));
    }

    std::cout << "constructing deicison tree" << std::endl;
    StochasticDecisionTree tree(FLAGS_depth, randomEngine, FLAGS_eta);

    for (int i = 0;; ++i) {
        std::cout << "starting epoch " << i << std::endl;
        std::shuffle(transitions.begin(), transitions.end(), randomEngine);
        tree.updateDists(transitions);
        tree.updateWeights(transitions);

        std::cout << "testing" << std::endl;

    }

    return 0;
}
