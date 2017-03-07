#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core.h"
#include "supervised_stochastic_decision_tree.h"

DEFINE_int32(seed, 1, "");
DEFINE_string(train_record, "", "record used in training");
DEFINE_string(test_record, "", "record used in testing");
DEFINE_int32(depth, 12, "depth of decision tree");
DEFINE_double(base_lr, 0.001, "learning rate");
DEFINE_double(gamma, 0.1, "decay of learning rate");
DEFINE_int32(step_size, 500, "gamma is applied every step_size iterations");
//DEFINE_double(momentum, 0.9, "momentum of momentum SGD");
DEFINE_int32(batch_size, 1000, "batch size");
DEFINE_int32(snapshot, 1000, "take snapshot every snapshot iterations");
DEFINE_int32(test_interval, 10, "test every test_interval iterations");
DEFINE_string(model, "", "parameter fi;e");
DEFINE_int32(resume_iter, 0, "iterations at which iterations are resumed");
DEFINE_int32(take, 0, "take first take transitions from record");
DEFINE_double(lambda, 0, "");
DEFINE_bool(train, true, "whether to train");

const int TOP_K = 5;

using namespace kurumi;

std::vector<std::pair<State, int> > loadRecord(const std::string& filename) {
    std::vector<std::pair<State, int> > transitions;
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    for (int i = 0; FLAGS_take > 0 ? i < FLAGS_take : !ifs.eof(); ++i) {
    //while (!ifs.eof()) {
        char decisionId;
        State state;

        ifs.read(&decisionId, sizeof(char));

        ifs.read(reinterpret_cast<char*>(&state.frameId), sizeof(int));

        ifs.read(reinterpret_cast<char*>(&state.me.hand), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.field), sizeof(CoreField));
        ifs.read(reinterpret_cast<char*>(&state.me.hasZenkeshi), sizeof(bool));
        ifs.read(reinterpret_cast<char*>(&state.me.fixedOjama), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.pendingOjama), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.unusedScore), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.currentChain), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.currentChainStartedFrameId), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.me.currentRensaResult), sizeof(RensaResult));
        ifs.read(reinterpret_cast<char*>(&state.me.fieldWhenGrounded), sizeof(CoreField));
        ifs.read(reinterpret_cast<char*>(&state.me.hasOjamaDropped), sizeof(bool));
        {
            Kumipuyo ka, kb;
            ifs.read(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&ka)), sizeof(Kumipuyo));
            ifs.read(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&kb)), sizeof(Kumipuyo));
            state.me.seq = KumipuyoSeq({ka, kb});
        }

        ifs.read(reinterpret_cast<char*>(&state.enemy.hand), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.field), sizeof(CoreField));
        ifs.read(reinterpret_cast<char*>(&state.enemy.hasZenkeshi), sizeof(bool));
        ifs.read(reinterpret_cast<char*>(&state.enemy.fixedOjama), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.pendingOjama), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.unusedScore), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.currentChain), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.currentChainStartedFrameId), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&state.enemy.currentRensaResult), sizeof(RensaResult));
        ifs.read(reinterpret_cast<char*>(&state.enemy.fieldWhenGrounded), sizeof(CoreField));
        ifs.read(reinterpret_cast<char*>(&state.enemy.hasOjamaDropped), sizeof(bool));
        {
            Kumipuyo ka, kb;
            ifs.read(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&ka)), sizeof(Kumipuyo));
            ifs.read(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&kb)), sizeof(Kumipuyo));
            state.enemy.seq = KumipuyoSeq({ka, kb});
        }

        transitions.emplace_back(std::move(state), decisionId);
    }
    transitions.erase(transitions.begin());
    transitions.pop_back();

    return std::move(transitions);
}

std::tuple<DTYPE, DTYPE, DTYPE> evaluate(const SupervisedStochasticDecisionTree& tree, const std::vector<std::pair<State, int> >& transitions) {
      DTYPE loss = 0, accuracy = 0, topkAccuracy = 0;
#pragma omp parallel for
      for (unsigned int j = 0; j < transitions.size(); ++j) {
          std::valarray<DTYPE> target(0.0, NUM_DECISIONS);
          target[transitions[j].second] = 1;
          auto pred = tree.predictDeterminately(transitions[j].first);

          const std::valarray<DTYPE> r = pred - target;
          loss += (r * r).sum();

          const int m = std::distance(std::begin(pred), std::max_element(std::begin(pred), std::end(pred)));
          accuracy += static_cast<DTYPE>(transitions[j].second == m);

          const DTYPE v = pred[transitions[j].second];
          std::partial_sort(std::begin(pred), std::begin(pred) + TOP_K, std::end(pred), std::greater<DTYPE>());
          topkAccuracy += static_cast<DTYPE>(std::find(std::begin(pred), std::begin(pred) + TOP_K, v) != std::begin(pred) + TOP_K);
      }

      return std::make_tuple(loss / transitions.size(), accuracy / transitions.size(), topkAccuracy / transitions.size());
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    google::InstallFailureSignalHandler();
#endif

    std::mt19937 randomEngine(FLAGS_seed);

    std::cout << "loading training record" << std::endl;
    auto trainTransitions = loadRecord(std::string(FLAGS_train_record));
    std::cout << "#trainTransitions = " << trainTransitions.size() << std::endl;

    std::cout << "loading test record" << std::endl;
    auto testTransitions = loadRecord(std::string(FLAGS_test_record));
    std::cout << "#testTransitions = " << testTransitions.size() << std::endl;

    if (FLAGS_model.empty()) {
        std::cout << "constructing deicison tree" << std::endl;
    }
    SupervisedStochasticDecisionTree tree(FLAGS_depth, FLAGS_lambda, randomEngine);
    if (!FLAGS_model.empty()) {
        std::cout << "loading model" << std::endl;
        tree.loadModel(FLAGS_model);
    }

    std::ofstream log("log.txt", std::ios::out);
    DTYPE eta = FLAGS_base_lr;
    for (int i = FLAGS_resume_iter + 1;; ++i) {
        std::cout << "starting epoch #" << i << std::endl;
        std::shuffle(trainTransitions.begin(), trainTransitions.end(), randomEngine);

        if (FLAGS_train) {
            std::cout << "training, depth = " << FLAGS_depth << ", eta = " << eta << ", batch_size = " << FLAGS_batch_size << std::endl;
            tree.update(trainTransitions);
        }

        if (i % FLAGS_test_interval == 0) {
            std::cout << "testing" << std::endl;
            DTYPE loss, accuracy, topkAccuracy;
            std::tie(loss, accuracy, topkAccuracy) = evaluate(tree, trainTransitions);
            std::cout << "train set: euclidean loss = " << loss << ", accuracy = " << accuracy << ", top-" << TOP_K << " accuracy = " << topkAccuracy << std::endl;
            log << i << " " << accuracy << " ";
            std::tie(loss, accuracy, topkAccuracy) = evaluate(tree, testTransitions);
            std::cout << "test set: euclidean loss = " << loss << ", accuracy = " << accuracy << ", top-" << TOP_K << " accuracy = " << topkAccuracy << std::endl;
            log << accuracy << std::endl;
        }

        if (i % FLAGS_step_size == 0) {
            eta *= FLAGS_gamma;
        }

        if (i % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "l1iter" << i << "seed" << FLAGS_seed << "depth" << FLAGS_depth << "base_lr" << FLAGS_base_lr << "gamma" << FLAGS_gamma << "step_size" << FLAGS_step_size << "batch_size" << FLAGS_batch_size << ".model";
            std::cout << "saving" << std::endl;
            tree.saveModel(ss.str());
        }
    }

    log.close();

    return 0;
}
