#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <array>
#include <memory>

/*#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/optional.hpp>*/

#include "core.h"
#include "supervised_stochastic_decision_tree.h"
#include "feature_extractor.h"
//#include "neural_decision_tree.h"

DEFINE_int32(seed, 1, "");
DEFINE_int32(depth, 10, "");
DEFINE_double(lambda, 0.01, "");
DEFINE_string(train_record, "", "record used in training");
DEFINE_string(test_record, "", "record used in testing");
//DEFINE_int32(num_trees, 50, "");
//DEFINE_int32(step_size, 500, "gamma is applied every step_size iterations");
//DEFINE_int32(batch_size, 1000, "batch size");
DEFINE_int32(snapshot, 1000, "take snapshot every snapshot iterations");
DEFINE_int32(test_interval, 10, "test every test_interval iterations");
DEFINE_string(model, "", "parameter file");
DEFINE_int32(resume_iter, 0, "iterations at which iterations are resumed");
DEFINE_int32(take, 0, "take first take transitions from record");
DEFINE_bool(train, true, "whether to train");
DEFINE_bool(append_log, false, "");
//DEFINE_string(solver, "solver.prototxt", "");

const int TOP_K = 5;

using namespace kurumi;

/*class LogisticRegressor {
public:
    LogisticRegressor(DTYPE eta, std::mt19937& randomEngine) : eta_(eta),
    weights_(NUM_DECISIONS * FEATURES),
    prevdWeights_(0.0, NUM_DECISIONS * FEATURES),
    deltaW_(0.0, NUM_DECISIONS * FEATURES),
    delta_(0.01, NUM_DECISIONS * FEATURES), randomEngine_(randomEngine) {
        std::normal_distribution<DTYPE> normal(0, 0.01);
        std::generate(std::begin(weights_), std::end(weights_), [&]{ return normal(randomEngine_); });
    }

    std::valarray<DTYPE> predict(const State& state) const {
        const auto features = FeatureExtractor::extract(state);
        std::valarray<DTYPE> values(NUM_DECISIONS);
        for (int i = 0; i < NUM_DECISIONS; ++i) {
            values[i] = (static_cast<std::valarray<DTYPE> >(weights_[std::slice(i * FEATURES, FEATURES, 1)]) * features).sum();
        }
        std::valarray<DTYPE> probs = std::exp(values);
        probs /= probs.sum();
        return probs;
    }

    Action selectAction(const State& state) const {
        const std::valarray<DTYPE> dist = predict(state) * state.legalActions;
        return Action(std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_));
    }

    void train(const std::vector<std::pair<State, int> >& transitions) {
        std::valarray<DTYPE> dWeights(0.0, NUM_DECISIONS * FEATURES);
#pragma omp parallel for
        for (unsigned int t = 0; t < transitions.size(); ++t) {
            const auto features = FeatureExtractor::extract(transitions[t].first);
            const int y = transitions[t].second;

            std::valarray<DTYPE> values(NUM_DECISIONS);
            for (int i = 0; i < NUM_DECISIONS; ++i) {
                values[i] = (static_cast<std::valarray<DTYPE> >(weights_[std::slice(i * FEATURES, FEATURES, 1)]) * features).sum();
            }
            std::valarray<DTYPE> dist = std::exp(values);
            dist /= dist.sum();

            dWeights[std::slice(y * FEATURES, FEATURES, 1)] += features;
            for (int i = 0; i < NUM_DECISIONS; ++i) {
                dWeights[std::slice(i * FEATURES, FEATURES, 1)] -= features * dist[i];
            }
        }

        static const DTYPE etaPlus = 1.2, etaMinus = 0.5, deltaMax = 50, deltaMin = 1e-6;
#pragma omp parallel for
        for (int i = 0; i < FEATURES * NUM_DECISIONS; ++i) {
            dWeights[i] /= transitions.size();
            if (prevdWeights_[i] * dWeights[i] > 0) {
                prevdWeights_[i] = dWeights[i];
                delta_[i] = std::min(delta_[i] * etaPlus, deltaMax);
                deltaW_[i] = (dWeights[i] > 0 ? 1 : -1) * delta_[i];
            } else if (prevdWeights_[i] * dWeights[i] < 0) {
                prevdWeights_[i] = 0;
                delta_[i] = std::max(delta_[i] * etaMinus, deltaMin);
                deltaW_[i] *= -1;
            } else {
                prevdWeights_[i] = dWeights[i];
                DTYPE sign = 0;
                if (dWeights[i] > 0) {
                    sign = 1;
                } else if (dWeights[i] < 0) {
                    sign = -1;
                }
                deltaW_[i] = sign * delta_[i];
            }
        }

        weights_ += deltaW_;
    }

    void saveModel(const std::string& filename) {
        std::ofstream ofs;
        ofs.open(filename.c_str(), std::ios::out | std::ios::binary);
#define WRITE(PARAM) for (auto& x : PARAM) ofs.write(reinterpret_cast<char*>(&x), sizeof(DTYPE));
        WRITE(weights_)
        WRITE(prevdWeights_)
        WRITE(deltaW_)
        WRITE(delta_)
#undef WRITE
        ofs.close();
    }


private:
    const DTYPE eta_;
    std::valarray<DTYPE> weights_, prevdWeights_, deltaW_, delta_;
    std::mt19937& randomEngine_;
};A*/

/*const int MAX_MINIBATCH_SIZE = 128;
class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(const std::string& solverParamFile) {
        caffe::SolverParameter solverParam;
        caffe::ReadProtoFromTextFileOrDie(solverParamFile, &solverParam);
        solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solverParam));
        net_ = solver_->net();
        probsBlob_ = net_->blob_by_name("probs");
        inputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                net_->layer_by_name("input_layer"));
    }

    std::vector<std::valarray<DTYPE> > predictDeterminately(const std::vector<std::pair<State, int> >& transitions) const {
        std::vector<std::valarray<DTYPE> > dists;
        dists.reserve(transitions.size());

        std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
        std::array<DTYPE, MAX_MINIBATCH_SIZE> dummy;
        std::fill(dummy.begin(), dummy.end(), 0);

        for (unsigned int t_i = 0; t_i < transitions.size(); t_i += MAX_MINIBATCH_SIZE) {
            std::fill(input.begin(), input.end(), 0);

            int minibatchSize = std::min(static_cast<int>(transitions.size() - t_i), MAX_MINIBATCH_SIZE);

            for (int m_i = 0; m_i < minibatchSize; ++m_i) {
                const auto features = FeatureExtractor::extract(transitions[t_i + m_i].first);
                std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);
            }
            inputDataIntoLayers(input, dummy);
            net_->Forward();
            for (int m_i = 0; m_i < minibatchSize; ++m_i) {
                std::valarray<DTYPE> dist(NUM_DECISIONS);
                for (int i = 0; i < NUM_DECISIONS; ++i) {
                    dist[i] = probsBlob_->data_at(m_i, i, 0, 0);
                }
                dists.emplace_back(dist);
            }
        }

        return std::move(dists);
    }

    void update(const std::vector<std::pair<State, int> >& transitions) {
        for (unsigned int t_i = 0; t_i < transitions.size(); t_i += MAX_MINIBATCH_SIZE) {
            int minibatchSize = std::min(static_cast<int>(transitions.size() - t_i), MAX_MINIBATCH_SIZE);
            std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
            std::array<DTYPE, MAX_MINIBATCH_SIZE> label;
            std::fill(input.begin(), input.end(), 0);
            std::fill(label.begin(), label.end(), 0);

            for (int m_i = 0; m_i < minibatchSize; ++m_i) {
                const auto features = FeatureExtractor::extract(transitions[t_i + m_i].first);
                std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);
                label[m_i] = static_cast<DTYPE>(transitions[t_i + m_i].second);
            }

            inputDataIntoLayers(input, label);
            solver_->Step(1);
        }
    }
private:
    std::shared_ptr<caffe::Solver<float> > solver_;
    boost::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::Blob<float> > probsBlob_;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > inputLayer_;

    void inputDataIntoLayers(const std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE>& input, const std::array<DTYPE, MAX_MINIBATCH_SIZE>& label) const {
        inputLayer_->Reset(const_cast<float*>(input.data()), const_cast<float*>(label.data()), MAX_MINIBATCH_SIZE);
    }
};*/

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

        transitions.emplace_back(std::move(state), Action(decisionId).simplify(state.me.seq.front().isRep()).decisionId);
    }
    transitions.erase(transitions.begin());
    transitions.pop_back();

    return std::move(transitions);
}

std::tuple<DTYPE, DTYPE> evaluate(const SupervisedStochasticDecisionTree& tree, const std::vector<std::pair<State, int> >& transitions) {
      DTYPE loss = 0, error = 0;
#pragma omp parallel for
      for (unsigned int j = 0; j < transitions.size(); ++j) {
          const std::valarray<DTYPE> pred = tree.predictDeterminately(transitions[j].first);
          std::valarray<DTYPE> target(0.0, NUM_DECISIONS);
          target[transitions[j].second] = 1;

          const std::valarray<DTYPE> r = pred - target;
          loss += (r * r).sum();

          const int m = std::distance(std::begin(pred), std::max_element(std::begin(pred), std::end(pred)));
          error += static_cast<DTYPE>(transitions[j].second != Action(m).simplify(transitions[j].first.me.seq.front().isRep()).decisionId);
      }

      return std::make_tuple(loss / transitions.size(), error / transitions.size());
}

/*std::tuple<DTYPE, DTYPE> evaluate(const NeuralDecisionTree& tree, const std::vector<std::pair<State, int> >& transitions) {
    const auto preds = tree.predictDeterminately(transitions);
      DTYPE loss = 0, error = 0;
      for (unsigned int j = 0; j < transitions.size(); ++j) {
          std::valarray<DTYPE> target(0.0, NUM_DECISIONS);
          target[transitions[j].second] = 1;

          const std::valarray<DTYPE> r = preds[j] - target;
          loss += (r * r).sum();

          const int m = std::distance(std::begin(preds[j]), std::max_element(std::begin(preds[j]), std::end(preds[j])));
          error += static_cast<DTYPE>(transitions[j].second != Action(m).simplify(transitions[j].first.me.seq.front().isRep()).decisionId);
      }

      return std::make_tuple(loss / transitions.size(), error / transitions.size());
}*/

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

    SupervisedStochasticDecisionTree tree(FLAGS_depth, FLAGS_lambda, randomEngine);
    //NeuralDecisionTree tree(randomEngine, FLAGS_solver);
    //MultiLayerPerceptron tree(FLAGS_solver);
    if (!FLAGS_model.empty()) {
        tree.loadModel(FLAGS_model);
    }

    std::ofstream log;
    if (FLAGS_append_log) {
        log.open("log.txt", std::ios::out | std::ios::app);
    } else {
        log.open("log.txt", std::ios::out);
    }
    for (int i = FLAGS_resume_iter + 1;; ++i) {
        std::cout << "starting epoch #" << i << std::endl;
        std::shuffle(trainTransitions.begin(), trainTransitions.end(), randomEngine);

        if (FLAGS_train) {
            std::cout << "training" << std::endl;
            tree.update(trainTransitions);
        }

        if (i % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "SDTiter" << i << ".model";
            std::cout << "saving" << std::endl;
            tree.saveModel(ss.str());
        }

        if (i % FLAGS_test_interval == 0) {
            std::cout << "testing" << std::endl;
            DTYPE loss, error;
            std::tie(loss, error) = evaluate(tree, trainTransitions);
            std::cout << "train set: euclidean loss = " << loss << ", error = " << error << std::endl;
            log << i << " " << error << " ";
            std::tie(loss, error) = evaluate(tree, testTransitions);
            std::cout << "test set: euclidean loss = " << loss << ", error = " << error << std::endl;
            log << error << std::endl;
        }
    }

    log.close();

    return 0;
}
