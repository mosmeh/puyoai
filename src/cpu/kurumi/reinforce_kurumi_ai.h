#ifndef KURUMI_REINFORCE_KURUMI_AI_H_
#define KURUMI_REINFORCE_KURUMI_AI_H_

#include <sstream>
#include <fstream>
#include <random>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"
#include "core/pattern/decision_book.h"

#include "reinforce_stochastic_decision_tree.h"

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/optional.hpp>

namespace kurumi {

const int MAX_MINIBATCH_SIZE = 100;

class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(const std::string& solverParamFile, std::mt19937& randomEngine) : randomEngine_(randomEngine){
        caffe::SolverParameter solverParam;
        caffe::ReadProtoFromTextFileOrDie(solverParamFile, &solverParam);
        solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solverParam));
        net_ = solver_->net();

        std::fill(dummyInputData_.begin(), dummyInputData_.end(), 0);
        std::fill(dummyGradData_.begin(), dummyGradData_.end(), 0);

        probsBlob_ = net_->blob_by_name("probs");
        inputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                net_->layer_by_name("input_layer"));
        gradInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
                net_->layer_by_name("grad_input_layer"));
    }

    std::valarray<DTYPE> predict(const State& state) {
        std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
        std::fill(input.begin(), input.end(), 0);
        const auto features = FeatureExtractor::extract(state);
        std::copy(std::begin(features), std::end(features), input.begin());

        inputDataIntoLayers(input, dummyGradData_);
        net_->Forward();

        std::valarray<DTYPE> dist(NUM_DECISIONS);
        for (int i = 0; i < NUM_DECISIONS; ++i) {
            dist[i] = probsBlob_->data_at(0, i, 0, 0);
        }

        return dist * state.legalActions;
    }

    std::pair<Action, std::valarray<DTYPE> > selectAction(const State& state) {
        const auto dist = predict(state);
        const int id = std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);
        return std::make_pair(Action(id), dist);
    }

    std::pair<Action, std::valarray<DTYPE> > selectAction(const State& state) const {
        return const_cast<MultiLayerPerceptron*>(this)->selectAction(state);
    }

    void update(const std::vector<std::pair<State, int> >& transitions, const DTYPE reward) {
        std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
        std::array<DTYPE, NUM_DECISIONS * MAX_MINIBATCH_SIZE> grad;

        for (unsigned int t_i = 0; t_i < transitions.size(); t_i += MAX_MINIBATCH_SIZE) {
            int minibatchSize = std::min(static_cast<int>(transitions.size() - t_i), MAX_MINIBATCH_SIZE);

            std::fill(input.begin(), input.end(), 0);
            std::fill(grad.begin(), grad.end(), 0);
            for (int m_i = 0; m_i < minibatchSize; ++m_i) {
                const auto features = FeatureExtractor::extract(transitions[t_i + m_i].first);
                std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);
            }

            inputDataIntoLayers(input, grad);
            net_->Forward();

            std::fill(grad.begin(), grad.end(), 0);
            for (int m_i = 0; m_i < minibatchSize; ++m_i) {
                const int y = transitions[t_i + m_i].second;
                const DTYPE prob = probsBlob_->data_at(m_i, y, 0, 0);
                grad[y + m_i * NUM_DECISIONS] = reward / prob;
            }

            inputDataIntoLayers(input, grad);

            solver_->Step(1);
        }
    }

    void loadModel(const std::string& filename) {
        net_->CopyTrainedLayersFrom(filename);
    }

    void saveModel(const std::string& filename) const {
        caffe::NetParameter netParam;
        net_->ToProto(&netParam);
        caffe::WriteProtoToBinaryFile(netParam, filename);
    }


private:
    std::shared_ptr<caffe::Solver<float> > solver_;
    boost::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::Blob<float> > probsBlob_;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > inputLayer_, gradInputLayer_;

    std::mt19937& randomEngine_;

    std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> dummyInputData_;
    std::array<DTYPE, NUM_DECISIONS * MAX_MINIBATCH_SIZE> dummyGradData_;

    void inputDataIntoLayers(const std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE>& input, const std::array<DTYPE, NUM_DECISIONS * MAX_MINIBATCH_SIZE>& grad) {
        std::cerr << "input" << std::endl;
        inputLayer_->Reset(const_cast<float*>(input.data()), const_cast<float*>(dummyInputData_.data()), MAX_MINIBATCH_SIZE);
        std::cerr << "grad" << std::endl;
        gradInputLayer_->Reset(const_cast<float*>(grad.data()), const_cast<float*>(dummyGradData_.data()), MAX_MINIBATCH_SIZE);
    }
};

class ReinforceKurumiAI : public AI {
public:
    ReinforceKurumiAI(int argc, char* argv[]);

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override;
    void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq);

private:
    int games_;
    KumipuyoSeq enemySeq_;
    std::mt19937 randomEngine_;
    //ReinforceStochasticDecisionTree tree_;
    MultiLayerPerceptron mlp_;
    std::ofstream ofs_;
    DecisionBook decisionBook_;
    std::vector<std::pair<State, int> > transitions_;
    int maxScore = 0, sumScore = 0, wins = 0, numDecisions = 0;
    DTYPE sumMu = 0, sumEntropy = 0;
    std::valarray<DTYPE> leafCount;

    void onGameHasEnded(const FrameRequest& req);
    int gameResultToInt(const GameResult result) const;
    void storeTransition(const std::pair<State, int>& trans) const;
    void storeTransition(const std::pair<State, int>& trans);
    void updateStats(const DTYPE mu, const DTYPE pentropy, const int leafId) const;
    void updateStats(const DTYPE mu, const DTYPE pentropy, const int leafId);
};

}

#endif
