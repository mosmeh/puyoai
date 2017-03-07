#ifndef DFKURUMI_STOCHASTIC_DECISION_TREE_H_
#define DFKURUMI_STOCHASTIC_DECISION_TREE_H_

#include <vector>
#include <array>
#include <valarray>
#include <random>
#include <fstream>
#include <future>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/client/ai/ai.h"

#include "core.h"
#include "feature_extractor.h"


namespace kurumi {

// This is based on Shallow Neural Decision Forest in [Kontschieder et al., 2015]. See http://research.microsoft.com/apps/pubs/?id=255952

class StochasticDecisionTree {
public:
    StochasticDecisionTree(int depth, float alpha, float beta, std::mt19937& randomEngine);

    std::pair<Action, std::valarray<float> > selectAction(const State& state) const;
    void update(const std::vector<std::pair<State, Action> >& transitions, const int reward);

    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    const int depth_;
    const float alpha_, beta_;

    const float EPS = 1e-7;
    const int LEAF_NODES = std::pow(2, depth_);
    const int SPLIT_NODES = std::pow(2, depth_) - 1;
    const int NODES = LEAF_NODES + SPLIT_NODES;

    std::mt19937& randomEngine_;
    std::valarray<float> weights_, biases_, distributions_;

    float sigmoid(const float x) const;
    float decisionFunction(const std::valarray<float>& features, const int n) const;
    std::valarray<float> pi(const int n) const;
};

}

#endif
