#ifndef KURUMI_STOCHASTIC_DECISION_TREE_H_
#define KURUMI_STOCHASTIC_DECISION_TREE_H_

#include <vector>
#include <valarray>
#include <random>
#include <fstream>

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
    StochasticDecisionTree(const int depth, std::mt19937& randomEngine);

    std::tuple<int, DTYPE, std::valarray<DTYPE> > predict(const State& state) const;
    std::valarray<DTYPE> predictDeterminately(const State& state) const;
    std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > selectAction(const State& state) const;
    std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > selectActionGreedily(const State& state) const;
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

protected:
    const int depth_;

    const int LEAF_NODES = std::pow(2, depth_);
    const int SPLIT_NODES = std::pow(2, depth_) - 1;
    const int NODES = LEAF_NODES + SPLIT_NODES;

    std::mt19937& randomEngine_;
    std::valarray<DTYPE> weights_, distributions_;

    DTYPE sigmoid(const DTYPE x) const;
    DTYPE decisionFunction(const std::valarray<DTYPE>& features, const int n) const;
    std::valarray<DTYPE> pi(const int n) const;
};

}

#endif

