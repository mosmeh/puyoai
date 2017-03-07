#ifndef DFKURUMI_STOCHASTIC_DECISION_TREE_H_
#define DFKURUMI_STOCHASTIC_DECISION_TREE_H_

#include <vector>
#include <valarray>
#include <random>
#include <fstream>
#include <sstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core.h"
#include "feature_extractor.h"

namespace kurumi {

// This is based on Shallow Neural Decision Forest in [Kontschieder et al., 2015]. See http://research.microsoft.com/apps/pubs/?id=255952

class StochasticDecisionTree {
public:
    StochasticDecisionTree(const int depth, std::mt19937& randomEngine, const DTYPE etaWeights);

    void updateWeights(const std::vector<std::pair<State, int> >& transitions);
    void updateDists(const std::vector<std::pair<State, int> >& transitions);

    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    const int depth_;
    DTYPE etaWeights_;

    const DTYPE EPS = 1e-7;
    const int LEAF_NODES = std::pow(2, depth_);
    const int SPLIT_NODES = std::pow(2, depth_) - 1;
    const int NODES = LEAF_NODES + SPLIT_NODES;
    const size_t MEMORY_SIZE = FEATURES * SPLIT_NODES + LEAF_NODES * NUM_DECISIONS;

    std::valarray<DTYPE> weights_, distributions_;

    int leftNode(const int index);
    int rightNode(const int index);
    DTYPE sigmoid(const DTYPE x) const;
    DTYPE decisionFunction(const std::valarray<DTYPE>& features, const int n) const;
    std::valarray<DTYPE> toValArray(std::slice_array<DTYPE> ary) const;
};

}

#endif
