#ifndef DFKURUMI_STOCHASTIC_DECISION_TREE_H_
#define DFKURUMI_STOCHASTIC_DECISION_TREE_H_

#include <vector>
#include <array>
#include <valarray>
#include <random>
#include <fstream>
#include <sstream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "base/base.h"
#include "core/client/ai/ai.h"

#include "core.h"
#include "feature_extractor.h"

namespace kurumi {

using namespace boost::interprocess;

// This is based on Shallow Neural Decision Forest in [Kontschieder et al., 2015]. See http://research.microsoft.com/apps/pubs/?id=255952

class StochasticDecisionTree {
public:
    StochasticDecisionTree(const int depth, std::mt19937& randomEngine);
    ~StochasticDecisionTree();

    std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > selectAction(const State& state) const;
    void update(const std::vector<std::pair<State, Action> >& transitions, const int reward);

    void setUpToLearn(const std::string& id, const DTYPE etaWeights, const DTYPE etaDists, const bool writer);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
    void sync();

private:
    const int depth_;
    DTYPE etaWeights_, etaDists_;

    const DTYPE EPS = 1e-7;
    const int LEAF_NODES = std::pow(2, depth_);
    const int SPLIT_NODES = std::pow(2, depth_) - 1;
    const int NODES = LEAF_NODES + SPLIT_NODES;
    const size_t MEMORY_SIZE = FEATURES * SPLIT_NODES + LEAF_NODES * NUM_DECISIONS;

    std::mt19937& randomEngine_;
    std::valarray<DTYPE> weights_, distributions_;

    std::string id_;

    bool writer_ = false;
    shared_memory_object smo_;
    mapped_region region_;
    DTYPE* sharedMemoryAddr_;

    void writeToSharedMemory();
    void readFromSharedMemory();
    int leftNode(const int index);
    int rightNode(const int index);
    DTYPE sigmoid(const DTYPE x) const;
    DTYPE decisionFunction(const std::valarray<DTYPE>& features, const int n) const;
    std::valarray<DTYPE> pi(const int n) const;
    std::valarray<DTYPE> toValArray(std::slice_array<DTYPE> ary) const;
};

}

#endif
