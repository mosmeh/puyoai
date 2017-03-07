#ifndef KURUMI_REINFORCE_STOCHASTIC_DECISION_TREE_H_
#define KURUMI_REINFORCE_STOCHASTIC_DECISION_TREE_H_

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "stochastic_decision_tree.h"

namespace kurumi {

using namespace boost::interprocess;

class ReinforceStochasticDecisionTree : public StochasticDecisionTree {
public:
    ReinforceStochasticDecisionTree(const int depth, std::mt19937& randomEngine, const std::string& id, const DTYPE etaWeights, const DTYPE etaDists, const DTYPE beta, const int minibatchSize, const bool writer);
    ~ReinforceStochasticDecisionTree();

    void update(const std::vector<std::pair<State, int> >& transitions, const DTYPE reward);

    void writeToSharedMemory();
    void readFromSharedMemory();

private:
    const DTYPE etaWeights_, etaDists_, beta_;
    const int minibatchSize_;

    const DTYPE EPS = 1e-7;
    const size_t MEMORY_SIZE = FEATURES * SPLIT_NODES + LEAF_NODES * NUM_DECISIONS;

    const std::string id_;
    const bool writer_;
    shared_memory_object smo_;
    mapped_region region_;
    DTYPE* sharedMemoryAddr_;

    std::vector<std::pair<std::valarray<DTYPE>, std::valarray<DTYPE> > > pool_;

    std::valarray<DTYPE> mdWeights_,// prevdWeights_, deltaWeights_, deltaW_,
                         mdDists_;//, prevdDists_, deltaDists_, deltaD_;
    int count_ = 0;

    int leftNode(const int index);
    int rightNode(const int index);
    std::valarray<DTYPE> toValArray(std::slice_array<DTYPE> ary) const;
};

}

#endif

