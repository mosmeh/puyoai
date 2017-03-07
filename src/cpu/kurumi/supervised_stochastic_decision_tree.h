#ifndef KURUMI_SUPERVISED_STOCHASTIC_DECISION_TREE_H_
#define KURUMI_SUPERVISED_STOCHASTIC_DECISION_TREE_H_

#include "stochastic_decision_tree.h"

namespace kurumi {

class SupervisedStochasticDecisionTree : public StochasticDecisionTree {
public:
    SupervisedStochasticDecisionTree(const int depth, const DTYPE lambda, std::mt19937& randomEngine);

    void update(const std::vector<std::pair<State, int> >& transitions);

private:
    const DTYPE EPS = 1e-7;
    DTYPE lambda_;
    std::valarray<DTYPE> prevdWeights_, delta_, deltaW_;

    int leftNode(const int index);
    int rightNode(const int index);
    std::valarray<DTYPE> toValArray(std::slice_array<DTYPE> ary) const;
    std::valarray<DTYPE> toValArray(std::mask_array<DTYPE> ary) const;
};

}

#endif

