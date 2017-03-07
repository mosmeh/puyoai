#ifndef NEURAL_DECISION_TREE_H_
#define NEURAL_DECISION_TREE_H_

#include <vector>
#include <random>
#include <fstream>
#include <array>
#include <valarray>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/optional.hpp>

#include "base/base.h"
#include "core/client/ai/ai.h"

#include "core.h"
#include "feature_extractor.h"

// This is based on Shallow Neural Decision Forest in [Kontschieder et al., 2015]. See http://research.microsoft.com/apps/pubs/?id=255952


namespace kurumi {

const int MAX_MINIBATCH_SIZE = 70;
const int DEPTH = 10;

class NeuralDecisionTree {
public:
    NeuralDecisionTree(std::mt19937& randomEngine, const std::string& solverParamFile);

    void loadModel(const std::string& filename);
    void saveModel(const std::string& filename) const;

    std::vector<std::valarray<DTYPE> > predict(const std::vector<std::pair<State, int> >& instances) const;
    std::vector<std::valarray<DTYPE> > predictDeterminately(const std::vector<std::pair<State, int> >& instances) const;
    void update(const std::vector<std::pair<State, int> >& instances);

protected:
    const DTYPE EPS = 1e-7;
    static constexpr int LEAF_NODES = std::pow(2, DEPTH);
    static constexpr int SPLIT_NODES = std::pow(2, DEPTH) - 1;
    static constexpr int NODES = LEAF_NODES + SPLIT_NODES;

    std::mt19937& randomEngine_;
    std::valarray<DTYPE> distributions_;
    std::array<DTYPE, SPLIT_NODES> prevGrad_, delta_, deltaW_;

    std::shared_ptr<caffe::Solver<float> > solver_;
    boost::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::Blob<float> > valuesBlob_;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > inputLayer_, gradInputLayer_;

    std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> inputDummyData_;
    std::array<DTYPE, SPLIT_NODES * MAX_MINIBATCH_SIZE> gradDummyData_;

    int leftNode(const int index) const;
    int rightNode(const int index) const;
    DTYPE sigmoid(DTYPE x) const;
    std::valarray<DTYPE> pi(const int n) const;
    std::vector<std::valarray<DTYPE> > getValues(const std::vector<std::pair<State, int> >& instances) const;
    void inputDataIntoLayers(const std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE>& input, const std::array<DTYPE, SPLIT_NODES * MAX_MINIBATCH_SIZE>& grad) const;
    std::valarray<DTYPE> toValArray(const std::slice_array<DTYPE>& ary) const;
};

}

#endif
