#include "neural_decision_tree.h"

namespace kurumi {

NeuralDecisionTree::NeuralDecisionTree(std::mt19937& randomEngine, const std::string& solverParamFile) : randomEngine_(randomEngine), distributions_(1.0 / NUM_DECISIONS, LEAF_NODES * NUM_DECISIONS) {
    caffe::SolverParameter solverParam;
    caffe::ReadProtoFromTextFileOrDie(solverParamFile, &solverParam);
    solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solverParam));
    net_ = solver_->net();

    std::fill(inputDummyData_.begin(), inputDummyData_.end(), 0);
    std::fill(gradDummyData_.begin(), gradDummyData_.end(), 0);
    std::fill(prevGrad_.begin(), prevGrad_.end(), 0);
    std::fill(delta_.begin(), delta_.end(), 1);
    std::fill(deltaW_.begin(), deltaW_.end(), 0);

    valuesBlob_ = net_->blob_by_name("values");
    inputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            net_->layer_by_name("input_layer"));
    gradInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
            net_->layer_by_name("grad_input_layer"));
}

void NeuralDecisionTree::loadModel(const std::string& filename) {
    net_->CopyTrainedLayersFrom(filename);
}

void NeuralDecisionTree::saveModel(const std::string& filename) const {
    caffe::NetParameter netParam;
    net_->ToProto(&netParam);
    caffe::WriteProtoToBinaryFile(netParam, filename);
}

int NeuralDecisionTree::leftNode(const int index) const {
    return 2 * index + 1;
}

int NeuralDecisionTree::rightNode(const int index) const {
    return 2 * index + 2;
}

std::valarray<DTYPE> NeuralDecisionTree::pi(const int n) const {
    return distributions_[std::slice(n * NUM_DECISIONS, NUM_DECISIONS, 1)];
}

DTYPE NeuralDecisionTree::sigmoid(DTYPE x) const {
    return 1.0 / (1 + std::exp(-x));
}

std::vector<std::valarray<DTYPE> > NeuralDecisionTree::getValues(const std::vector<std::pair<State, int> >& instances) const {
    std::vector<std::valarray<DTYPE> > values(instances.size(), std::valarray<DTYPE>(SPLIT_NODES));

    std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
    for (unsigned int t_i = 0; t_i < instances.size(); t_i += MAX_MINIBATCH_SIZE) {
        int minibatchSize = std::min(static_cast<int>(instances.size() - t_i), MAX_MINIBATCH_SIZE);

        std::fill(input.begin(), input.end(), 0);
        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            const auto features = FeatureExtractor::extract(instances[t_i + m_i].first);
            std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);
        }
        inputDataIntoLayers(input, gradDummyData_);
        net_->Forward();
        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            for (int i = 0; i < SPLIT_NODES; ++i) {
                values[t_i + m_i][i] = valuesBlob_->data_at(m_i, i, 0, 0);
            }
        }
    }

    return std::move(values);
}

std::vector<std::valarray<DTYPE> > NeuralDecisionTree::predict(const std::vector<std::pair<State, int> >& instances) const {
    const auto values = getValues(instances);

    std::vector<std::valarray<DTYPE> > results(instances.size(), std::valarray<DTYPE>(NUM_DECISIONS));;
    results.reserve(instances.size());

    for (unsigned int i = 0; i < instances.size(); ++i) {
        int n = 0;
        DTYPE mu = 1;
        while (n < SPLIT_NODES) {
            const DTYPE d = sigmoid(values[i][n]);
            const bool branchToLeft = std::bernoulli_distribution(d)(randomEngine_);
            mu *= branchToLeft ? d : (1 - d);
            n += n + 1 + (1 - static_cast<int>(branchToLeft));
        }

        results[i] = pi(n - SPLIT_NODES);
    }

    return std::move(results);
}

std::vector<std::valarray<DTYPE> > NeuralDecisionTree::predictDeterminately(const std::vector<std::pair<State, int> >& instances) const {
    const auto values = getValues(instances);

    std::vector<std::valarray<DTYPE> > results(instances.size(), std::valarray<DTYPE>(0.0, NUM_DECISIONS));

    for (unsigned int t = 0; t < instances.size(); ++t) {
        std::valarray<DTYPE> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
          mu[leftNode(i)] = sigmoid(values[t][i]) * mu[i];
          mu[rightNode(i)] = (1 - sigmoid(values[t][i])) * mu[i];
        }

        for (int i = 0; i < LEAF_NODES; ++i) {
          results[t] += pi(i) * mu[i + SPLIT_NODES];
        }
    }

    return std::move(results);
}

void NeuralDecisionTree::update(const std::vector<std::pair<State, int> >& instances) {
    const auto values = getValues(instances);
    std::vector<std::valarray<DTYPE> > mus(instances.size(), std::valarray<DTYPE>(LEAF_NODES));
    for (unsigned int t = 0; t < instances.size(); ++t) {
        std::valarray<DTYPE> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
          mu[leftNode(i)] = sigmoid(values[t][i]) * mu[i];
          mu[rightNode(i)] = (1 - sigmoid(values[t][i])) * mu[i];
        }

        mus[t] = mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)];
    }

    std::valarray<DTYPE> estimatedPi(1.0 / NUM_DECISIONS, NUM_DECISIONS * LEAF_NODES);
    for (int l = 0; l < 20; ++l) {
        std::valarray<DTYPE> tempPi(0.0, NUM_DECISIONS * LEAF_NODES);
        for (unsigned int i = 0; i < instances.size(); ++i) {
            const int y = instances[i].second;
            const std::valarray<DTYPE> yProbs = mus[i] * toValArray(estimatedPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);
            tempPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)] += yProbs / (yProbs.sum() + EPS);
        }

        DTYPE diff = 0;
        for (int i = 0; i < LEAF_NODES; ++i) {
            std::valarray<DTYPE> dist = tempPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)];
            dist /= (dist.sum() + EPS);
            diff += std::abs(toValArray(estimatedPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)]) - dist).sum();
            estimatedPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)] = dist;
        }

        if (diff < 1e-5) {
            break;
        }
    }

    distributions_ = estimatedPi;

    /*std::array<DTYPE, SPLIT_NODES> avgGrad;
    std::fill(avgGrad.begin(), avgGrad.end(), 0);

    for (unsigned int t_i = 0; t_i < instances.size(); ++t_i) {
        const int y = instances[t_i].second;
        const std::valarray<DTYPE> yProbs = mus[t_i] * toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);
        std::valarray<DTYPE> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = yProbs / (yProbs.sum() + EPS);
        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
            a[i] = a[leftNode(i)] + a[rightNode(i)];
            avgGrad[i] += (sigmoid(values[t_i][i]) * a[rightNode(i)] - (1 - sigmoid(values[t_i][i])) * a[leftNode(i)]) / instances.size();
        }
    }

    std::array<DTYPE, SPLIT_NODES * MAX_MINIBATCH_SIZE> grad;
    std::fill(grad.begin(), grad.end(), 0);

    static const DTYPE etaPlus = 1.2, etaMinus = 0.5, deltaMax = 50, deltaMin = 1e-6;
    DTYPE sumDeltaSq = 0;
    for (int i = 0; i < SPLIT_NODES; ++i) {
        if (prevGrad_[i] * avgGrad[i] > 0) {
            prevGrad_[i] = avgGrad[i];
            delta_[i] = std::min(delta_[i] * etaPlus, deltaMax);
            deltaW_[i] = (avgGrad[i] > 0 ? 1 : -1) * delta_[i];
        } else if (prevGrad_[i] * avgGrad[i] < 0) {
            prevGrad_[i] = 0;
            delta_[i] = std::max(delta_[i] * etaMinus, deltaMin);
            deltaW_[i] *= -1;
        } else {
            prevGrad_[i] = avgGrad[i];
            DTYPE sign = 0;
            if (avgGrad[i] > 0) {
                sign = 1;
            } else if (avgGrad[i] < 0) {
                sign = -1;
            }
            deltaW_[i] = sign * delta_[i];
        }
        sumDeltaSq += delta_[i] * delta_[i];
        avgGrad[i] = deltaW_[i];
    }
    std::cout << "|delta| = " << std::sqrt(sumDeltaSq) << std::endl;

    std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
    for (unsigned int t_i = 0; t_i < instances.size(); t_i += MAX_MINIBATCH_SIZE) {
        int minibatchSize = std::min(static_cast<int>(instances.size() - t_i), MAX_MINIBATCH_SIZE);
        std::fill(input.begin(), input.end(), 0);
        std::fill(grad.begin(), grad.end(), 0);

        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            const auto features = FeatureExtractor::extract(instances[t_i + m_i].first);
            std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);
            std::copy(avgGrad.begin(), avgGrad.end(), grad.begin() + m_i * SPLIT_NODES);
        }

        inputDataIntoLayers(input, grad);
        solver_->Step(1);
    }*/
    std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE> input;
    std::array<DTYPE, SPLIT_NODES * MAX_MINIBATCH_SIZE> grad;

    for (unsigned int t_i = 0; t_i < instances.size(); t_i += MAX_MINIBATCH_SIZE) {
        int minibatchSize = std::min(static_cast<int>(instances.size() - t_i), MAX_MINIBATCH_SIZE);
        std::fill(input.begin(), input.end(), 0);
        std::fill(grad.begin(), grad.end(), 0);

        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            const auto features = FeatureExtractor::extract(instances[t_i + m_i].first);
            std::copy(std::begin(features), std::end(features), input.begin() + m_i * FEATURES);

            const int y = instances[t_i + m_i].second;
            const std::valarray<DTYPE> yProbs = mus[t_i] * toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);
            std::valarray<DTYPE> a(NODES);
            a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = yProbs / (yProbs.sum() + EPS);
            for (int i = SPLIT_NODES - 1; i >= 0; --i) {
                a[i] = a[leftNode(i)] + a[rightNode(i)];
                grad[i + m_i * SPLIT_NODES] += (sigmoid(values[t_i + m_i][i]) * a[rightNode(i)] - (1 - sigmoid(values[t_i + m_i][i])) * a[leftNode(i)]) / minibatchSize;
            }
        }

        inputDataIntoLayers(input, grad);
        solver_->Step(1);
    }
}

void NeuralDecisionTree::inputDataIntoLayers(const std::array<DTYPE, FEATURES * MAX_MINIBATCH_SIZE>& input, const std::array<DTYPE, SPLIT_NODES * MAX_MINIBATCH_SIZE>& grad) const {
    inputLayer_->Reset(const_cast<float*>(input.data()), const_cast<float*>(inputDummyData_.data()), MAX_MINIBATCH_SIZE);
    gradInputLayer_->Reset(const_cast<float*>(grad.data()), const_cast<float*>(gradDummyData_.data()), MAX_MINIBATCH_SIZE);
}

std::valarray<DTYPE> NeuralDecisionTree::toValArray(const std::slice_array<DTYPE>& ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

}
