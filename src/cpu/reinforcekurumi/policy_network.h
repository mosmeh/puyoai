#ifndef KURUMI_POLICY_NETWORK_H_
#define KURUMI_POLICY_NETWORK_H_

#include <vector>
#include <array>
#include <memory>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/optional.hpp>

#include "base/base.h"
#include "core/plan/plan.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"
#include "core/field_constant.h"
#include "core/puyo_controller.h"

namespace kurumi {

const int MAX_NUM_LEGAL_DECISIONS = 22;

const int FIELD_SIZE = FieldConstant::WIDTH * (FieldConstant::HEIGHT + 1);
const int SINGLE_PLAYER_FIELD_DATA_SIZE = FIELD_SIZE * 5;
const int AUX_DATA_SIZE = (5 * 4 + 3 * 6 + 9) * 2 + MAX_NUM_LEGAL_DECISIONS * 2;

const int MAX_MINIBATCH_SIZE = 32;
const int MINIBATCH_SINGLE_PLAYER_FIELD_DATA_SIZE = SINGLE_PLAYER_FIELD_DATA_SIZE * MAX_MINIBATCH_SIZE;
const int MINIBATCH_AUX_DATA_SIZE = AUX_DATA_SIZE * MAX_MINIBATCH_SIZE;

const Decision DECISIONS[MAX_NUM_LEGAL_DECISIONS] = {
    Decision(2, 3), Decision(3, 3), Decision(3, 1), Decision(4, 1),
    Decision(5, 1), Decision(1, 2), Decision(2, 2), Decision(3, 2),
    Decision(4, 2), Decision(5, 2), Decision(6, 2),
    Decision(1, 1), Decision(2, 1), Decision(4, 3), Decision(5, 3),
    Decision(6, 3), Decision(1, 0), Decision(2, 0), Decision(3, 0),
    Decision(4, 0), Decision(5, 0), Decision(6, 0),
};

using FieldInputData = std::array<float, MINIBATCH_SINGLE_PLAYER_FIELD_DATA_SIZE>;
using AuxInputData = std::array<float, MINIBATCH_AUX_DATA_SIZE>;
using ActionRewardInputData = std::array<float, MAX_MINIBATCH_SIZE>;
using LegalityInputData = std::array<float, MAX_MINIBATCH_SIZE * MAX_NUM_LEGAL_DECISIONS>;

struct Action {
    Decision decision;
    int decisionId;

    Action(int decisionId_) : decisionId(decisionId_) {
        decision = DECISIONS[decisionId];
    }

    Action(Decision decision_) : decision(decision_) {
        decisionId = std::distance(DECISIONS, std::find(DECISIONS, DECISIONS + MAX_NUM_LEGAL_DECISIONS, decision));
    }
};

struct State {
    int frameId;
    PlayerState me;
    PlayerState enemy;
    std::vector<Action> legalActions;

    State(int frameId_, PlayerState me_, PlayerState enemy_) : frameId(frameId_), me(me_), enemy(enemy_) {
        int rep = me.seq.front().isRep() ? 11 : 22;
        for (int i = 0; i < rep; ++i) {
            if (PuyoController::isReachable(me.field, DECISIONS[i])) {
                legalActions.push_back(Action(DECISIONS[i]));
            }
        }
    }
};

class PolicyNetwork {
public:
    PolicyNetwork(const std::string& solver_param_file, int seed);

    void loadModel(const std::string& model_file);
    void saveModel(const std::string& model_file) const;
    std::pair<Action, float> selectActionSoftmax(const State& state) const;
    std::pair<Action, float> selectActionSoftmax(const State& state);
    std::vector<std::pair<Action, float> > selectActionSoftmax(const std::vector<State>& batch);
    void update(const std::vector<std::pair<State, Action> >& transitions, GameResult result);

private:
    std::mt19937 randomEngine_;

    std::shared_ptr<caffe::Solver<float> > solver_;
    boost::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::Blob<float> > probBlob_;
    boost::shared_ptr<caffe::MemoryDataLayer<float> > meFieldInputLayer_,
                                                      enemyFieldInputLayer_,
                                                      auxInputLayer_,
                                                      actionInputLayer_,
                                                      rewardInputLayer_,
                                                      legalityInputLayer_;

    ActionRewardInputData dummyInputDataAR_;
    LegalityInputData dummyInputDataL_;

    int puyoColorToNum(PuyoColor color) const;
    std::array<std::array<float, SINGLE_PLAYER_FIELD_DATA_SIZE>, 2>
        extractFieldData(const State& state) const;
    std::array<float, AUX_DATA_SIZE> extractAuxData(const State& state) const;
    void inputDataIntoLayers(const FieldInputData& meFieldInput,
                             const FieldInputData& enemyFieldInput,
                             const AuxInputData& auxInput,
                             const ActionRewardInputData& actionInput,
                             const ActionRewardInputData& rewardInput,
                             const LegalityInputData& legalityInput) const;
    int gameResultToInt(GameResult result) const;
};

}

#endif
