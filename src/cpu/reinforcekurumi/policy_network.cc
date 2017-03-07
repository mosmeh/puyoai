#include "policy_network.h"

DEFINE_bool(print_prob, false, "");

namespace kurumi {

PolicyNetwork::PolicyNetwork(const std::string& solverParamFile, int seed) : randomEngine_(seed) {
    caffe::SolverParameter solverParam;
    caffe::ReadProtoFromTextFileOrDie(solverParamFile, &solverParam);
    solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solverParam));

    net_ = solver_->net();

    probBlob_ = net_->blob_by_name("prob");

    std::fill(dummyInputDataAR_.begin(), dummyInputDataAR_.end(), 0);
    std::fill(dummyInputDataL_.begin(), dummyInputDataL_.end(), 0);

    meFieldInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("me_field_input_layer"));
    enemyFieldInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("enemy_field_input_layer"));
    auxInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("aux_input_layer"));
    actionInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("action_input_layer"));
    rewardInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("reward_input_layer"));
    legalityInputLayer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(
        net_->layer_by_name("legality_input_layer"));
}

int PolicyNetwork::puyoColorToNum(PuyoColor color) const {
    switch (color) {
        case PuyoColor::EMPTY:
            return -1;
        case PuyoColor::OJAMA:
            return 0;
        default:
            return static_cast<int>(color) - 3;
    }
}

std::array<std::array<float, SINGLE_PLAYER_FIELD_DATA_SIZE>, 2>
PolicyNetwork::extractFieldData(const State& state) const {
    std::array<float, SINGLE_PLAYER_FIELD_DATA_SIZE> meFieldData, enemyFieldData;
    std::fill(meFieldData.begin(), meFieldData.end(), 0);
    std::fill(enemyFieldData.begin(), enemyFieldData.end(), 0);

    int idx = 0;
    for (int y = 1; y <= FieldConstant::HEIGHT + 1; ++y) {
        for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
            /*meFieldData[x - 1 + (y - 1) * FieldConstant::WIDTH +
              puyoColorToNum(state.me.field.color(x, y)) * FIELD_SIZE] = 1;
            enemyFieldData[x - 1 + (y - 1) * FieldConstant::WIDTH +
              puyoColorToNum(state.enemy.field.color(x, y)) * FIELD_SIZE] = 1;*/
            /*meFieldData[idx + puyoColorToNum(state.me.field.color(x, y)) * FIELD_SIZE] = 1;
            enemyFieldData[idx + puyoColorToNum(state.enemy.field.color(x, y)) * FIELD_SIZE] = 1;*/
            auto meColor = state.me.field.color(x, y);
            if (meColor != PuyoColor::EMPTY) {
                meFieldData.at(idx + puyoColorToNum(meColor) * FIELD_SIZE) = 1;
            }
            auto enemyColor = state.enemy.field.color(x, y);
            if (enemyColor != PuyoColor::EMPTY) {
                enemyFieldData.at(idx + puyoColorToNum(enemyColor) * FIELD_SIZE) = 1;
            }
            idx += 1;
        }
    }

    return {meFieldData, enemyFieldData};
}

std::array<float, AUX_DATA_SIZE> PolicyNetwork::extractAuxData(const State& state) const {
    auto me = state.me;
    auto enemy = state.enemy;

    std::array<float, AUX_DATA_SIZE> auxData;
    std::fill(auxData.begin(), auxData.end(), 0);

    for (int i = 0; i < 2; ++i) {
        if (me.seq.get(i).axis != PuyoColor::EMPTY) {
            auxData.at((2 * i + 0) * 5 + puyoColorToNum(me.seq.get(i).axis)) = 1;
        }
        if (me.seq.get(i).child != PuyoColor::EMPTY) {
            auxData.at((2 * i + 1) * 5 + puyoColorToNum(me.seq.get(i).child)) = 1;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (enemy.seq.get(i).axis != PuyoColor::EMPTY) {
            auxData.at((2 * (i + 2) + 0) * 5 + puyoColorToNum(enemy.seq.get(i).axis)) = 1;
        }
        if (enemy.seq.get(i).child != PuyoColor::EMPTY) {
            auxData.at((2 * (i + 2) + 1) * 5 + puyoColorToNum(enemy.seq.get(i).child)) = 1;
        }
    }

    for (auto action : state.legalActions) {
        auxData.at(30 + action.decisionId) = static_cast<float>(me.field.framesToDropNext(action.decision)) / 134;
        auxData.at(52 + action.decisionId) = me.field.isChigiriDecision(action.decision);
    }

    int idx = 74;

    for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
        auxData.at(idx++) = static_cast<float>(me.field.height(x)) / (FieldConstant::HEIGHT + 1);
        auxData.at(idx++) = static_cast<float>(me.field.ridgeHeight(x)) / (FieldConstant::HEIGHT + 1);
        auxData.at(idx++) = static_cast<float>(me.field.valleyDepth(x)) / (FieldConstant::HEIGHT + 1);

        auxData.at(idx++) = static_cast<float>(enemy.field.height(x)) / (FieldConstant::HEIGHT + 1);
        auxData.at(idx++) = static_cast<float>(enemy.field.ridgeHeight(x)) / (FieldConstant::HEIGHT + 1);
        auxData.at(idx++) = static_cast<float>(enemy.field.valleyDepth(x)) / (FieldConstant::HEIGHT + 1);
    }

    auxData.at(idx++) = static_cast<float>(me.field.countReachableSpaces()) / FIELD_SIZE;
    auxData.at(idx++) = static_cast<float>(me.field.countUnreachableSpaces()) / FIELD_SIZE;
    auxData.at(idx++) = static_cast<float>(enemy.field.countReachableSpaces()) / FIELD_SIZE;
    auxData.at(idx++) = static_cast<float>(enemy.field.countUnreachableSpaces()) / FIELD_SIZE;

    auxData.at(idx++) = me.isRensaOngoing();
    auxData.at(idx++) = enemy.isRensaOngoing();

    auxData.at(idx++) = me.hasZenkeshi;
    auxData.at(idx++) = enemy.hasZenkeshi;

    static const int ASSUMED_MAX_FRAMES = 4000;
    auxData.at(idx++) = static_cast<float>(me.rensaFinishingFrameId() - state.frameId) / ASSUMED_MAX_FRAMES;
    auxData.at(idx++) = static_cast<float>(enemy.rensaFinishingFrameId() - state.frameId) / ASSUMED_MAX_FRAMES;

    static const int ASSUMED_MAX_NUM_OJAMA = 4000;
    auxData.at(idx++) = static_cast<float>(me.totalOjama(enemy)) / ASSUMED_MAX_NUM_OJAMA;
    auxData.at(idx++) = static_cast<float>(enemy.totalOjama(me)) / ASSUMED_MAX_NUM_OJAMA;

    auxData.at(idx++) = static_cast<float>(me.noticedOjama()) / ASSUMED_MAX_NUM_OJAMA;
    auxData.at(idx++) = static_cast<float>(enemy.noticedOjama()) / ASSUMED_MAX_NUM_OJAMA;

    auxData.at(idx++) = static_cast<float>(me.fixedOjama) / ASSUMED_MAX_NUM_OJAMA;
    auxData.at(idx++) = static_cast<float>(enemy.fixedOjama) / ASSUMED_MAX_NUM_OJAMA;

    auxData.at(idx++) = static_cast<float>(me.pendingOjama) / ASSUMED_MAX_NUM_OJAMA;
    auxData.at(idx++) = static_cast<float>(enemy.pendingOjama) / ASSUMED_MAX_NUM_OJAMA;

    return auxData;
}

void PolicyNetwork::loadModel(const std::string& model_file) {
    net_->CopyTrainedLayersFrom(model_file);
}

void PolicyNetwork::saveModel(const std::string& model_file) const {
    caffe::NetParameter netParam;
    net_->ToProto(&netParam);
    caffe::WriteProtoToBinaryFile(netParam, model_file);
}

std::pair<Action, float> PolicyNetwork::selectActionSoftmax(const State& state) const {
    return const_cast<PolicyNetwork*>(this)->selectActionSoftmax(state);
}

std::pair<Action, float> PolicyNetwork::selectActionSoftmax(const State& state) {
    return selectActionSoftmax(std::vector<State>({state})).front();
}

std::vector<std::pair<Action, float> > PolicyNetwork::selectActionSoftmax(const std::vector<State>& batch) {
    std::vector<std::pair<Action, float> > results;
    results.reserve(batch.size());

    for (unsigned int b_i = 0; b_i < batch.size(); b_i += MAX_MINIBATCH_SIZE) {
        FieldInputData meFieldInput, enemyFieldInput;
        AuxInputData auxInput;

        std::fill(meFieldInput.begin(), meFieldInput.end(), 0);
        std::fill(enemyFieldInput.begin(), enemyFieldInput.end(), 0);
        std::fill(auxInput.begin(), auxInput.end(), 0);

        int minibatchSize = std::min(static_cast<int>(batch.size() - b_i), MAX_MINIBATCH_SIZE);

        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            auto fieldData = extractFieldData(batch.at(b_i + m_i));
            std::copy(fieldData[0].begin(), fieldData[0].end(), meFieldInput.begin() + m_i * SINGLE_PLAYER_FIELD_DATA_SIZE);
            std::copy(fieldData[1].begin(), fieldData[1].end(), enemyFieldInput.begin() + m_i * SINGLE_PLAYER_FIELD_DATA_SIZE);

            auto auxData = extractAuxData(batch.at(b_i + m_i));
            std::copy(auxData.begin(), auxData.end(), auxInput.begin() + m_i * AUX_DATA_SIZE);
        }

        inputDataIntoLayers(meFieldInput, enemyFieldInput, auxInput, dummyInputDataAR_, dummyInputDataAR_, dummyInputDataL_);
        net_->Forward();

        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            std::array<float, MAX_NUM_LEGAL_DECISIONS> probs;
            std::fill(probs.begin(), probs.end(), 0);
            for (auto action : batch.at(b_i + m_i).legalActions) {
                probs[action.decisionId] = probBlob_->data_at(m_i, action.decisionId, 0, 0);
            }

            if (FLAGS_print_prob) {
              for (int i = 0; i < MAX_NUM_LEGAL_DECISIONS; ++i) {
                  std::cerr << probs.at(i) << " ";
              }
              std::cerr << std::endl;
            }

            auto idx = std::discrete_distribution<int>(probs.begin(), probs.end())(randomEngine_);
            results.emplace_back(DECISIONS[idx], probs.at(idx));
        }
    }

    return results;
}

void PolicyNetwork::update(const std::vector<std::pair<State, Action> >& transitions, GameResult result) {
    std::vector<State> states;
    for (unsigned int i = 0; i < transitions.size(); ++i) {
      if (transitions[i].first.legalActions.size() > 0) {
        states.emplace_back(transitions[i].first);
      }
    }
    int reward = gameResultToInt(result);

    for (unsigned int t_i = 0; t_i < transitions.size(); t_i += MAX_MINIBATCH_SIZE) {
        FieldInputData meFieldInput, enemyFieldInput;
        AuxInputData auxInput;
        ActionRewardInputData actionInput, rewardInput;
        LegalityInputData legalityInput;

        std::fill(meFieldInput.begin(), meFieldInput.end(), 0);
        std::fill(enemyFieldInput.begin(), enemyFieldInput.end(), 0);
        std::fill(auxInput.begin(), auxInput.end(), 0);
        std::fill(actionInput.begin(), actionInput.end(), 0);
        std::fill(rewardInput.begin(), rewardInput.end(), 0);
        std::fill(legalityInput.begin(), legalityInput.end(), 0);

        int minibatchSize = std::min(static_cast<int>(transitions.size() - t_i), MAX_MINIBATCH_SIZE);

        for (int m_i = 0; m_i < minibatchSize; ++m_i) {
            auto fieldData = extractFieldData(transitions[t_i + m_i].first);
            std::copy(fieldData[0].begin(), fieldData[0].end(), meFieldInput.begin() + m_i * SINGLE_PLAYER_FIELD_DATA_SIZE);
            std::copy(fieldData[1].begin(), fieldData[1].end(), enemyFieldInput.begin() + m_i * SINGLE_PLAYER_FIELD_DATA_SIZE);

            auto auxData = extractAuxData(transitions[t_i + m_i].first);
            std::copy(auxData.begin(), auxData.end(), auxInput.begin() + m_i * AUX_DATA_SIZE);

            actionInput[m_i] = transitions[t_i + m_i].second.decisionId;
            rewardInput[m_i] = reward;
            for (auto action : transitions.at(t_i + m_i).first.legalActions) {
              legalityInput[m_i * MAX_NUM_LEGAL_DECISIONS + action.decisionId] = 1;
            }
        }

        inputDataIntoLayers(meFieldInput, enemyFieldInput, auxInput, actionInput, rewardInput, legalityInput);
        solver_->Step(1);
    }
}

int PolicyNetwork::gameResultToInt(GameResult result) const {
    switch (result) {
        case GameResult::P1_WIN:
            return 1;
        case GameResult::P2_WIN:
            return -1;
        case GameResult::DRAW:
            return 0;
        default:
            DCHECK(false);
            return 0;
    }
}

void PolicyNetwork::inputDataIntoLayers(const FieldInputData& meFieldInput,
                                   const FieldInputData& enemyFieldInput,
                                   const AuxInputData& auxInput,
                                   const ActionRewardInputData& actionInput,
                                   const ActionRewardInputData& rewardInput,
                                   const LegalityInputData& legalityInput) const {
    meFieldInputLayer_->Reset(const_cast<float*>(meFieldInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
    enemyFieldInputLayer_->Reset(const_cast<float*>(enemyFieldInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
    auxInputLayer_->Reset(const_cast<float*>(auxInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
    actionInputLayer_->Reset(const_cast<float*>(actionInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
    rewardInputLayer_->Reset(const_cast<float*>(rewardInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
    legalityInputLayer_->Reset(const_cast<float*>(legalityInput.data()),
      const_cast<float*>(dummyInputDataAR_.data()), MAX_MINIBATCH_SIZE);
}

}
