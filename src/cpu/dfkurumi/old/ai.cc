#include "ai.h"
#include <iostream>
#include <iterator>

DEFINE_int32(snapshot, 10000, "save parameters every snapshot games");
DEFINE_int32(resumegames, 0, "games at which parameters file was saved");
DEFINE_bool(evaluation, false, "if true, do not update parameters");
DEFINE_string(model, "", "parameters file");
DEFINE_bool(print, false, "print action distribution");
DEFINE_int32(depth, 10, "depth of decision tree");
DEFINE_double(alpha, 0.001, "learning rate a");
DEFINE_double(beta, 0.1, "learning rate b");

namespace kurumi {

DFKurumiAI::DFKurumiAI(int argc, char* argv[], std::mt19937& randomEngine) :
  AI(argc, argv, "dfkurumi"), games_(FLAGS_resumegames), enemySeq_("...."), stochasticDecisionTree_(FLAGS_depth, FLAGS_alpha, FLAGS_beta, randomEngine) {
    if (!FLAGS_model.empty()) {
        stochasticDecisionTree_.loadModel(FLAGS_model);
    }
}

DropDecision DFKurumiAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(frameId);
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    auto meState = me;
    meState.seq = seq;
    auto enemyState = enemy;
    enemyState.seq = enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_;
    const auto state = State(frameId, meState, enemyState);

    const auto res = stochasticDecisionTree_.selectAction(state);
    if (FLAGS_print) {
        std::copy(std::begin(res.second), std::end(res.second), std::ostream_iterator<float>(std::cerr, " "));
        std::cerr << std::endl;
    }
    if (FLAGS_evaluation) {
        return DropDecision(res.first.decision);
    } else {
        //for (int i = 0; i < 8; ++i) storeTransition({state, Action(Decision(1, 2))});
        storeTransition({state, res.first});
        return DropDecision(Decision(res.first.decision));
    }
}

void DFKurumiAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

void DFKurumiAI::onGameHasEnded(const FrameRequest& req) {
    UNUSED_VARIABLE(req);
    if (!FLAGS_evaluation) {
        stochasticDecisionTree_.update(transitions_, gameResultToInt(req.gameResult));
        //stochasticDecisionTree_.update(transitions_, 1);
        transitions_.clear();
        transitions_.shrink_to_fit();

        ++games_;
        if (games_ % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "iter_" << games_ << ".model";
            stochasticDecisionTree_.saveModel(ss.str());
        }
    }
}

int DFKurumiAI::gameResultToInt(const GameResult result) const {
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

void DFKurumiAI::storeTransition(const std::pair<State, Action>& trans) const {
    const_cast<DFKurumiAI*>(this)->storeTransition(trans);
}

void DFKurumiAI::storeTransition(const std::pair<State, Action>& trans) {
    transitions_.emplace_back(trans);
}

}
