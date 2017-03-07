#include "ai.h"

DEFINE_int32(seed, 1, "seed");
DEFINE_int32(snapshot, 10000, "");
DEFINE_int32(resumegames, 0, "");
DEFINE_bool(evaluation, false, "");
DEFINE_string(solver, "solver.prototxt", "parameter file for solver of policy network");
DEFINE_string(model, "", "");

namespace kurumi {

ReinforceKurumiAI::ReinforceKurumiAI(int argc, char* argv[]) :
  AI(argc, argv, "reinforcekurumi"), games(FLAGS_resumegames), enemySeq("...."), policyNetwork_(FLAGS_solver, FLAGS_seed) {
    if (!FLAGS_model.empty()) {
        policyNetwork_.loadModel(FLAGS_model);
    }
}

DropDecision ReinforceKurumiAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(frameId);
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    auto meState = me;
    meState.seq = seq;
    auto enemyState = enemy;
    enemyState.seq = enemySeq;
    const auto state = State(frameId, meState, enemyState);

    if (FLAGS_evaluation) {
        return DropDecision(policyNetwork_.selectActionSoftmax(state).first.decision);
    } else {
        const auto action = policyNetwork_.selectActionSoftmax(state).first;
        storeTransition({state, action});
        return DropDecision(Decision(action.decision));
    }
}

void ReinforceKurumiAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq = seq;
}

void ReinforceKurumiAI::onGameHasEnded(const FrameRequest& req) {
    UNUSED_VARIABLE(req);
    if (!FLAGS_evaluation) {
        policyNetwork_.update(transitions_, req.gameResult);
        transitions_.clear();
        transitions_.shrink_to_fit();

        ++games;
        if (games % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "iter_" << games << ".caffemodel";
            policyNetwork_.saveModel(ss.str());
        }
    }
}

void ReinforceKurumiAI::storeTransition(const std::pair<State, Action>& trans) const {
    const_cast<ReinforceKurumiAI*>(this)->storeTransition(trans);
}

void ReinforceKurumiAI::storeTransition(const std::pair<State, Action>& trans) {
    transitions_.push_back(trans);
}

}
