#include "ai.h"
#include <sstream>
#include <iostream>

DEFINE_string(expert, "../mayah/run.sh", "expert AI");
DEFINE_string(recname, "", "");

namespace kurumi {

DuelRecorderAI::DuelRecorderAI(int argc, char* argv[]) :
  PuppetAI(argc, argv, "duelrecorder", FLAGS_expert), enemySeq_("....") {
    std::stringstream ss;
    ss << FLAGS_recname << ".duel";
    ofs_.open(ss.str(), std::ios::out | std::ios::binary);
}

DuelRecorderAI::~DuelRecorderAI() {
    ofs_.close();
}

DropDecision DuelRecorderAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    std::cerr << "think" << std::endl;
    if (masterResponse_.isValid()) {
      std::cerr << "valid" << std::endl;
        auto meState = me;
        meState.seq = seq;
        auto enemyState = enemy;
        enemyState.seq = (enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_);
        std::cerr << "c state" << std::endl;
        auto state = State(frameId, meState, enemyState);
        std::cerr << "saving" << std::endl;
        saveTransition(state);
    }

    return DropDecision(Decision(3, 2));
}

void DuelRecorderAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

void DuelRecorderAI::saveTransition(const State& state) const {
      std::cerr << "saving" << std::endl;
    const_cast<DuelRecorderAI*>(this)->saveTransition(state);
}

void DuelRecorderAI::saveTransition(const State state) {
    std::cerr << "writing" << std::endl;
    char decisionId = Action(masterResponse_.decision).decisionId;
    ofs_.write(&decisionId, sizeof(char));
    ofs_.write(reinterpret_cast<char*>(const_cast<State*>(&state)), sizeof(State));
    std::cerr << "wrote" << std::endl;
}

}
