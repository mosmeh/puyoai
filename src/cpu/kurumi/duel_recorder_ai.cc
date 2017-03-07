#include "duel_recorder_ai.h"

DEFINE_string(expert, "../mayah/run.sh", "expert AI");
DEFINE_string(record_name, "save.duel", "");

namespace kurumi {

DuelRecorderAI::DuelRecorderAI(int argc, char* argv[]) :
  PuppetAI(argc, argv, "duel_recorder", FLAGS_expert), enemySeq_("...."), ofs_(FLAGS_record_name, std::ios::out | std::ios::binary) {}

DuelRecorderAI::~DuelRecorderAI() {
    ofs_.close();
}

DropDecision DuelRecorderAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    if (masterResponse_.isValid()) {
        auto meState = me;
        meState.seq = seq;
        auto enemyState = enemy;
        enemyState.seq = enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_;
        auto state = State(frameId, meState, enemyState);
        saveTransition(std::move(state));
    }

    return DropDecision(Decision(3, 2));
}

void DuelRecorderAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

void DuelRecorderAI::saveTransition(const State& state) const {
    const_cast<DuelRecorderAI*>(this)->saveTransition(state);
}

void DuelRecorderAI::saveTransition(const State& state) {
    char decisionId = Action(masterResponse_.decision).decisionId;
    ofs_.write(&decisionId, sizeof(char));
    //ofs_.write(reinterpret_cast<char*>(const_cast<State*>(&state)), sizeof(State));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.frameId)), sizeof(int));

    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.hand)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<CoreField*>(&state.me.field)), sizeof(CoreField));
    ofs_.write(reinterpret_cast<char*>(const_cast<bool*>(&state.me.hasZenkeshi)), sizeof(bool));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.fixedOjama)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.pendingOjama)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.unusedScore)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.currentChain)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.me.currentChainStartedFrameId)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<RensaResult*>(&state.me.currentRensaResult)), sizeof(RensaResult));
    ofs_.write(reinterpret_cast<char*>(const_cast<CoreField*>(&state.me.fieldWhenGrounded)), sizeof(CoreField));
    ofs_.write(reinterpret_cast<char*>(const_cast<bool*>(&state.me.hasOjamaDropped)), sizeof(bool));
    ofs_.write(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&state.me.seq.get(0))), sizeof(Kumipuyo));
    ofs_.write(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&state.me.seq.get(1))), sizeof(Kumipuyo));

    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.hand)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<CoreField*>(&state.enemy.field)), sizeof(CoreField));
    ofs_.write(reinterpret_cast<char*>(const_cast<bool*>(&state.enemy.hasZenkeshi)), sizeof(bool));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.fixedOjama)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.pendingOjama)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.unusedScore)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.currentChain)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<int*>(&state.enemy.currentChainStartedFrameId)), sizeof(int));
    ofs_.write(reinterpret_cast<char*>(const_cast<RensaResult*>(&state.enemy.currentRensaResult)), sizeof(RensaResult));
    ofs_.write(reinterpret_cast<char*>(const_cast<CoreField*>(&state.enemy.fieldWhenGrounded)), sizeof(CoreField));
    ofs_.write(reinterpret_cast<char*>(const_cast<bool*>(&state.enemy.hasOjamaDropped)), sizeof(bool));
    ofs_.write(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&state.enemy.seq.get(0))), sizeof(Kumipuyo));
    ofs_.write(reinterpret_cast<char*>(const_cast<Kumipuyo*>(&state.enemy.seq.get(1))), sizeof(Kumipuyo));
}

}
