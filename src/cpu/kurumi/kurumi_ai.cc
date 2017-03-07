#include "kurumi_ai.h"

DEFINE_int32(seed, 1, "seed");
DEFINE_int32(depth, 10, "depth of decision tree");
DEFINE_string(model, "", "parameter file");
DEFINE_bool(greedy, false, "");
DEFINE_string(solver, "solver.prototxt", "");

namespace kurumi {

KurumiAI::KurumiAI(int argc, char* argv[]) :
  AI(argc, argv, "kurumi"), enemySeq_("...."), randomEngine_(FLAGS_seed), /*mlp_(FLAGS_solver) {*/tree_(FLAGS_depth, randomEngine_) {
    decisionBook_.load("decision.toml");
    if (!FLAGS_model.empty()) {
        tree_.loadModel(FLAGS_model);
    }
}

DropDecision KurumiAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    if (!enemy.hasZenkeshi) {
        auto d = decisionBook_.nextDecision(f, seq);
        if (d.isValid()) {
            return DropDecision(d, "opening book");
        }
    }

    auto meState = me;
    meState.seq = seq;
    auto enemyState = enemy;
    enemyState.seq = (enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_);
    const auto state = State(frameId, meState, enemyState);

    auto res = tree_.selectAction(state);
    if (FLAGS_greedy) {
        res = tree_.selectActionGreedily(state);
    }
    auto action = state.legalActions.sum() > 0 ? res.first : Action(Decision(3, 0));

    int leafId;
    DTYPE mu;
    std::tie(leafId, mu, std::ignore) = res.second;
    const auto& dist = std::get<2>(res.second);
    /*const auto dist = mlp_.predict(state);
    const int id = std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);
    const Action action(id);*/

    const DTYPE entropy = -(dist * std::log(dist + static_cast<DTYPE>(1e-7))).sum() / std::log(2);
    const DTYPE maxEntropy = std::log(state.legalActions.sum() + 1e-7) / std::log(2);

    std::stringstream ss;
    ss.precision(4);
    ss << std::fixed
        << "entropy: " << entropy << "bits (" << entropy / maxEntropy * 100 << "%) "
        << "leaf: " << leafId << " "
        << "mu: " << mu << std::endl
        << "dist: ";
    std::copy(std::begin(dist), std::end(dist), std::ostream_iterator<DTYPE>(ss, " "));

    return DropDecision(action.simplify(seq.front().isRep()).decision, ss.str());
}

void KurumiAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

}

