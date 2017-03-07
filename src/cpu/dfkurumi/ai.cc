#include "ai.h"

DEFINE_int32(seed, 1, "seed");
DEFINE_int32(depth, 10, "depth of decision tree");
DEFINE_bool(learn, false, "if true, update parameters");
DEFINE_int32(snapshot, 10000, "save parameters every snapshot games");
DEFINE_string(model, "", "parameter file");
DEFINE_int32(resume_games, 0, "games at which parameter file was saved");
DEFINE_bool(writer, false, "writes parameters to shared memory");
DEFINE_int32(sync_freq, 500, "sync parameters every sync_freq games");
DEFINE_double(eta_weights, 0.001, "learning rate a");
DEFINE_double(eta_dists, 0.001, "learning rate b");
DEFINE_string(id, "", "");

namespace kurumi {

DFKurumiAI::DFKurumiAI(int argc, char* argv[]) :
  AI(argc, argv, "dfkurumi"), games_(FLAGS_resume_games), enemySeq_("...."), randomEngine_(FLAGS_seed), stochasticDecisionTree_(FLAGS_depth, randomEngine_) {
    if (FLAGS_learn) {
        std::stringstream ss;
        ss << "KurumiParameter" << FLAGS_id;
        stochasticDecisionTree_.setUpToLearn(ss.str(), FLAGS_eta_weights, FLAGS_eta_dists, FLAGS_writer);
    }
    if (!FLAGS_model.empty()) {
        stochasticDecisionTree_.loadModel(FLAGS_model);
    }
}

DropDecision DFKurumiAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    auto meState = me;
    meState.seq = seq;
    auto enemyState = enemy;
    enemyState.seq = (enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_);
    const auto state = State(frameId, meState, enemyState);

    const auto res = stochasticDecisionTree_.selectAction(state);
    const auto& action = res.first;

    int leafId;
    DTYPE mu;
    std::tie(leafId, mu, std::ignore) = res.second;
    const auto& dist = std::get<2>(res.second);

    const DTYPE entropy = -(dist * std::log(dist)).sum() / std::log(2);
    static const DTYPE maxEntropy = std::log(NUM_DECISIONS) / std::log(2);

    std::stringstream ss;
    ss << "depth=" << FLAGS_depth << " ";
    if (FLAGS_writer) {
        ss << "etaWeights=" << FLAGS_eta_weights << " "
           << "etaDists=" << FLAGS_eta_dists << " "
           << "syncFreq=" << FLAGS_sync_freq << " ";
    }
    ss << std::endl << "entropy: " << entropy << "bits (" << entropy / maxEntropy * 100 << "%) "
       << "leaf: " << leafId << " "
       << "mu: " << mu << std::endl
       << "dist: ";
    std::copy(std::begin(dist), std::end(dist), std::ostream_iterator<DTYPE>(ss, " "));

    if (FLAGS_learn && FLAGS_writer) {
        storeTransition({state, action});
    }
    return DropDecision(action.simplify(seq.front().isRep()).decision, ss.str());
}

void DFKurumiAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

void DFKurumiAI::onGameHasEnded(const FrameRequest& req) {
    ++games_;
    if (FLAGS_learn) {
        if (FLAGS_writer) {
            stochasticDecisionTree_.update(transitions_, gameResultToInt(req.gameResult));
            transitions_.clear();
            transitions_.shrink_to_fit();

            if (games_ % FLAGS_snapshot == 0) {
                std::stringstream ss;
                ss << "games=" << games_ << ","
                   << "eta_weights=" << FLAGS_eta_weights << ","
                   << "eta_dists=" << FLAGS_eta_dists << ","
                   << "sync_freq=" << FLAGS_sync_freq << ","
                   << "depth=" << FLAGS_depth << ".model";
                stochasticDecisionTree_.saveModel(ss.str());
            }
            if (games_ % FLAGS_sync_freq == 0) {
                stochasticDecisionTree_.sync();
            }
        } else {
            stochasticDecisionTree_.sync();
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
