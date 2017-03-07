#include "reinforce_kurumi_ai.h"

DEFINE_int32(seed, 1, "seed");
DEFINE_int32(depth, 10, "depth of decision tree");
DEFINE_int32(snapshot, 10000, "save parameters every snapshot games");
DEFINE_string(model, "", "parameter file");
DEFINE_int32(resume_games, 0, "games at which parameter file was saved");
DEFINE_bool(writer, false, "writes parameters to shared memory");
DEFINE_int32(sync_freq, 500, "sync parameters every sync_freq games");
DEFINE_double(eta_weights, 0.001, "learning rate a");
DEFINE_double(eta_dists, 0.001, "learning rate b");
DEFINE_double(beta, 0.0001, "");
DEFINE_int32(minibatch_size, 100, "");
DEFINE_string(id, "", "");
DEFINE_string(solver, "solver.prototxt", "");

namespace kurumi {

ReinforceKurumiAI::ReinforceKurumiAI(int argc, char* argv[]) :
  AI(argc, argv, "reinforce_kurumi"), games_(FLAGS_resume_games), enemySeq_("...."), randomEngine_(FLAGS_seed), mlp_(FLAGS_solver, randomEngine_){//tree_(FLAGS_depth, randomEngine_, FLAGS_id, FLAGS_eta_weights, FLAGS_eta_dists, FLAGS_beta, FLAGS_minibatch_size, FLAGS_writer) {
    decisionBook_.load("decision.toml");
    leafCount = std::valarray<DTYPE>(0.0, 1023);
    if (!FLAGS_model.empty()) {
        //tree_.loadModel(FLAGS_model);
        mlp_.loadModel(FLAGS_model);
    }
    if (FLAGS_writer) {
        std::stringstream ss;
        ss << "id=" << FLAGS_id << "-reinforce-log.txt";
        ofs_.open(ss.str(), std::ios::out | std::ios::app);
        //tree_.writeToSharedMemory();
    } else {
        //tree_.readFromSharedMemory();
    }
}

DropDecision ReinforceKurumiAI::think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                   const PlayerState& me, const PlayerState& enemy, bool fast) const
{
    UNUSED_VARIABLE(f);
    UNUSED_VARIABLE(fast);

    auto meState = me;
    meState.seq = seq;
    auto enemyState = enemy;
    enemyState.seq = (enemySeq_ == KumipuyoSeq("....") ? seq : enemySeq_);
    const auto state = State(frameId, meState, enemyState);

    /*const auto res = tree_.selectAction(state);
    auto action = state.legalActions.sum() > 0 ? res.first : Action(Decision(3, 0));*/
    Action action(0);
    std::valarray<DTYPE> dist(0.0, 22);
    std::tie(action, dist) = mlp_.selectAction(state);

    if (!enemy.hasZenkeshi) {
        auto d = decisionBook_.nextDecision(f, seq);
        if (d.isValid()) {
            action = Action(d);
        }
    }
    int mu = 0;
    int leafId = 0;
    /*int leafId;
    DTYPE mu;
    std::tie(leafId, mu, std::ignore) = res.second;
    const auto& dist = std::get<2>(res.second);*/

    const DTYPE entropy = -(dist * std::log(dist + static_cast<DTYPE>(1e-7))).sum() / std::log(2);
    const DTYPE maxEntropy = log2(state.legalActions.sum() + 1e-7);

    updateStats(mu, entropy, leafId);

    std::stringstream ss;
    ss.precision(4);
    ss << std::fixed << "id=" << FLAGS_id << " depth=" << FLAGS_depth << " ";
    /*if (FLAGS_writer) {
        ss << "etaWeights=" << FLAGS_eta_weights << " "
           << "etaDists=" << FLAGS_eta_dists << " "
           << "beta=" << FLAGS_beta << " ";
    } else {
        ss << "syncFreq=" << FLAGS_sync_freq << " ";
    }*/
    ss << std::endl << "entropy: " << entropy << "bits (" << entropy / maxEntropy * 100 << "%) "
       /*<< "leaf: " << leafId << " "
       << "mu: " << mu << std::endl*/
       << "dist: ";
    std::copy(std::begin(dist), std::end(dist), std::ostream_iterator<DTYPE>(ss, " "));

    if (FLAGS_writer) {
        storeTransition({state, action.decisionId});
    }
    return DropDecision(action.simplify(seq.front().isRep()).decision, ss.str());
}

void ReinforceKurumiAI::gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq) {
  UNUSED_VARIABLE(frameId);
  UNUSED_VARIABLE(enemyField);

  enemySeq_ = seq;
}

void ReinforceKurumiAI::onGameHasEnded(const FrameRequest& req) {
    ++games_;
    if (FLAGS_writer) {
        //tree_.update(transitions_, gameResultToInt(req.gameResult));
        mlp_.update(transitions_, gameResultToInt(req.gameResult));
        transitions_.clear();
        transitions_.shrink_to_fit();

        int score = req.myPlayerFrameRequest().score;
        maxScore = std::max(maxScore, score);
        sumScore += score;
        if (req.gameResult == GameResult::P1_WIN) {
            wins++;
        }

        //tree_.writeToSharedMemory();
        if (games_ % FLAGS_snapshot == 0) {
            std::stringstream ss;
            ss << "id=" << FLAGS_id << ","
               << "games=" << games_ << //","
               /*<< "eta_weights=" << FLAGS_eta_weights << ","
               << "eta_dists=" << FLAGS_eta_dists << ","
               << "beta=" << FLAGS_beta << ","
               << "sync_freq=" << FLAGS_sync_freq << ","
               << "depth=" << FLAGS_depth <<*/ ".model";
            //tree_.saveModel(ss.str());
            mlp_.saveModel(ss.str());

            DTYPE winrate = static_cast<DTYPE>(wins) / FLAGS_snapshot;
            leafCount /= leafCount.sum();
            ofs_ << games_ << " "
                 << maxScore << " "
                 << static_cast<DTYPE>(sumScore) / FLAGS_snapshot << " "
                 << winrate << " "
                 << 1.96 * std::sqrt(winrate * (1 - winrate) / FLAGS_snapshot) << " "
                 << sumMu / numDecisions << " "
                 << sumEntropy / numDecisions << " "
                 << -(leafCount * std::log(leafCount + static_cast<DTYPE>(1e-7))).sum() / std::log(1023) << std::endl;
            maxScore = 0;
            sumScore = 0;
            wins = 0;
            sumMu = 0;
            sumEntropy = 0;
            std::fill(std::begin(leafCount), std::end(leafCount), 0.0);
            numDecisions = 0;
        }
    } else if (games_ % FLAGS_sync_freq == 0) {
        //tree_.readFromSharedMemory();
    }
}

int ReinforceKurumiAI::gameResultToInt(const GameResult result) const {
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

void ReinforceKurumiAI::storeTransition(const std::pair<State, int>& trans) const {
    const_cast<ReinforceKurumiAI*>(this)->storeTransition(trans);
}

void ReinforceKurumiAI::storeTransition(const std::pair<State, int>& trans) {
    transitions_.emplace_back(trans);
}
void ReinforceKurumiAI::updateStats(const DTYPE mu, const DTYPE pentropy, const int leafId) const {
    const_cast<ReinforceKurumiAI*>(this)->updateStats(mu, pentropy, leafId);
}
void ReinforceKurumiAI::updateStats(const DTYPE mu, const DTYPE entropy, const int leafId) {
    sumMu += mu;
    sumEntropy += entropy;
    leafCount[leafId] += 1;
    numDecisions++;
}

}
