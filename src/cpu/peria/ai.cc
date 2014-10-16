#include "cpu/peria/ai.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <sstream>
#include <string>
#include <vector>

#include "core/algorithm/plan.h"
#include "core/constant.h"
#include "core/frame_request.h"

#include "cpu/peria/pattern.h"

namespace peria {

struct Ai::Attack {
  int score;
  int end_frame_id;
};

struct Ai::Control {
  std::string message;
  int score = 0;
  Decision decision;
};

// TODO: (want to implement)
// - Search decisions for all known |seq|
// --- Count the number of HAKKA-able KumiPuyos

Ai::Ai(int argc, char* argv[]): ::AI(argc, argv, "peria") {}

Ai::~Ai() {}

DropDecision Ai::think(int frame_id,
                       const CoreField& field,
                       const KumipuyoSeq& seq,
                       const AdditionalThoughtInfo& info,
                       bool fast) {
  UNUSED_VARIABLE(frame_id);
  UNUSED_VARIABLE(info);
  UNUSED_VARIABLE(fast);
  using namespace std::placeholders;

  Control control;
  auto evaluate = std::bind(Ai::Eval, _1, attack_.get(), &control);
  Plan::iterateAvailablePlans(field, seq, 2, evaluate);
  return DropDecision(control.decision, control.message);
}

void Ai::onGameWillBegin(const FrameRequest& /*frame_request*/) {
  attack_.reset();
}

void Ai::onEnemyGrounded(const FrameRequest& frame_request) {
  const PlainField& enemy = frame_request.enemyPlayerFrameRequest().field;
  CoreField field(enemy);
  field.forceDrop();
  RensaResult result = field.simulate();

  if (result.chains == 0) {
    // TODO: Check required puyos to start RENSA.
    attack_.reset();
    return;
  }

  attack_.reset(new Attack);
  attack_->score = result.score;
  attack_->end_frame_id = frame_request.frameId + result.frames;
}

int Ai::PatternMatch(const RefPlan& plan, std::string* name) {
  int sum = 0;
  int best = 0;

  const CoreField& field = plan.field();
  std::ostringstream oss;
  for (const Pattern& pattern : Pattern::GetAllPattern()) {
    int score = pattern.Match(field);
    sum += score;
    if (score > best) {
      best = score;
      oss << " " << pattern.name() << " " << score << "/" << pattern.score();
    }
  }
  *name = oss.str();

  return sum;
}

void Ai::Eval(const RefPlan& plan, Attack* attack, Control* control) {
  int score = 0;
  int value = 0;
  std::ostringstream oss;
  std::string message;

  value = PatternMatch(plan, &message);
  oss << "Pattern(" << message << "," << value << ")_";
  score += value;

  int future = 0;
  Plan::iterateAvailablePlans(
      plan.field(), KumipuyoSeq(), 1,
      [&future](const RefPlan& p) {
        future = std::max(future, p.rensaResult().score);
      });

  if (plan.isRensaPlan()) {
    const int kAcceptablePuyo = 3;
    if (attack &&
        attack->score >= SCORE_FOR_OJAMA * kAcceptablePuyo &&
        attack->score < plan.score()) {
      value = plan.score();
      oss << "Counter(" << value << ")_";
      score += value;
    }

    value = plan.rensaResult().score;
    oss << "Current(" << value << ")_";
    score += value;

    value = future / 2;
    oss << "Future(" << value << ")";
    score += value;
  } else {
    value = future / 2;
    oss << "Future(" << value << ")";
    score += value;
  }

  if (score > control->score) {
    control->score = score;
    control->message = oss.str();
    control->decision = plan.decisions().front();
  }
}

}  // namespace peria
