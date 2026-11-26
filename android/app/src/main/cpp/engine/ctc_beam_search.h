#ifndef HAIBINGTECH_KWS_CTC_BEAM_SEARCH_H_
#define HAIBINGTECH_KWS_CTC_BEAM_SEARCH_H_

#include <cmath>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/runtime_config.h"

namespace haibingtech {
namespace kws {

struct BeamNode {
  int token = 0;
  int frame = 0;
  float prob = 0.0f;
};

struct TokenProb {
  std::string token;
  int token_id = 0;
  int frame = 0;
  float prob = 0.0f;
};

struct BeamHyp {
  std::vector<int> prefix;
  double pb = 0.0;
  double pnb = 0.0;
  std::vector<BeamNode> nodes;
};

struct KeywordHit {
  std::string keyword;
  float score = 0.0f;
  int start_frame = 0;
  int end_frame = 0;
  std::vector<TokenProb> token_probs;
};

// CTC prefix beam search ported from wekws/model/loss.py.
// The first version intentionally preserves Python edge cases, including the
// is_sublist() upper bound and the top-k-before-threshold behavior.
class CtcBeamSearch {
 public:
  CtcBeamSearch(const CtcConfig& config, const std::vector<Keyword>& keywords);

  void ProcessFrame(const float* probs, int vocab_size, int frame);
  bool FindBestHit(KeywordHit* hit) const;
  // Fills the first hit in current beam order for each configured keyword.
  // Unmatched keywords keep an empty KeywordHit, so callers can index hits by
  // keyword position. This mirrors Python/infer_core.py and deliberately does
  // not select the highest score across hyps.
  bool FindBestHits(std::vector<KeywordHit>* hits) const;
  void Reset();

  size_t num_hyps() const { return cur_hyps_.size(); }

 private:
  struct Entry {
    BeamHyp hyp;
  };

  std::pair<size_t, bool> GetOrCreate(
      std::vector<Entry>* entries,
      std::map<std::vector<int>, size_t>* index,
      const std::vector<int>& prefix) const;

  static double Score(const BeamHyp& hyp) { return hyp.pb + hyp.pnb; }
  static bool IsCloseToZero(double value, double abs_tol) {
    return std::fabs(value) <= abs_tol;
  }
  static int IsSublist(const std::vector<int>& main,
                       const std::vector<int>& check);

  CtcConfig config_;
  std::vector<Keyword> keywords_;
  std::unordered_set<int> token_set_;
  std::vector<Entry> cur_hyps_;
};

}  // namespace kws
}  // namespace haibingtech

#endif  // HAIBINGTECH_KWS_CTC_BEAM_SEARCH_H_
