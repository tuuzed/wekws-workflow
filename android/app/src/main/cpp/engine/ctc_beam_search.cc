#include "engine/ctc_beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace haibingtech {
namespace kws {

CtcBeamSearch::CtcBeamSearch(const CtcConfig& config,
                             const std::vector<Keyword>& keywords)
    : config_(config), keywords_(keywords) {
  token_set_.insert(config_.blank_id);
  for (const Keyword& keyword : keywords_) {
    for (int token : keyword.tokens) token_set_.insert(token);
  }
  Reset();
}

void CtcBeamSearch::Reset() {
  cur_hyps_.clear();
  Entry entry;
  entry.hyp.pb = 1.0;
  entry.hyp.pnb = 0.0;
  cur_hyps_.push_back(entry);
}

std::pair<size_t, bool> CtcBeamSearch::GetOrCreate(
    std::vector<Entry>* entries,
    std::map<std::vector<int>, size_t>* index,
    const std::vector<int>& prefix) const {
  auto found = index->find(prefix);
  if (found != index->end()) {
    return {found->second, false};
  }
  const size_t id = entries->size();
  Entry entry;
  entry.hyp.prefix = prefix;
  entries->push_back(std::move(entry));
  index->emplace(prefix, id);
  return {id, true};
}

void CtcBeamSearch::ProcessFrame(const float* probs, int vocab_size, int frame) {
  // 2.1 First beam prune: top-k tokens, then threshold and keyword filtering.
  std::vector<std::pair<float, int>> top_k;
  top_k.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    top_k.emplace_back(probs[i], i);
  }
  std::sort(top_k.begin(), top_k.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
  if (static_cast<int>(top_k.size()) > config_.score_beam_size) {
    top_k.resize(config_.score_beam_size);
  }

  std::vector<int> filter_index;
  std::vector<double> filter_prob;
  for (const auto& item : top_k) {
    if (item.first > config_.prob_threshold &&
        token_set_.find(item.second) != token_set_.end()) {
      filter_index.push_back(item.second);
      filter_prob.push_back(item.first);
    }
  }
  if (filter_index.empty()) return;

  std::vector<Entry> next;
  next.reserve(cur_hyps_.size() * (filter_index.size() + 1) + 1);
  std::map<std::vector<int>, size_t> index;

  for (size_t fi = 0; fi < filter_index.size(); ++fi) {
    const int s = filter_index[fi];
    const double ps = filter_prob[fi];

    for (const Entry& cur_entry : cur_hyps_) {
      const BeamHyp& hyp = cur_entry.hyp;
      const int last = hyp.prefix.empty() ? -1 : hyp.prefix.back();

      if (s == config_.blank_id) {
        size_t id = GetOrCreate(&next, &index, hyp.prefix).first;
        BeamHyp& next_hyp = next[id].hyp;
        next_hyp.nodes = hyp.nodes;
        next_hyp.pb += hyp.pb * ps + hyp.pnb * ps;
      } else if (s == last) {
        if (!IsCloseToZero(hyp.pnb, config_.isclose_abs_tol)) {
          size_t id = GetOrCreate(&next, &index, hyp.prefix).first;
          BeamHyp& next_hyp = next[id].hyp;
          next_hyp.nodes = hyp.nodes;
          next_hyp.pnb += hyp.pnb * ps;
          if (!next_hyp.nodes.empty() && ps > next_hyp.nodes.back().prob) {
            next_hyp.nodes.back().prob = static_cast<float>(ps);
            next_hyp.nodes.back().frame = frame;
          }
        }
        if (!IsCloseToZero(hyp.pb, config_.isclose_abs_tol)) {
          std::vector<int> new_prefix = hyp.prefix;
          new_prefix.push_back(s);
          size_t id = GetOrCreate(&next, &index, new_prefix).first;
          BeamHyp& next_hyp = next[id].hyp;
          next_hyp.nodes = hyp.nodes;
          BeamNode node;
          node.token = s;
          node.frame = frame;
          node.prob = static_cast<float>(ps);
          next_hyp.nodes.push_back(node);
          next_hyp.pnb += hyp.pb * ps;
        }
      } else {
        std::vector<int> new_prefix = hyp.prefix;
        new_prefix.push_back(s);
        auto created = GetOrCreate(&next, &index, new_prefix);
        BeamHyp& next_hyp = next[created.first].hyp;
        if (!next_hyp.nodes.empty()) {
          if (ps > next_hyp.nodes.back().prob) {
            next_hyp.nodes.pop_back();
            BeamNode node;
            node.token = s;
            node.frame = frame;
            node.prob = static_cast<float>(ps);
            next_hyp.nodes.push_back(node);
          }
        } else {
          next_hyp.nodes = hyp.nodes;
          BeamNode node;
          node.token = s;
          node.frame = frame;
          node.prob = static_cast<float>(ps);
          next_hyp.nodes.push_back(node);
        }
        next_hyp.pnb += hyp.pb * ps + hyp.pnb * ps;
      }
    }
  }

  // 2.2 Second beam prune. stable_sort keeps Python's sorted() tie behavior.
  std::stable_sort(next.begin(), next.end(),
                   [](const Entry& a, const Entry& b) {
                     return Score(a.hyp) > Score(b.hyp);
                   });
  if (static_cast<int>(next.size()) > config_.path_beam_size) {
    next.resize(config_.path_beam_size);
  }
  cur_hyps_ = std::move(next);
}

int CtcBeamSearch::IsSublist(const std::vector<int>& main,
                             const std::vector<int>& check) {
  if (main.size() < check.size()) return -1;
  if (main.size() == check.size()) return main == check ? 0 : -1;
  // Keep the exact upper bound used by wekws/score_ctc.py.
  for (size_t i = 0; i < main.size() - check.size(); ++i) {
    bool matched = true;
    for (size_t j = 0; j < check.size(); ++j) {
      if (main[i + j] != check[j]) {
        matched = false;
        break;
      }
    }
    if (matched) return static_cast<int>(i);
  }
  return -1;
}

bool CtcBeamSearch::FindBestHit(KeywordHit* hit) const {
  for (const Entry& entry : cur_hyps_) {
    const BeamHyp& hyp = entry.hyp;
    for (const Keyword& keyword : keywords_) {
      const int offset = IsSublist(hyp.prefix, keyword.tokens);
      if (offset < 0) continue;
      if (offset + static_cast<int>(keyword.tokens.size()) >
          static_cast<int>(hyp.nodes.size())) {
        continue;
      }

      double score = 1.0;
      for (size_t i = 0; i < keyword.tokens.size(); ++i) {
        score *= static_cast<double>(
            hyp.nodes[static_cast<size_t>(offset) + i].prob);
      }
      KeywordHit result;
      result.keyword = keyword.text;
      result.score = static_cast<float>(std::sqrt(score));
      result.start_frame =
          hyp.nodes[static_cast<size_t>(offset)].frame;
      result.end_frame =
          hyp.nodes[static_cast<size_t>(offset + keyword.tokens.size() - 1)].frame;
      for (size_t i = 0; i < keyword.tokens.size(); ++i) {
        const BeamNode& node =
            hyp.nodes[static_cast<size_t>(offset) + i];
        TokenProb token_prob;
        token_prob.token_id = node.token;
        token_prob.frame = node.frame;
        token_prob.prob = node.prob;
        token_prob.token = i < keyword.chars.size()
                               ? keyword.chars[i]
                               : std::to_string(node.token);
        result.token_probs.push_back(token_prob);
      }
      if (hit != nullptr) *hit = result;
      return true;
    }
  }
  return false;
}

bool CtcBeamSearch::FindBestHits(std::vector<KeywordHit>* hits) const {
  if (hits == nullptr) return false;

  // Match Python/infer_core.py: scan hyps in beam order and keep the first hit
  // for every keyword. Do not select by score: a lower-scored path can rank
  // higher in the final beam and is the one Python reports.
  std::vector<KeywordHit> current(keywords_.size());
  std::vector<bool> found(keywords_.size(), false);

  for (const Entry& entry : cur_hyps_) {
    const BeamHyp& hyp = entry.hyp;
    for (size_t k = 0; k < keywords_.size(); ++k) {
      if (found[k]) continue;
      const Keyword& keyword = keywords_[k];
      const int offset = IsSublist(hyp.prefix, keyword.tokens);
      if (offset < 0) continue;
      if (offset + static_cast<int>(keyword.tokens.size()) >
          static_cast<int>(hyp.nodes.size())) {
        continue;
      }

      double score = 1.0;
      for (size_t i = 0; i < keyword.tokens.size(); ++i) {
        score *= static_cast<double>(
            hyp.nodes[static_cast<size_t>(offset) + i].prob);
      }

      KeywordHit result;
      result.keyword = keyword.text;
      result.score = static_cast<float>(std::sqrt(score));
      result.start_frame = hyp.nodes[static_cast<size_t>(offset)].frame;
      result.end_frame =
          hyp.nodes[static_cast<size_t>(offset + keyword.tokens.size() - 1)]
              .frame;
      for (size_t i = 0; i < keyword.tokens.size(); ++i) {
        const BeamNode& node =
            hyp.nodes[static_cast<size_t>(offset) + i];
        TokenProb token_prob;
        token_prob.token_id = node.token;
        token_prob.frame = node.frame;
        token_prob.prob = node.prob;
        token_prob.token = i < keyword.chars.size()
                               ? keyword.chars[i]
                               : std::to_string(node.token);
        result.token_probs.push_back(token_prob);
      }
      current[k] = std::move(result);
      found[k] = true;
    }
  }

  *hits = std::move(current);
  for (bool ok : found) {
    if (ok) return true;
  }
  return false;
}

}  // namespace kws
}  // namespace haibingtech
