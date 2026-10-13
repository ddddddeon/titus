package matcher

import (
	"sort"
	"strings"

	"github.com/praetorian-inc/titus/pkg/types"
)

// CrossRuleDeduplicator suppresses redundant matches across different rules
// within the same blob when they detect the same underlying secret.
//
// When multiple rules (e.g., np.aws.1, np.aws.2, np.aws.6) match overlapping
// credential data in the same file, only the most informative match is kept.
// Matches are clustered by shared captured group values, then scored.
type CrossRuleDeduplicator struct {
	rules       map[string]*types.Rule
	canValidate func(ruleID string) bool
}

// NewCrossRuleDeduplicator creates a deduplicator with rule metadata and
// validator awareness.
func NewCrossRuleDeduplicator(
	rules map[string]*types.Rule,
	canValidate func(ruleID string) bool,
) *CrossRuleDeduplicator {
	return &CrossRuleDeduplicator{
		rules:       rules,
		canValidate: canValidate,
	}
}

// SetCanValidate updates the validator awareness function after construction.
// Passing nil disables validator-based scoring (all rules treated equally).
func (d *CrossRuleDeduplicator) SetCanValidate(fn func(ruleID string) bool) {
	d.canValidate = fn
}

// Deduplicate takes all matches from a single blob and returns only the
// non-redundant subset. Matches are clustered by shared captured group values,
// then for each cluster the most informative match is kept. Finally a generic
// catch-all match is dropped when a provider-specific match overlaps it and
// contains its captured value.
func (d *CrossRuleDeduplicator) Deduplicate(matches []*types.Match) []*types.Match {
	if len(matches) <= 1 {
		return matches
	}

	clusters := d.clusterBySharedValues(matches)

	result := make([]*types.Match, 0, len(matches))
	for _, cluster := range clusters {
		if len(cluster) == 1 {
			result = append(result, cluster[0])
			continue
		}
		winner := d.pickWinner(cluster)
		result = append(result, winner)
	}
	return d.dropRedundantGenerics(result)
}

// dropRedundantGenerics removes a generic match when a provider-specific match
// overlaps it and fully contains its captured value. A generic catch-all rule
// frequently captures only a substring of a secret that a specific rule already
// reports in full (for example an Anthropic key seen as a truncated generic API
// key). Exact-value clustering does not catch this because the captured values
// differ. Filtering the generic out, rather than clustering it, means two
// independent provider-specific findings are never merged or discarded, even
// when a single short generic value is a substring of both.
func (d *CrossRuleDeduplicator) dropRedundantGenerics(matches []*types.Match) []*types.Match {
	specific := make([]*types.Match, 0, len(matches))
	for _, m := range matches {
		if !d.isGeneric(m) {
			specific = append(specific, m)
		}
	}
	if len(specific) == 0 {
		return matches
	}

	kept := make([]*types.Match, 0, len(matches))
	for _, m := range matches {
		if d.isGeneric(m) && shadowedBySpecific(m, specific) {
			continue
		}
		kept = append(kept, m)
	}
	return kept
}

// shadowedBySpecific reports whether a specific match overlaps the generic
// match and contains its captured value, making the generic redundant.
func shadowedBySpecific(generic *types.Match, specific []*types.Match) bool {
	for _, s := range specific {
		if spansOverlap(generic, s) && groupValueContains(s, generic) {
			return true
		}
	}
	return false
}

// clusterBySharedValues groups matches whose captured group values overlap.
// Uses union-find: if match A and match B share any exact group value,
// they belong to the same cluster.
func (d *CrossRuleDeduplicator) clusterBySharedValues(matches []*types.Match) [][]*types.Match {
	n := len(matches)
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}

	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]] // path compression
			x = parent[x]
		}
		return x
	}

	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}

	// Map: group value → index of first match that has it
	valueIndex := make(map[string]int)

	for i, m := range matches {
		for _, g := range m.Groups {
			val := string(g)
			if val == "" {
				continue
			}
			if prev, exists := valueIndex[val]; exists {
				union(i, prev)
			} else {
				valueIndex[val] = i
			}
		}
	}

	// Collect clusters by root
	clusterMap := make(map[int][]*types.Match)
	for i, m := range matches {
		root := find(i)
		clusterMap[root] = append(clusterMap[root], m)
	}

	clusters := make([][]*types.Match, 0, len(clusterMap))
	for _, c := range clusterMap {
		clusters = append(clusters, c)
	}

	// Sort clusters deterministically by minimum match start offset.
	// This eliminates nondeterminism from Go's map iteration order.
	sort.Slice(clusters, func(i, j int) bool {
		return minStartOffset(clusters[i]) < minStartOffset(clusters[j])
	})

	return clusters
}

// minStartOffset returns the smallest Location.Offset.Start among all matches
// in the cluster. Used to produce a deterministic cluster ordering.
func minStartOffset(cluster []*types.Match) int64 {
	min := cluster[0].Location.Offset.Start
	for _, m := range cluster[1:] {
		if m.Location.Offset.Start < min {
			min = m.Location.Offset.Start
		}
	}
	return min
}

// pickWinner selects the most informative match from a cluster.
// Priority: has validator > non-generic rule > group count > total captured length > pattern length.
func (d *CrossRuleDeduplicator) pickWinner(cluster []*types.Match) *types.Match {
	best := 0
	bestScore := d.score(cluster[0])
	for i := 1; i < len(cluster); i++ {
		s := d.score(cluster[i])
		if s.Better(bestScore) {
			best = i
			bestScore = s
		}
	}
	return cluster[best]
}

// score computes the ranking criteria for a match.
func (d *CrossRuleDeduplicator) score(m *types.Match) matchScore {
	hasValidator := false
	if d.canValidate != nil {
		hasValidator = d.canValidate(m.RuleID)
	}

	groupsLen := 0
	for _, g := range m.Groups {
		groupsLen += len(g)
	}

	patternLen := 0
	isGeneric := false
	if r, ok := d.rules[m.RuleID]; ok {
		patternLen = len(r.Pattern)
		isGeneric = ruleHasCategory(r, "generic")
	}

	return matchScore{
		hasValidator: hasValidator,
		isGeneric:    isGeneric,
		groupCount:   len(m.Groups),
		groupsLen:    groupsLen,
		patternLen:   patternLen,
		ruleID:       m.RuleID,
	}
}

// isGeneric reports whether the match's rule carries the "generic" category.
func (d *CrossRuleDeduplicator) isGeneric(m *types.Match) bool {
	if r, ok := d.rules[m.RuleID]; ok {
		return ruleHasCategory(r, "generic")
	}
	return false
}

// spansOverlap reports whether the two matches cover overlapping byte ranges.
func spansOverlap(a, b *types.Match) bool {
	return a.Location.Offset.Start < b.Location.Offset.End &&
		b.Location.Offset.Start < a.Location.Offset.End
}

// groupValueContains reports whether some non-empty captured value of inner is
// a substring of a captured value of outer.
func groupValueContains(outer, inner *types.Match) bool {
	for _, gi := range inner.Groups {
		vi := string(gi)
		if vi == "" {
			continue
		}
		for _, go_ := range outer.Groups {
			vo := string(go_)
			if vo == "" {
				continue
			}
			if strings.Contains(vo, vi) {
				return true
			}
		}
	}
	return false
}

// ruleHasCategory reports whether the rule is tagged with the given category.
func ruleHasCategory(r *types.Rule, category string) bool {
	for _, c := range r.Categories {
		if c == category {
			return true
		}
	}
	return false
}

// matchScore captures the ranking criteria for comparing matches.
type matchScore struct {
	hasValidator bool
	isGeneric    bool // rule carries the "generic" category (catch-all)
	groupCount   int
	groupsLen    int
	patternLen   int
	ruleID       string // used as deterministic tiebreaker
}

// Better returns true if s is ranked higher than other.
func (s matchScore) Better(other matchScore) bool {
	if s.hasValidator != other.hasValidator {
		return s.hasValidator
	}
	// A provider-specific rule beats a generic catch-all rule (e.g. np.aws.2
	// over np.generic.2). Generic rules exist to catch credentials that no
	// specific rule recognises, so whenever a specific rule also fires on the
	// same secret it is the more informative match.
	if s.isGeneric != other.isGeneric {
		return !s.isGeneric
	}
	if s.groupCount != other.groupCount {
		return s.groupCount > other.groupCount
	}
	if s.groupsLen != other.groupsLen {
		return s.groupsLen > other.groupsLen
	}
	if s.patternLen != other.patternLen {
		return s.patternLen > other.patternLen
	}
	// Deterministic tiebreaker: prefer lexicographically smaller RuleID.
	// This ensures identical-score matches always resolve to the same winner.
	return s.ruleID < other.ruleID
}
