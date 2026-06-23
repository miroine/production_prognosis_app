"""
FieldVista — decision-analysis engine
=====================================

A pure, Streamlit-free engine for discrete influence diagrams and the decision
trees they compile to. Mirrors the structure of tools like Prisma:

  * DECISION nodes  — a set of named options the decision-maker chooses.
  * CHANCE nodes    — a set of mutually-exclusive outcomes with probabilities,
                      optionally conditional on parent chance nodes (a CPT).
  * VALUE node      — the objective (e.g. NPV / Cashflow); each tree leaf
                      carries a numeric value, typed by the user or pulled
                      from a FieldVista engine run.

Capabilities
------------
  * Conditional probability tables (a chance node's distribution depends on
    the states of its chance parents).
  * Topological compilation of the diagram into a decision tree following a
    user-given decision order, with informational arcs (a decision made
    *knowing* a chance parent's realised state).
  * Rollback: expected value at chance nodes, max at decision nodes; returns
    the expected value of the optimal policy and the policy itself.
  * EVPI — expected value of perfect information on a chosen chance node.
  * Discrete Bayesian belief updating on a CPT given observed evidence.

All functions are pure (no Streamlit, no I/O) and deterministic, so they are
unit-tested directly in test_decision.py.

Author: Merouane Hamdani. MIT licensed. Not affiliated with or endorsed by
Equinor or any commercial decision-analysis vendor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Optional


FP_DECISION_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Node model
# ---------------------------------------------------------------------------
@dataclass
class DecisionNode:
    """A choice the decision-maker controls."""
    name: str
    options: list                      # list[str] of option labels


@dataclass
class ChanceNode:
    """An uncertainty with discrete, mutually-exclusive outcomes.

    `parents` are the *chance* nodes this node's probabilities are conditioned
    on (the CPT). `cpt` maps a tuple of parent-state labels (in `parents`
    order) to a list of probabilities aligned with `outcomes`. With no parents,
    use the key () for the single unconditional distribution.
    """
    name: str
    outcomes: list                     # list[str]
    parents: list = field(default_factory=list)        # list[str] (chance)
    cpt: dict = field(default_factory=dict)            # {tuple: [p,...]}

    def distribution(self, parent_states: dict) -> list:
        """Return the probability list for the given parent states.

        `parent_states` maps parent name → realised outcome label. Falls back
        to a uniform distribution if the exact CPT row is missing (keeps the
        engine robust to a partially-specified table)."""
        key = tuple(parent_states.get(p) for p in self.parents)
        probs = self.cpt.get(key)
        if probs is None:
            # try the unconditional row, then uniform
            probs = self.cpt.get(())
        if probs is None or len(probs) != len(self.outcomes):
            n = len(self.outcomes)
            return [1.0 / n] * n if n else []
        s = float(sum(probs))
        if s <= 0:
            n = len(self.outcomes)
            return [1.0 / n] * n if n else []
        return [float(p) / s for p in probs]   # normalised


@dataclass
class ValueNode:
    """The objective node. The actual leaf values live in the diagram's
    `leaf_values` map (one value per full scenario), so this just names the
    objective and its units for display."""
    name: str = "Value"
    units: str = "$MM"


# ---------------------------------------------------------------------------
# Influence diagram
# ---------------------------------------------------------------------------
@dataclass
class InfluenceDiagram:
    """A discrete influence diagram.

    `sequence` is the ORDER in which nodes are encountered along a tree branch
    (decision and chance node names interleaved). Informational arcs are
    implied by the sequence: any chance node appearing *before* a decision in
    the sequence is observed before that decision is made. The value node is
    always the leaf.
    """
    decisions: dict = field(default_factory=dict)      # name -> DecisionNode
    chances: dict = field(default_factory=dict)        # name -> ChanceNode
    value: ValueNode = field(default_factory=ValueNode)
    sequence: list = field(default_factory=list)       # ordered node names
    # leaf_values: {scenario_key (frozenset of (node,state)) : value}. Looked
    # up by the most specific matching scenario; missing → 0.0.
    leaf_values: dict = field(default_factory=dict)

    # -- structure helpers --------------------------------------------------
    def node_kind(self, name: str) -> Optional[str]:
        if name in self.decisions:
            return "decision"
        if name in self.chances:
            return "chance"
        return None

    def states_of(self, name: str) -> list:
        if name in self.decisions:
            return list(self.decisions[name].options)
        if name in self.chances:
            return list(self.chances[name].outcomes)
        return []

    def validate(self) -> list:
        """Return a list of human-readable problems (empty = OK)."""
        problems = []
        for nm in self.sequence:
            if self.node_kind(nm) is None:
                problems.append(f"Sequence node '{nm}' is not defined.")
        for nm, c in self.chances.items():
            for p in c.parents:
                if p not in self.chances:
                    problems.append(
                        f"Chance node '{nm}' has parent '{p}' that is not a "
                        f"chance node.")
                # parent must precede child in the sequence
                if (p in self.sequence and nm in self.sequence
                        and self.sequence.index(p) > self.sequence.index(nm)):
                    problems.append(
                        f"Parent '{p}' must come before child '{nm}' in the "
                        f"sequence.")
            # CPT coverage check
            if c.parents:
                parent_state_lists = [self.states_of(p) for p in c.parents]
                for combo in product(*parent_state_lists):
                    if combo not in c.cpt and () not in c.cpt:
                        problems.append(
                            f"Chance node '{nm}' missing CPT row for "
                            f"parents {combo}.")
                        break
        if not self.sequence:
            problems.append("Sequence is empty — add nodes to the tree.")
        return problems


# ---------------------------------------------------------------------------
# Tree compilation
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    """A node in the compiled decision tree."""
    kind: str                          # "decision" | "chance" | "leaf"
    label: str                         # node name or "" for leaf
    # branches: list of (branch_label, probability_or_None, child TreeNode)
    branches: list = field(default_factory=list)
    value: Optional[float] = None      # set on leaves and after rollback
    prob: Optional[float] = None       # probability of reaching (chance edge)
    optimal_branch: Optional[int] = None   # index of chosen branch (decisions)


def compile_tree(diagram: InfluenceDiagram) -> TreeNode:
    """Expand the influence diagram into a decision tree following the
    diagram's `sequence`. Chance nodes use their CPT conditioned on the chance
    states realised earlier on the branch."""
    seq = diagram.sequence

    def build(idx: int, path: dict) -> TreeNode:
        # path: {node_name: realised_state} for nodes earlier on this branch
        if idx >= len(seq):
            val = _leaf_value(diagram, path)
            return TreeNode(kind="leaf", label="", value=val)
        name = seq[idx]
        kind = diagram.node_kind(name)
        if kind == "decision":
            node = TreeNode(kind="decision", label=name)
            for opt in diagram.decisions[name].options:
                child = build(idx + 1, {**path, name: opt})
                node.branches.append((opt, None, child))
            return node
        # chance
        cnode = diagram.chances[name]
        probs = cnode.distribution(path)
        node = TreeNode(kind="chance", label=name)
        for outcome, p in zip(cnode.outcomes, probs):
            child = build(idx + 1, {**path, name: outcome})
            node.branches.append((outcome, float(p), child))
        return node

    return build(0, {})


def _leaf_value(diagram: InfluenceDiagram, path: dict) -> float:
    """Look up the value for a fully-specified scenario `path`.

    Matching is by the most specific subset key present in leaf_values: we try
    the full scenario first, then drop nodes from the end of the sequence until
    a stored key matches. Keys are frozensets of (node, state) pairs."""
    if not diagram.leaf_values:
        return 0.0
    # Build candidate keys from most specific to least.
    items = [(n, path[n]) for n in diagram.sequence if n in path]
    for cut in range(len(items), -1, -1):
        key = frozenset(items[:cut])
        if key in diagram.leaf_values:
            return float(diagram.leaf_values[key])
    # also try exact full-path frozenset
    full = frozenset(path.items())
    if full in diagram.leaf_values:
        return float(diagram.leaf_values[full])
    return 0.0


# ---------------------------------------------------------------------------
# Rollback (expected-value solution)
# ---------------------------------------------------------------------------
def rollback(tree: TreeNode, maximize: bool = True) -> float:
    """Solve the tree by backward induction. Annotates each node's `value`
    (and `optimal_branch` on decisions) in place and returns the root value."""
    if tree.kind == "leaf":
        return float(tree.value or 0.0)
    if tree.kind == "chance":
        ev = 0.0
        for _, p, child in tree.branches:
            ev += (p or 0.0) * rollback(child, maximize)
        tree.value = ev
        return ev
    # decision: choose the best child
    best_val = None
    best_idx = 0
    for i, (_, _, child) in enumerate(tree.branches):
        cv = rollback(child, maximize)
        if best_val is None or (cv > best_val if maximize else cv < best_val):
            best_val = cv
            best_idx = i
    tree.value = best_val if best_val is not None else 0.0
    tree.optimal_branch = best_idx
    return tree.value


def optimal_policy(tree: TreeNode) -> dict:
    """After rollback, walk the optimal branches and return the decision
    policy {decision_name: chosen_option} along the optimal path. (For
    decisions downstream of chance nodes the policy is state-dependent; this
    returns the first-encountered optimal choice per decision, which is the
    headline recommendation.)"""
    policy = {}

    def walk(node):
        if node.kind == "leaf":
            return
        if node.kind == "decision" and node.optimal_branch is not None:
            opt_label, _, child = node.branches[node.optimal_branch]
            policy.setdefault(node.label, opt_label)
            walk(child)
        else:
            # chance: descend the highest-probability branch for the headline
            # path (all branches share the same downstream decisions anyway)
            if node.branches:
                _, _, child = max(node.branches,
                                  key=lambda b: (b[1] or 0.0))
                walk(child)

    walk(tree)
    return policy


# ---------------------------------------------------------------------------
# EVPI — expected value of perfect information
# ---------------------------------------------------------------------------
def evpi_on_chance(diagram: InfluenceDiagram, chance_name: str,
                   maximize: bool = True) -> dict:
    """Expected value of perfect information for resolving `chance_name`
    BEFORE the decisions.

    EVPI = EV(with perfect info) − EV(optimal without info). With perfect
    information the decision-maker learns the outcome first, then optimises;
    we compute this by moving the chance node to the front of the sequence and
    re-solving, then differencing against the base solution."""
    if chance_name not in diagram.chances:
        raise ValueError(f"'{chance_name}' is not a chance node.")
    # Base solution (no extra info)
    base_tree = compile_tree(diagram)
    base_ev = rollback(base_tree, maximize)
    # Perfect-info: put the chance node first, solve, expected over its
    # marginal distribution.
    seq2 = [chance_name] + [n for n in diagram.sequence if n != chance_name]
    d2 = InfluenceDiagram(
        decisions=diagram.decisions, chances=diagram.chances,
        value=diagram.value, sequence=seq2,
        leaf_values=diagram.leaf_values)
    pi_tree = compile_tree(d2)
    pi_ev = rollback(pi_tree, maximize)
    return {"chance": chance_name, "ev_base": base_ev, "ev_perfect": pi_ev,
            "evpi": pi_ev - base_ev if maximize else base_ev - pi_ev}


# ---------------------------------------------------------------------------
# Bayesian belief updating
# ---------------------------------------------------------------------------
def bayes_update(prior: list, likelihood: list) -> list:
    """Posterior ∝ prior × likelihood, normalised. `prior` and `likelihood`
    are aligned lists over the same hypothesis space."""
    if len(prior) != len(likelihood):
        raise ValueError("prior and likelihood must be the same length.")
    unnorm = [float(p) * float(l) for p, l in zip(prior, likelihood)]
    s = sum(unnorm)
    if s <= 0:
        n = len(prior)
        return [1.0 / n] * n if n else []
    return [u / s for u in unnorm]


def joint_to_posterior(chance: ChanceNode, parent_states: dict,
                       observed_outcome: str,
                       parent_priors: dict) -> dict:
    """Given a chance node with a CPT P(node | parents), an OBSERVED node
    outcome, and priors over each parent state, return the posterior over each
    parent via Bayes' rule (single-parent or independent-parents case).

    Returns {parent_name: {state: posterior_prob}}. This is the discrete
    belief-updating primitive the UI exposes (e.g. observing production tells
    you about the geological state that drives it)."""
    posteriors = {}
    try:
        out_idx = chance.outcomes.index(observed_outcome)
    except ValueError:
        return posteriors
    for parent in chance.parents:
        states = list(parent_priors.get(parent, {}).keys())
        if not states:
            continue
        prior = [parent_priors[parent][s] for s in states]
        # likelihood P(observed | parent=state), marginalising other parents
        # uniformly when present (kept simple & transparent for screening).
        like = []
        for s in states:
            ps = dict(parent_states)
            ps[parent] = s
            dist = chance.distribution(ps)
            like.append(dist[out_idx] if out_idx < len(dist) else 0.0)
        post = bayes_update(prior, like)
        posteriors[parent] = {st: pv for st, pv in zip(states, post)}
    return posteriors


# ---------------------------------------------------------------------------
# Convenience: scenario enumeration (for the leaf-value editor)
# ---------------------------------------------------------------------------
def enumerate_leaves(diagram: InfluenceDiagram) -> list:
    """Return every full scenario as an ordered list of (node, state) tuples —
    one per tree leaf — so the UI can present a row per leaf for value entry
    or engine linking."""
    state_lists = [(n, diagram.states_of(n)) for n in diagram.sequence]
    leaves = []
    names = [n for n, _ in state_lists]
    for combo in product(*[s for _, s in state_lists]):
        leaves.append(list(zip(names, combo)))
    return leaves


def leaf_key(scenario: list) -> frozenset:
    """Canonical leaf_values key for a scenario [(node,state),...]."""
    return frozenset(scenario)


def tree_stats(tree: TreeNode) -> dict:
    """Count nodes/leaves for display."""
    n_dec = n_chance = n_leaf = 0
    stack = [tree]
    while stack:
        t = stack.pop()
        if t.kind == "leaf":
            n_leaf += 1
        elif t.kind == "decision":
            n_dec += 1
        elif t.kind == "chance":
            n_chance += 1
        for _, _, ch in t.branches:
            stack.append(ch)
    return {"decisions": n_dec, "chances": n_chance, "leaves": n_leaf}


# ---------------------------------------------------------------------------
# Serialization (save / load a diagram to a plain dict → YAML/JSON)
# ---------------------------------------------------------------------------
def diagram_to_doc(diagram: "InfluenceDiagram") -> dict:
    """Serialise a diagram to a plain, JSON/YAML-safe dict.

    The only non-trivial part is `leaf_values`, whose keys are frozensets of
    (node, state) tuples; we store each as an ordered scenario list so the
    round-trip is exact. CPT keys (tuples of parent states) become lists, with
    the unconditional row keyed by the empty list."""
    decisions = [{"name": d.name, "options": list(d.options)}
                 for d in diagram.decisions.values()]
    chances = []
    for c in diagram.chances.values():
        cpt = []
        for combo, probs in c.cpt.items():
            cpt.append({"parents": list(combo), "probs": list(probs)})
        chances.append({"name": c.name, "outcomes": list(c.outcomes),
                        "parents": list(c.parents), "cpt": cpt})
    leaves = []
    for key, val in diagram.leaf_values.items():
        # key is a frozenset of (node, state); order it by the sequence.
        scn = sorted(key, key=lambda ns: (diagram.sequence.index(ns[0])
                     if ns[0] in diagram.sequence else 999))
        leaves.append({"scenario": [list(ns) for ns in scn],
                       "value": float(val)})
    return {"fieldvista_decision_diagram": {
        "version": FP_DECISION_VERSION,
        "value": {"name": diagram.value.name, "units": diagram.value.units},
        "sequence": list(diagram.sequence),
        "decisions": decisions, "chances": chances, "leaf_values": leaves}}


def doc_to_diagram(doc: dict) -> "InfluenceDiagram":
    """Inverse of diagram_to_doc. Accepts either the wrapped doc or the inner
    dict."""
    root = doc.get("fieldvista_decision_diagram", doc)
    decisions = {}
    for d in root.get("decisions", []):
        decisions[d["name"]] = DecisionNode(d["name"],
                                            list(d.get("options", [])))
    chances = {}
    for c in root.get("chances", []):
        cpt = {}
        for row in c.get("cpt", []):
            cpt[tuple(row.get("parents", []))] = list(row.get("probs", []))
        chances[c["name"]] = ChanceNode(
            c["name"], list(c.get("outcomes", [])),
            parents=list(c.get("parents", [])), cpt=cpt)
    v = root.get("value", {})
    value = ValueNode(v.get("name", "NPV"), v.get("units", "$MM"))
    leaf_values = {}
    for lf in root.get("leaf_values", []):
        scn = [tuple(ns) for ns in lf.get("scenario", [])]
        leaf_values[frozenset(scn)] = float(lf.get("value", 0.0))
    return InfluenceDiagram(
        decisions=decisions, chances=chances, value=value,
        sequence=list(root.get("sequence", [])), leaf_values=leaf_values)
