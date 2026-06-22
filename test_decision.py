"""
Unit tests for the decision-analysis engine (fp_decision.py).

Hand-verified against textbook decision-tree / EVPI / Bayes references.
"""
import sys
import importlib.util

_spec = importlib.util.spec_from_file_location("fp_decision", "fp_decision.py")
dz = importlib.util.module_from_spec(_spec)
sys.modules["fp_decision"] = dz
_spec.loader.exec_module(dz)

_passed = 0
_failed = 0


def check(name, got, expected, tol=1e-6, rel=False):
    global _passed, _failed
    try:
        if isinstance(expected, bool) or isinstance(got, bool):
            ok = (got == expected)
        elif isinstance(expected, (int, float)):
            denom = abs(expected) if (rel and expected) else 1.0
            ok = abs(float(got) - float(expected)) / denom <= tol
        else:
            ok = (got == expected)
    except Exception:
        ok = False
    if ok:
        _passed += 1
        print(f"  [PASS] {name}: {got}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}: got {got}, expected {expected}")


def section(t):
    print(f"\n{t}")


# ===========================================================================
section("1. Classic single-decision / single-chance tree")
# Decision: Drill | Don't.  Don't = 0.
# Drill → chance Reservoir: Wet (p=0.6, +200), Dry (p=0.4, -120).
# EV(Drill) = 0.6*200 + 0.4*(-120) = 120 - 48 = 72.  EV(Don't)=0.
# Optimal = Drill, EV = 72.
dec = dz.DecisionNode("Drill?", ["Drill", "Don't"])
ch = dz.ChanceNode("Reservoir", ["Wet", "Dry"], parents=[],
                   cpt={(): [0.6, 0.4]})
val = dz.ValueNode("NPV", "$MM")
leaf = {
    dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Wet")]): 200.0,
    dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Dry")]): -120.0,
    dz.leaf_key([("Drill?", "Don't"), ("Reservoir", "Wet")]): 0.0,
    dz.leaf_key([("Drill?", "Don't"), ("Reservoir", "Dry")]): 0.0,
}
diag = dz.InfluenceDiagram(
    decisions={"Drill?": dec}, chances={"Reservoir": ch}, value=val,
    sequence=["Drill?", "Reservoir"], leaf_values=leaf)
check("validate clean", diag.validate(), [])
tree = dz.compile_tree(diag)
ev = dz.rollback(tree, maximize=True)
check("EV(optimal) = 72", ev, 72.0)
pol = dz.optimal_policy(tree)
check("optimal decision = Drill", pol.get("Drill?"), "Drill")
stats = dz.tree_stats(tree)
check("leaf count = 4", stats["leaves"], 4)

section("2. Decision flips when economics change")
# Make Wet payoff only +100: EV(Drill)=0.6*100+0.4*(-120)=60-48=12 (>0 still)
leaf2 = dict(leaf)
leaf2[dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Wet")])] = 100.0
diag2 = dz.InfluenceDiagram(
    decisions={"Drill?": dec}, chances={"Reservoir": ch}, value=val,
    sequence=["Drill?", "Reservoir"], leaf_values=leaf2)
check("EV with +100 wet = 12", dz.rollback(dz.compile_tree(diag2)), 12.0)
# Now -50 dry payoff worse: Wet +100/Dry -200 → 0.6*100+0.4*-200=60-80=-20 <0
leaf3 = dict(leaf2)
leaf3[dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Dry")])] = -200.0
diag3 = dz.InfluenceDiagram(
    decisions={"Drill?": dec}, chances={"Reservoir": ch}, value=val,
    sequence=["Drill?", "Reservoir"], leaf_values=leaf3)
t3 = dz.compile_tree(diag3)
check("EV picks Don't (=0) when Drill EV<0", dz.rollback(t3), 0.0)
check("policy = Don't", dz.optimal_policy(t3).get("Drill?"), "Don't")

section("3. EVPI on the reservoir uncertainty")
# Perfect info: learn Wet/Dry first.
#   If Wet: max(Drill=200, Don't=0) = 200
#   If Dry: max(Drill=-120, Don't=0) = 0
# EV(perfect) = 0.6*200 + 0.4*0 = 120.  Base EV = 72.  EVPI = 48.
e = dz.evpi_on_chance(diag, "Reservoir", maximize=True)
check("EV perfect info = 120", e["ev_perfect"], 120.0)
check("EV base = 72", e["ev_base"], 72.0)
check("EVPI = 48", e["evpi"], 48.0)

section("4. Conditional probability table (CPT)")
# Geology G: Good(0.5)/Poor(0.5). Production P depends on G:
#   P(High|Good)=0.8, P(Low|Good)=0.2 ; P(High|Poor)=0.3, P(Low|Poor)=0.7
g = dz.ChanceNode("Geology", ["Good", "Poor"], parents=[],
                  cpt={(): [0.5, 0.5]})
p = dz.ChanceNode("Production", ["High", "Low"], parents=["Geology"],
                  cpt={("Good",): [0.8, 0.2], ("Poor",): [0.3, 0.7]})
# value only depends on production: High=+300, Low=-50
lv = {}
for gs in ("Good", "Poor"):
    for ps in ("High", "Low"):
        v = 300.0 if ps == "High" else -50.0
        lv[dz.leaf_key([("Geology", gs), ("Production", ps)])] = v
cd = dz.InfluenceDiagram(
    decisions={}, chances={"Geology": g, "Production": p}, value=val,
    sequence=["Geology", "Production"], leaf_values=lv)
check("CPT validate clean", cd.validate(), [])
# EV = P(High)*300 + P(Low)*-50.
# P(High) = 0.5*0.8 + 0.5*0.3 = 0.55 ; P(Low)=0.45
# EV = 0.55*300 + 0.45*(-50) = 165 - 22.5 = 142.5
check("EV with CPT = 142.5", dz.rollback(dz.compile_tree(cd)), 142.5)

section("5. Bayesian belief updating")
# Same G→P. Observe Production=High. Posterior over Geology:
#   P(Good|High) ∝ P(High|Good)P(Good) = 0.8*0.5 = 0.40
#   P(Poor|High) ∝ P(High|Poor)P(Poor) = 0.3*0.5 = 0.15
#   normalise: 0.40/0.55 = 0.7273 ; 0.15/0.55 = 0.2727
post = dz.joint_to_posterior(
    p, parent_states={}, observed_outcome="High",
    parent_priors={"Geology": {"Good": 0.5, "Poor": 0.5}})
check("posterior Good|High ≈ 0.7273", post["Geology"]["Good"], 0.72727,
      tol=1e-4)
check("posterior Poor|High ≈ 0.2727", post["Geology"]["Poor"], 0.27273,
      tol=1e-4)
# direct bayes_update primitive
bu = dz.bayes_update([0.5, 0.5], [0.8, 0.3])
check("bayes_update normalised sums to 1", sum(bu), 1.0)
check("bayes_update[0] ≈ 0.7273", bu[0], 0.72727, tol=1e-4)

section("6. Distribution normalisation + fallback")
cn = dz.ChanceNode("X", ["a", "b", "c"], parents=[], cpt={(): [2, 2, 4]})
check("unnormalised CPT normalises", cn.distribution({}), [0.25, 0.25, 0.5])
cn2 = dz.ChanceNode("Y", ["a", "b"], parents=["Z"], cpt={})  # missing
check("missing CPT → uniform", cn2.distribution({"Z": "q"}), [0.5, 0.5])

section("7. Leaf enumeration matches tree leaves")
leaves = dz.enumerate_leaves(diag)
check("enumerate_leaves count = tree leaves",
      len(leaves), dz.tree_stats(dz.compile_tree(diag))["leaves"])

print(f"\n{'='*52}")
print(f"DECISION-ANALYSIS TESTS: {_passed} passed, {_failed} failed")
print(f"{'='*52}")
sys.exit(0 if _failed == 0 else 1)
