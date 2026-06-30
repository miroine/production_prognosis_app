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

section("8. Diagram serialization round-trip (save/load)")
_doc = dz.diagram_to_doc(diag)
_d2 = dz.doc_to_diagram(_doc)
check("round-trip sequence", _d2.sequence, diag.sequence)
check("round-trip decision options",
      _d2.decisions["Drill?"].options, ["Drill", "Don't"])
check("round-trip chance distribution",
      _d2.chances["Reservoir"].distribution({}), [0.6, 0.4])
# EV must be identical after a save/load cycle.
check("round-trip EV preserved",
      dz.rollback(dz.compile_tree(_d2)), 72.0)
# CPT diagram (conditional) round-trips too.
_doc_c = dz.diagram_to_doc(cd)
_cd2 = dz.doc_to_diagram(_doc_c)
check("round-trip CPT EV = 142.5",
      dz.rollback(dz.compile_tree(_cd2)), 142.5)

section("9. Sensitivity & comparison analytics")
# Reuse the Drill/Don't tree (diag): EV=72, Drill optimal.
# Policy outcomes under optimal (Drill): Wet 0.6→200, Dry 0.4→-120.
_oc = dz.policy_outcomes(diag)
check("policy outcomes count = 2", len(_oc), 2)
check("policy EV from outcomes = 72",
      sum(p * v for p, v in _oc), 72.0)
_st = dz.policy_stats(_oc)
check("policy stats EV = 72", _st["ev"], 72.0)
check("policy P(loss) = 0.4 (Dry is negative)", _st["p_loss"], 0.4)
# Forced 'Don't' first decision → all outcomes 0.
_oc_dont = dz.policy_outcomes(diag, first_decision="Drill?",
                              forced_option="Don't")
check("forced Don't → EV 0", sum(p * v for p, v in _oc_dont), 0.0)
# CDF ends at 1.0
_cdf = dz.cdf_points(_oc)
check("CDF cumulative ends at 1.0", _cdf[-1][1], 1.0, tol=1e-9)
# Probability tornado: Reservoir is the only chance → non-zero swing.
_pt = dz.probability_tornado(diag, delta=0.2)
check("prob tornado has Reservoir", _pt[0]["node"], "Reservoir")
check("prob tornado swing > 0", _pt[0]["swing"] > 0, True)
# Value tornado: 4 leaves, biggest swing from the Drill/Wet payoff.
_vt = dz.value_tornado(diag, delta=0.2)
check("value tornado non-empty", len(_vt) > 0, True)
check("value tornado swing >= 0", _vt[0]["swing"] >= 0, True)
# Decision-flip threshold on Reservoir P(Wet): at low P(Wet) Don't wins,
# at high P(Wet) Drill wins → at least one flip in (0,1).
_ft = dz.decision_flip_threshold(diag, "Reservoir", outcome_index=0)
check("flip threshold first decision = Drill?",
      _ft["first_decision"], "Drill?")
check("decision flips as P(Wet) rises", len(_ft["flips"]) >= 1, True)
# Flip should be near P(Wet) where 200*p -120*(1-p) = 0 → p = 120/320 = 0.375
if _ft["flips"]:
    check("flip near P(Wet)=0.375", _ft["flips"][0]["p"], 0.375, tol=0.03)

section("10. Analytics serialization independence")
# analytics must not mutate the original diagram
_ev_before = dz.rollback(dz.compile_tree(diag))
dz.probability_tornado(diag); dz.value_tornado(diag)
check("diagram EV unchanged after analytics",
      dz.rollback(dz.compile_tree(diag)), _ev_before)

section("11. Risk attitude — utility & certainty equivalent")
# Utility round-trip: CE(U(x)) == x.
check("CE inverts utility (x=100, R=500)",
      dz.certainty_equivalent(dz.exp_utility(100.0, 500.0), 500.0), 100.0)
# Risk-neutral (R=None) → utility solve == EV solve == 72.
_sn = dz.solve(diag, risk_tolerance=None)
check("risk-neutral solve EV = 72", _sn["value"], 72.0)
# Risk-averse: CE of the Drill lottery must be BELOW its EV (72).
# Drill: 0.6·200 + 0.4·(-120). With finite R the CE < 72.
_sa = dz.solve(diag, risk_tolerance=200.0)
check("risk-averse CE < EV (downside penalised)", _sa["value"] < 72.0, True)
# Strong risk aversion can flip the optimal decision to 'Don't' (CE 0).
_sa2 = dz.solve(diag, risk_tolerance=60.0)
check("strong risk aversion flips to Don't",
      _sa2["policy"].get("Drill?"), "Don't")

section("12. EVII — imperfect information")
# Perfect reliability (identity) → EVII == EVPI (= 48).
_perfect = {"Wet": {"Wet": 1.0, "Dry": 0.0},
            "Dry": {"Wet": 0.0, "Dry": 1.0}}
_e1 = dz.evii_on_chance(diag, "Reservoir", _perfect)
check("EVII with perfect test = EVPI = 48", _e1["evii"], 48.0, tol=1e-6)
# Useless test (no discrimination) → EVII = 0.
_useless = {"Wet": {"good": 0.5, "bad": 0.5},
            "Dry": {"good": 0.5, "bad": 0.5}}
_e2 = dz.evii_on_chance(diag, "Reservoir", _useless)
check("EVII with useless test = 0", _e2["evii"], 0.0, tol=1e-6)
# Imperfect test (80% reliable) → 0 < EVII < EVPI.
_imp = {"Wet": {"good": 0.8, "bad": 0.2},
        "Dry": {"good": 0.3, "bad": 0.7}}
_e3 = dz.evii_on_chance(diag, "Reservoir", _imp)
check("imperfect EVII between 0 and EVPI",
      0.0 < _e3["evii"] < 48.0, True)

section("13. Distribution-valued leaves (Swanson)")
_sw = dz.leaf_distribution_branches(10, 50, 90)
check("Swanson weights 0.3/0.4/0.3", [w for _, w in _sw], [0.30, 0.40, 0.30])
check("Swanson mean = 0.3·10+0.4·50+0.3·90 = 50",
      sum(v * w for v, w in _sw), 50.0)
# Expand a leaf into a distribution and check EV shifts by its mean.
_dl = {dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Wet")]):
       (150.0, 200.0, 260.0)}   # mean = 0.3·150+0.4·200+0.3·260 = 203
_dd = dz.expand_distribution_leaves(diag, _dl)
# new EV(Drill) = 0.6·203 + 0.4·(-120) = 121.8 - 48 = 73.8
check("distribution-expanded EV = 73.8",
      dz.rollback(dz.compile_tree(_dd)), 73.8, tol=1e-6)

section("14. Monte-Carlo sampling of policy outcomes")
# Sampling the Drill tree (Wet 0.6→200, Dry 0.4→-120) should reproduce the
# analytic mean (72) and the two outcome values.
_samp = dz.sample_policy_outcomes(diag, n=20000, seed=1)
check("MC sample count = 20000", len(_samp), 20000)
_mean = sum(_samp) / len(_samp)
check("MC mean ≈ analytic EV 72", _mean, 72.0, tol=3.0)
check("MC only yields leaf values {200,-120}",
      set(round(v) for v in _samp) <= {200, -120}, True)
_pct = dz.percentiles(_samp)
# P10 should be the low (-120 region), P90 the high (200)
check("MC P90 = 200 (upside)", _pct[90], 200.0)
check("MC P10 = -120 (downside)", _pct[10], -120.0)

section("15. Multi-objective (NPV − carbon·CO2)")
# Add CO2 to the Drill leaves: Drilling emits, Don't emits nothing.
_co2 = {
    dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Wet")]): 100.0,
    dz.leaf_key([("Drill?", "Drill"), ("Reservoir", "Dry")]): 100.0,
    dz.leaf_key([("Drill?", "Don't"), ("Reservoir", "Wet")]): 0.0,
    dz.leaf_key([("Drill?", "Don't"), ("Reservoir", "Dry")]): 0.0,
}
# At carbon_price=0 → same as before (Drill, EV 72).
_m0 = dz.solve_multiobjective(diag, _co2, carbon_price=0.0)
check("multiobj @0 carbon = EV 72", _m0["value"], 72.0)
check("multiobj @0 picks Drill", _m0["policy"].get("Drill?"), "Drill")
# At a high carbon price the 100 t CO2 penalty makes Drill's combined value
# negative → flips to Don't. Combined Drill EV = 72 − cp·100.
# Flip when 72 − 100·cp < 0 → cp > 0.72.
_m1 = dz.solve_multiobjective(diag, _co2, carbon_price=1.0)
check("multiobj @ high carbon flips to Don't",
      _m1["policy"].get("Drill?"), "Don't")
check("multiobj reports EV primary (NPV) separately",
      _m0["ev_primary"], 72.0)
# Flip scan should find the crossover near 0.72.
_scan = dz.carbon_price_flip_scan(diag, _co2, price_max=2.0, steps=81)
check("carbon flip scan finds a flip", len(_scan["flips"]) >= 1, True)
if _scan["flips"]:
    check("carbon flip near 0.72", _scan["flips"][0]["carbon_price"], 0.72,
          tol=0.05)

print(f"\n{'='*52}")
print(f"DECISION-ANALYSIS TESTS: {_passed} passed, {_failed} failed")
print(f"{'='*52}")
sys.exit(0 if _failed == 0 else 1)
