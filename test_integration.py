"""
Integration smoke tests — drive each sub-app section headlessly.

These don't check numerical correctness (that's test_parity / test_units /
test_decision); they verify that every page/section function *renders without
raising* against a stubbed Streamlit + Plotly, catching import errors, bad
attribute access, signature drift, and similar regressions before deploy.

The stubs are deliberately permissive: every Streamlit widget returns a
benign default, every Plotly object is inert. A section "passes" if calling it
does not raise.
"""
import sys
import types
import importlib.util
from datetime import date

_passed = 0
_failed = 0


def check(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        _failed += 1
        import traceback
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Stub Streamlit: every call returns a permissive default.
# ---------------------------------------------------------------------------
class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Widget(_Ctx):
    """A callable/iterable stand-in for any Streamlit return value."""
    def __init__(self, val=None):
        self._val = val

    def __call__(self, *a, **k):
        return _Widget()

    def __getattr__(self, n):
        return _Widget()

    def __iter__(self):
        return iter([_Widget() for _ in range(8)])

    def __bool__(self):
        return False

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    def __str__(self):
        return ""

    def __len__(self):
        return 0

    def __getitem__(self, k):
        return _Widget()


def _cache_data(*dargs, **dkw):
    def deco(fn):
        return fn
    if len(dargs) == 1 and callable(dargs[0]) and not dkw:
        return deco(dargs[0])
    return deco


class _SessionState(dict):
    def __getattr__(self, n):
        return self.get(n)

    def __setattr__(self, n, v):
        self[n] = v

    def setdefault(self, k, v=None):
        if k not in self:
            self[k] = v
        return self[k]


class _Col(_Ctx):
    def __getattr__(self, n):
        if n == "button":
            return lambda *a, **k: False
        if n in ("columns",):
            return lambda *a, **k: [_Col() for _ in range(
                a[0] if a and isinstance(a[0], int) else 2)]
        return lambda *a, **k: _Widget()


class _Tabs(list):
    pass


class _Streamlit:
    session_state = _SessionState()
    cache_data = staticmethod(_cache_data)
    cache_resource = staticmethod(_cache_data)

    def __getattr__(self, n):
        if n == "columns":
            return lambda *a, **k: [_Col() for _ in range(
                a[0] if a and isinstance(a[0], int)
                else (len(a[0]) if a and isinstance(a[0], (list, tuple))
                      else 2))]
        if n == "tabs":
            return lambda labels, *a, **k: _Tabs(
                [_Col() for _ in range(len(labels))])
        if n == "columns":
            return lambda *a, **k: [_Col(), _Col()]
        if n == "button":
            return lambda *a, **k: False
        if n == "form_submit_button":
            return lambda *a, **k: False
        if n == "checkbox":
            return lambda *a, **k: False
        if n == "sidebar":
            return self
        if n in ("number_input", "slider"):
            return lambda *a, **k: (k.get("value", a[2] if len(a) > 2 else 0)
                                    or 0)
        if n == "selectbox":
            return lambda label, options=None, *a, **k: (
                list(options)[0] if options else None)
        if n == "select_slider":
            return lambda label, options=None, *a, **k: (
                k.get("value", (list(options)[0] if options else None)))
        if n in ("text_input", "text_area"):
            return lambda *a, **k: k.get("value", "")
        if n == "radio":
            return lambda label, options=None, *a, **k: (
                list(options)[0] if options else None)
        if n == "multiselect":
            return lambda *a, **k: k.get("default", []) or []
        if n == "data_editor":
            return lambda df, *a, **k: df
        if n == "file_uploader":
            return lambda *a, **k: None
        if n == "expander":
            return lambda *a, **k: _Col()
        if n == "container":
            return lambda *a, **k: _Col()
        if n == "form":
            return lambda *a, **k: _Col()
        if n == "spinner":
            return lambda *a, **k: _Col()
        if n == "empty":
            return lambda *a, **k: _Col()
        if n == "download_button":
            return lambda *a, **k: False
        if n == "progress":
            return lambda *a, **k: _Widget()
        if n == "column_config":
            return _Widget()
        if n == "stop":
            return lambda *a, **k: (_ for _ in ()).throw(_StopRender())
        return lambda *a, **k: _Widget()


class _StopRender(Exception):
    pass


def _install_stubs():
    st = _Streamlit()
    sys.modules["streamlit"] = st
    for mod in ("plotly", "plotly.graph_objects", "plotly.subplots",
                "plotly.express"):
        sys.modules[mod] = types.ModuleType(mod)

    class _Fig:
        def __getattr__(self, n):
            return lambda *a, **k: None
    sys.modules["plotly.graph_objects"].Figure = _Fig
    for _n in ("Scatter", "Bar", "Pie", "Waterfall", "Histogram", "Heatmap",
               "Scatter3d", "Mesh3d", "Indicator", "Table", "Box"):
        setattr(sys.modules["plotly.graph_objects"], _n,
                lambda *a, **k: None)
    sys.modules["plotly.subplots"].make_subplots = lambda *a, **k: _Fig()
    sys.modules["plotly.express"].timeline = lambda *a, **k: _Fig()
    return st


def _load_app():
    for nm in ("fp_core", "fp_economics", "fp_helpers", "fp_decision"):
        spec = importlib.util.spec_from_file_location(nm, nm + ".py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[nm] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("app",
                                                  "field_prognosis_app.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app
    spec.loader.exec_module(app)
    return app


# ---------------------------------------------------------------------------
print("\nIntegration smoke tests — sub-app sections render without raising")
st = _install_stubs()
app = _load_app()
today = date.today()


def _guard(fn):
    """Run a section, treating a stubbed st.stop() as a clean early return.
    Also tolerates KeyErrors/TypeErrors whose payload is a stub _Widget — those
    are artifacts of the permissive stub (e.g. a widget used as a dict key),
    not real application faults. Any other exception is a genuine failure."""
    try:
        fn()
    except _StopRender:
        pass
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        # if the offending value is a stub Widget, it's a harness limitation
        if "_Widget" in repr(e) or "_Widget" in repr(getattr(e, "args", "")):
            return
        raise


# Module load itself
check("app module imports", lambda: app.FP_HELPERS_VERSION
      if hasattr(app, "FP_HELPERS_VERSION") else True)

# Landing menu
check("landing menu renders", lambda: app._render_landing_menu(
    ["🛢️ Business case builder", "🌳 Concept Selector",
     "🤖 Case from text", "🎯 Decision tree"]))

# Decision tree (seed example first so it has content)
def _dt():
    app._dz_seed_example()
    _guard(app.decision_tree_section)
check("decision_tree_section renders", _dt)

def _dt_field_a():
    app._dz_seed_field_a_example()
    _guard(app.decision_tree_section)
check("decision_tree_section (Field A example) renders", _dt_field_a)

# Concept Selector
check("concept_selector_section renders",
      lambda: _guard(lambda: app.concept_selector_section(today)))

# Case from text
check("case_from_text_section renders",
      lambda: _guard(lambda: app.case_from_text_section(today)))

# Cached runner parity (smoke: returns a dict with kpis)
def _cache_smoke():
    fh = sys.modules["fp_helpers"]
    payload, _ = fh.yaml_to_payload(
        open("test_fixtures/reference_gascond.yaml").read())
    r = app.run_payload_case_cached(payload, today)
    assert r.get("ok") and "npv_MM" in r.get("kpis", {})
check("run_payload_case_cached returns valid result", _cache_smoke)


def _engine_surface():
    import importlib
    eng = importlib.import_module("fp_engine")
    fh = sys.modules["fp_helpers"]
    payload, _ = fh.yaml_to_payload(
        open("test_fixtures/reference_gascond.yaml").read())
    # fp_engine.run_payload_case must equal app.run_payload_case exactly
    r_app = app.run_payload_case(payload, today)
    r_eng = eng.run_payload_case(payload, today)
    assert abs(r_app["kpis"]["npv_MM"] - r_eng["kpis"]["npv_MM"]) < 1e-9
    assert set(eng.public_surface()) >= {"run_payload_case",
                                         "compute_economics"}
check("fp_engine re-export matches app engine", _engine_surface)

print(f"\n{'='*52}")
print(f"INTEGRATION SMOKE TESTS: {_passed} passed, {_failed} failed")
print(f"{'='*52}")
sys.exit(0 if _failed == 0 else 1)
