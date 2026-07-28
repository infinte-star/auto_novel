"""The actionability invariant, enforced at gate-registration time.

REDESIGN_V2 §3.4 ③. Two defect classes cost this repo real FPY' before they were
named, and both are discoverable at registration rather than by post-mortem:

  **Latching gate** — a blocking verdict conditioned on state the current attempt
  cannot change. Every forced retry it buys is a guaranteed first-pass failure.
  Three measured instances (`book_wide_fossils` on a frozen book-cumulative
  ratio, `chapter_mode_monotony` counting a genre label, `CONTRACT_SYSTEM`
  fabricating an ability whitelist) cost 4.0pt of library FPY' between them.
  `scope` makes it structural: a `book`-scope quantity may advise, never reject.

  **Dead key** — a threshold that sits outside the measured distribution, so it
  either never fires (`fingerprint_warn_threshold`, 0.65 against a measured max
  of 0.448 — deleted as unreachable) or always does (`book_fossil_hard_ratio`
  0.20, below its own 0.30 candidacy floor). `proof` makes stating the
  distribution a precondition of registering at all.

These tests are about the REGISTRY's contract, not about any gate's numbers.
Zero LLM calls.
"""
import unittest

import quality
from quality import GATE_SCOPES, REGISTRY, REPAIR_LAYERS, GateRegistry


class RegisterValidationTest(unittest.TestCase):
    """`register` must reject the two defect shapes rather than accept them."""

    def setUp(self):
        self.reg = GateRegistry()

    def _register(self, **kw):
        kw.setdefault("config_key", "x_enabled")
        return self.reg.register("x", **kw)(lambda *a, **k: {})

    def test_missing_proof_is_a_registration_error(self):
        with self.assertRaises(ValueError) as cm:
            self._register()
        self.assertIn("proof", str(cm.exception))

    def test_blank_proof_is_not_a_proof(self):
        for blank in ("", "   ", "\n\t"):
            with self.subTest(blank=repr(blank)):
                with self.assertRaises(ValueError):
                    self._register(proof=blank)

    def test_unknown_scope_rejected(self):
        with self.assertRaises(ValueError):
            self._register(scope="novel", proof="measured")

    def test_unknown_repair_layer_rejected(self):
        with self.assertRaises(ValueError):
            self._register(repair="L9", proof="measured")

    def test_valid_registration_round_trips(self):
        self._register(scope="card", repair="L2", proof="  fired 3/40  ")
        self.assertEqual(self.reg.scope("x"), "card")
        self.assertEqual(self.reg.repair("x"), "L2")
        self.assertEqual(self.reg.proof("x"), "fired 3/40")   # stripped


class MayBlockTest(unittest.TestCase):
    """`may_block` is the invariant in one predicate."""

    def setUp(self):
        self.reg = GateRegistry()

    def _add(self, name, scope, repair):
        self.reg.register(name, config_key=f"{name}_enabled", scope=scope,
                          repair=repair, proof="measured")(lambda *a, **k: {})

    def test_book_scope_never_blocks_even_with_a_repair_layer(self):
        self._add("g", "book", "L0")
        self.assertFalse(self.reg.may_block("g"))

    def test_advisory_never_blocks_even_at_chapter_scope(self):
        self._add("g", "chapter", "advisory")
        self.assertFalse(self.reg.may_block("g"))

    def test_chapter_and_card_scope_with_a_repair_may_block(self):
        for scope in ("chapter", "card"):
            for repair in ("L0", "L1", "L2"):
                with self.subTest(scope=scope, repair=repair):
                    reg = GateRegistry()
                    reg.register("g", config_key="g_enabled", scope=scope,
                                 repair=repair, proof="m")(lambda *a, **k: {})
                    self.assertTrue(reg.may_block("g"))

    def test_unregistered_signal_defaults_to_the_safe_answer(self):
        # Same reasoning as `repair()` defaulting to advisory: an unknown signal
        # must not be able to reject a chapter.
        self.assertFalse(self.reg.may_block("never_registered"))
        self.assertEqual(self.reg.scope("never_registered"), "book")
        self.assertEqual(self.reg.proof("never_registered"), "")


class ListGatesFilterTest(unittest.TestCase):

    def setUp(self):
        self.reg = GateRegistry()
        for name, phase, scope, repair in (
            ("a", "review", "chapter", "L0"),
            ("b", "planning", "card", "L2"),
            ("c", "review", "book", "advisory"),
        ):
            self.reg.register(name, config_key=f"{name}_e", phase=phase,
                              scope=scope, repair=repair,
                              proof="m")(lambda *x, **k: {})

    def test_scope_filter(self):
        self.assertEqual(set(self.reg.list_gates(scope="card")), {"b"})
        self.assertEqual(set(self.reg.list_gates(scope="book")), {"c"})

    def test_filters_compose(self):
        self.assertEqual(set(self.reg.list_gates(phase="review", scope="chapter")), {"a"})
        self.assertEqual(self.reg.list_gates(phase="planning", scope="chapter"), {})

    def test_no_filter_returns_all(self):
        self.assertEqual(set(self.reg.list_gates()), {"a", "b", "c"})


class LiveRegistryTest(unittest.TestCase):
    """The real registry must satisfy the invariant, not just the class."""

    def test_every_gate_declares_a_proof(self):
        missing = [n for n in REGISTRY.list_gates() if not REGISTRY.proof(n)]
        self.assertEqual(missing, [], "every gate must cite its measured distribution")

    def test_every_proof_cites_evidence_or_says_it_has_none(self):
        # A proof that names neither a measurement nor its absence is decoration.
        # The honest forms are: a census/replay number, or an explicit
        # "never ran / UNVALIDATED" admission.
        #
        # `Recomputed` is the third: `tools/orphan_gates.py` recomputes a gate from
        # primary data (chapter texts + archived cards + metrics rows) rather than
        # replaying an archived verdict, which is the only way to measure a gate the
        # engine never stored a result for. Without it, a gate whose measured rate is
        # a bare count ("0 firings", "1/638") would read as unevidenced — the exact
        # opposite of the truth.
        TOKENS = ("census", "replay", "Recomputed", "measured", "UNVALIDATED", "%")
        for name in REGISTRY.list_gates():
            with self.subTest(gate=name):
                self.assertTrue(any(t in REGISTRY.proof(name) for t in TOKENS),
                                f"{name}'s proof cites no evidence: "
                                f"{REGISTRY.proof(name)!r}")

    def test_every_gate_declares_a_known_scope_and_layer(self):
        for name in REGISTRY.list_gates():
            with self.subTest(gate=name):
                self.assertIn(REGISTRY.scope(name), GATE_SCOPES)
                self.assertIn(REGISTRY.repair(name), REPAIR_LAYERS)

    def test_no_book_scope_gate_is_allowed_to_block(self):
        for name in REGISTRY.list_gates(scope="book"):
            with self.subTest(gate=name):
                self.assertFalse(REGISTRY.may_block(name))

    def test_book_wide_fossils_is_chapter_scope_because_of_in_current(self):
        # The gate the invariant was written for. Its ratio is book-cumulative,
        # but the hard verdict now ALSO requires the chapter under review to
        # contain the phrase — a conjunct this attempt can falsify. Remove
        # `in_current` and the honest scope becomes `book`, which would strip its
        # blocking power. Pinning this keeps the coupling visible.
        self.assertEqual(REGISTRY.scope("book_wide_fossils"), "chapter")
        self.assertTrue(REGISTRY.may_block("book_wide_fossils"))
        book = {ch: f"第{ch}章\n\n" + "声音压得很低" + chr(0x4E00 + ch) * 200
                for ch in range(1, 9)}
        book[9] = "第9章\n\n" + chr(0x4E00 + 9) * 200
        clean = quality.book_wide_fossils(book, {"novel": {}}, current_chapter=9)
        self.assertEqual(clean["hard_fossils"], [],
                         "a chapter without the phrase must be clearable")

    def test_planning_gates_are_card_scope(self):
        # A planning-phase gate measures the plan, so its subject is the card.
        # Blocking there is free: nothing has been written yet.
        for name in REGISTRY.list_gates(phase="planning"):
            with self.subTest(gate=name):
                self.assertEqual(REGISTRY.scope(name), "card")


if __name__ == "__main__":
    unittest.main()
