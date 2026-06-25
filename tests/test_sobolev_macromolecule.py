import importlib.util
import unittest

import numpy as np

import sobolev_macromolecule as sm


def straight_chain(n, step):
    x = np.arange(n, dtype=np.float64) * float(step)
    return np.column_stack([x, np.zeros(n), np.zeros(n)])


class SobolevMacromoleculeTests(unittest.TestCase):
    def test_factory_presets_capture_domain_constants(self):
        rna = sm.get_macromolecule_config("rna")
        protein = sm.get_macromolecule_config("protein")
        dsdna = sm.get_macromolecule_config("dna")

        self.assertEqual(rna.bead_name, "C1'")
        self.assertAlmostEqual(rna.ideal_bond_distance, 5.95)
        self.assertEqual(protein.bead_name, "C_alpha")
        self.assertAlmostEqual(protein.ideal_bond_distance, 3.80)
        self.assertAlmostEqual(protein.rg_exponent, 0.33)
        self.assertEqual(dsdna.rg_mode, "none")
        self.assertGreater(dsdna.bend_stiffness, 0.0)

    def test_expected_rg_respects_domain_modes(self):
        protein = sm.create_macromolecule("protein")
        dsdna = sm.create_macromolecule("dsdna")

        self.assertGreater(protein.expected_rg(200), 0.0)
        self.assertEqual(dsdna.expected_rg(2000), 0.0)

    def test_bond_energy_is_minimized_at_domain_bond_length(self):
        engine = sm.create_macromolecule("protein")
        coords = straight_chain(8, engine.config.ideal_bond_distance)
        terms = engine.energy_terms(coords)

        self.assertLess(terms["bond"], 1e-8)
        self.assertEqual(terms["contacts"], 0.0)

    def test_contact_term_penalizes_unsatisfied_restraint(self):
        engine = sm.create_macromolecule("rna")
        coords = straight_chain(5, engine.config.ideal_bond_distance)
        contact_map = np.zeros((5, 5), dtype=np.float64)
        contact_map[0, 4] = contact_map[4, 0] = 1.0

        terms = engine.energy_terms(coords, contact_map)
        self.assertGreater(terms["contacts"], 0.0)

    def test_dsdna_bending_penalizes_kinks_but_not_straight_chain(self):
        engine = sm.create_macromolecule("dsdna")
        straight = straight_chain(6, engine.config.ideal_bond_distance)
        kinked = straight.copy()
        kinked[3:, 1] = 10.0

        straight_bend = engine.energy_terms(straight)["bend"]
        kinked_bend = engine.energy_terms(kinked)["bend"]
        self.assertLess(straight_bend, 1e-8)
        self.assertGreater(kinked_bend, straight_bend)

    def test_watson_crick_contact_map_pairs_strands_antiparallel(self):
        contact_map = sm.watson_crick_contact_map(4)
        expected_pairs = {(0, 7), (1, 6), (2, 5), (3, 4)}
        got_pairs = set(zip(*np.where(np.triu(contact_map, k=1) > 0)))

        self.assertEqual(got_pairs, expected_pairs)

    def test_overrides_make_custom_factory_configs(self):
        engine = sm.create_macromolecule("protein", sigma_clash=4.5, default_steps=17)

        self.assertAlmostEqual(engine.config.sigma_clash, 4.5)
        self.assertEqual(engine.config.default_steps, 17)

    def test_polish_reports_missing_jax_cleanly_or_runs_tiny_smoke(self):
        engine = sm.create_macromolecule("rna", default_steps=1)
        coords = straight_chain(4, engine.config.ideal_bond_distance)

        if importlib.util.find_spec("jax") is None:
            with self.assertRaises(sm.JaxUnavailableError):
                engine.polish(coords, n_steps=1)
            return

        polished = engine.polish(coords, n_steps=1)
        self.assertEqual(polished.shape, coords.shape)
        self.assertTrue(np.isfinite(polished).all())


if __name__ == "__main__":
    unittest.main()
