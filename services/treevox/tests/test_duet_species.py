"""Unit tests for treevox.duet_species — the SPCD remap.

The remap exists to stop two silent drops (see the module docstring), so these
tests assert on the properties that make it safe rather than on a snapshot of
the table: every bucket is single-class, no bucket mixes signatures, and the
codes each tool would drop are reported rather than mapped.

The measured proof that the remap is inert against the real binary lives in the
integration test; these run without DUET installed.
"""

from __future__ import annotations

from treevox import duet_species


class TestRemapTable:
    def test_covers_the_species_both_tools_accept(self):
        assert len(duet_species.SPCD_TO_DUET) == 274

    def test_bounds_litter_layers(self):
        # Memory scales with the number of distinct species handed to DUET, so
        # the point of the remap is this ceiling holding for any domain.
        assert len(set(duet_species.SPCD_TO_DUET.values())) == 12

    def test_representative_is_itself_mappable(self):
        # A representative that DUET or duet-tools would drop would defeat the
        # whole exercise.
        for representative in set(duet_species.SPCD_TO_DUET.values()):
            assert representative in duet_species.SPCD_TO_DUET

    def test_remap_is_idempotent(self):
        for spcd, representative in duet_species.SPCD_TO_DUET.items():
            assert duet_species.SPCD_TO_DUET[representative] == representative, (
                f"remapping {spcd} -> {representative} is not a fixed point"
            )

    def test_bucket_members_share_a_duet_signature(self):
        signatures = duet_species._load_duet_species()
        for spcd, representative in duet_species.SPCD_TO_DUET.items():
            assert signatures[spcd] == signatures[representative], (
                f"SPCD {spcd} -> {representative} changes DUET's litter parameters"
            )

    def test_bucket_members_share_a_duet_tools_class(self):
        # The bug this guards: DUET's `wo` signature holds junipers (coniferous)
        # and oaks (deciduous), so a signature-only collapse would file oak
        # litter under conifer.
        classes = duet_species._load_duet_tools_classes()
        for spcd, representative in duet_species.SPCD_TO_DUET.items():
            assert classes[spcd] == classes[representative], (
                f"SPCD {spcd} -> {representative} changes the coniferous/"
                f"deciduous classification"
            )


class TestWoodlandGroupStaysSplit:
    """The concrete case the (signature, class) key exists for.

    Rocky Mountain juniper and Gambel oak share DUET's `wo` litter signature but
    not duet-tools' class. A juniper-oak woodland is a common interior-West fuel
    type, so this is a real stand, not a constructed one.
    """

    JUNIPER = 66
    GAMBEL_OAK = 814

    def test_both_are_mappable(self):
        assert not duet_species.unmappable({self.JUNIPER, self.GAMBEL_OAK})

    def test_they_do_not_collapse_together(self):
        mapping = duet_species.remap({self.JUNIPER, self.GAMBEL_OAK})
        assert mapping[self.JUNIPER] != mapping[self.GAMBEL_OAK]

    def test_each_keeps_its_class(self):
        classes = duet_species._load_duet_tools_classes()
        mapping = duet_species.remap({self.JUNIPER, self.GAMBEL_OAK})
        assert classes[mapping[self.JUNIPER]] == "coniferous"
        assert classes[mapping[self.GAMBEL_OAK]] == "deciduous"


class TestUnmappable:
    def test_reports_codes_duet_does_not_know(self):
        # Great Basin bristlecone pine: a real FIA species absent from DUET's
        # table. Measured: DUET drops it and exits 0.
        assert duet_species.unmappable({142}) == {142}

    def test_reports_codes_duet_tools_cannot_classify(self):
        # In DUET's table but absent from duet-tools' REF_SPECIES.csv, so its
        # litter is dropped at import. 0 and 1000 are the generic
        # softwood/hardwood codes an imputation step emits.
        assert duet_species.unmappable({0, 1000}) == {0, 1000}
        assert duet_species.unmappable({476, 477, 478, 479, 850}) == {
            476,
            477,
            478,
            479,
            850,
        }

    def test_passes_known_species(self):
        # Ponderosa pine, Douglas-fir, northern red oak.
        assert duet_species.unmappable({122, 202, 833}) == set()

    def test_remap_skips_unmappable(self):
        assert duet_species.remap({122, 142}) == {122: duet_species.SPCD_TO_DUET[122]}


class TestRemapOnRealStands:
    def test_california_oak_woodland_collapses(self):
        # The real inventory profiled during design: 11 species, 4 layers.
        stand = {801, 807, 805, 839, 981, 818, 333, 127, 312, 730, 122}
        mapping = duet_species.remap(stand)
        assert len(mapping) == len(stand)
        assert len(set(mapping.values())) == 4

    def test_rocky_mountain_conifer_stand(self):
        stand = {122, 202, 108, 746, 66, 73, 17, 374, 19, 93, 113, 64}
        mapping = duet_species.remap(stand)
        assert len(mapping) == len(stand)
        assert len(set(mapping.values())) == 8
