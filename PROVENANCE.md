# Provenance — BDBV Virion Scene Components

## Scope Statement

This is a **non-infectious, integrative, multi-scale structural visualization
of a Bundibugyo ebolavirus–associated virion architecture**, assembled from
public structural components, homologous templates, and computationally relaxed
placement/packing coordinates.

This is a **hypothesis-generating model**, not an experimentally validated
complete virion reconstruction and not a genome or infectious-virus
reconstruction.

No peer-reviewed cryo-EM structures of a 2026 Ituri/Kampala BDBV outbreak
isolate have been deposited in public databases at time of writing. The
structural assets are synthetic coarse-grained proxies whose physical
parameters (radii, masses, bond lengths, assembly densities) are drawn from
published filovirus biophysics, not from a deposited cryo-EM/tomography
reconstruction of an intact BDBV virion.

## PDB Accession Audit

The following PDB IDs were initially assigned and have been corrected after
review against RCSB annotations:

| PDB ID | Initial label | Corrected annotation (per RCSB) | Status |
|---|---|---|---|
| 6N7J | "Ebolavirus Glycoprotein trimer" | BDBV223 Fab bound to synthetic peptide of Bundibugyo virus Glycoprotein Stalk; 16-residue GP entity + human antibody chains; X-ray diffraction at 3.68 Å | **Corrected**: this is a stalk peptide + antibody complex, NOT a full GP trimer. Cannot be used as a full GP spike structural template. |
| 4LDB | "Ebolavirus VP40 matrix protein" | Zaire ebolavirus VP40 dimer; X-ray at 2.6 Å | **Corrected**: this is Zaire ebolavirus (EBOV), not Bundibugyo (BDBV). Must be labeled as a homologous template, not a BDBV structure. |
| 7ZPE | "Nucleoprotein-RNA complex" | Human branched-chain keto acid dehydrogenase kinase in complex with ligand; organism Homo sapiens | **Corrected**: this is a human metabolic enzyme, completely unrelated to ebolavirus. This was a wrong PDB ID — likely a typo or placeholder. Removed from the scene. |

**Action taken**: 7ZPE has been removed from `build_chimerax_virion.py` and
the provenance table. 6N7J has been relabeled as a stalk-peptide + antibody
complex, not a full GP trimer. 4LDB has been relabeled as a Zaire ebolavirus
homolog, not a BDBV structure. The NP-RNA component currently has no valid
PDB template — it is represented as a coarse-grained bead only.

## Provenance Table

| Component | Source Structure / Model | Organism | Experimental Method | Resolution / Confidence | Chain IDs / Entities | Copy Number (per 50 nm segment) | Placement Rule | Uncertainty / Speculative Assumptions |
|---|---|---|---|---|---|---|---|---|
| GP stalk peptide + BDBV223 Fab | PDB 6N7J | Bundibugyo virus (GP stalk peptide) + Homo sapiens (antibody) | X-ray diffraction | 3.68 Å | 16-residue GP stalk peptide, BDBV223 Fab heavy chain, Fab light chain | 225 (coarse-grained positions) | Packed on outer envelope at ~10 nm spacing; radial position from Sobolev graph relaxation | 6N7J is a 16-residue stalk peptide bound to antibody, NOT a full GP trimer. It cannot serve as a full GP spike structural template. The 225 coarse-grained positions represent GP spike placements, but the atomic detail at each position is NOT from 6N7J. A full GP trimer structure (e.g., EBOV GP from PDB 3CSY or 5JQ3) would be needed for atomic-resolution spike rendering. |
| VP40 dimer (matrix) | PDB 4LDB | Zaire ebolavirus (EBOV), NOT Bundibugyo (BDBV) | X-ray diffraction | 2.6 Å | VP40 dimer | 370 | Tiled on inner leaflet at ~6 nm circumferential and axial spacing; radial position at envelope_radius - bead_radius | 4LDB is a Zaire ebolavirus VP40, not BDBV VP40. VP40 sequence identity between EBOV and BDBV is ~60%. This is a homologous template, not a species-matched structure. Oligomerization state (dimer vs. octamer vs. hexamer) may vary along the virion. |
| NP-RNA bead (nucleocapsid) | No valid PDB template identified | N/A | N/A (coarse-grained proxy only) | N/A | N/A | 100 | Helical core along central axis; ~7 nm spacing; one NP bead per ~6 nt of genome | The originally assigned PDB 7ZPE was a human metabolic enzyme, not a nucleoprotein. No species-matched NP-RNA structure has been identified. The NP-RNA component is represented as a coarse-grained bead with radius 4.0 nm from published filovirus biophysics, not from an experimental structure. Genome length (18,940 nt) is from NC_014373.1 metadata — no sequence is stored. |
| Glycan branch (surface) | No PDB template | N/A | N/A (speculative) | N/A | N/A | 900 | Attached to GP trimer outermost bead; random branch direction | Glycan composition and branching are speculative. Real BDBV GP glycosylation patterns are not fully characterized. Currently skipped in the ChimeraX visualization. |
| VP40 octamer | PDB 4LDB (same, oligomeric form assumed) | Zaire ebolavirus (homolog) | X-ray (monomer) | 2.6 Å | VP40 octamer (assumed) | 0 (in scene manifest, not in current USDA) | Not placed — listed in asset library but copy number is zero in the relaxed segment | Octameric VP40 ring structures are known from in vitro studies but their prevalence in intact virions is uncertain. |

## What Is Experimental

- **6N7J**: X-ray structure of BDBV223 Fab bound to a 16-residue synthetic
  peptide from the Bundibugyo virus GP stalk. This is experimentally
  determined and species-matched, but it covers only a small stalk epitope,
  not the full GP trimer.
- **4LDB**: X-ray structure of Zaire ebolavirus VP40 dimer. Experimentally
  determined but NOT species-matched (EBOV, not BDBV).
- **NC_014373.1**: genome length (18,940 nt) from NCBI metadata. No
  nucleotide sequence is stored in this repository — only the integer
  length and the accession number as a metadata pointer.

## What Is Predicted / Synthetic

- **Spatial coordinates**: all instance positions are computed by the
  SobolevMacro graph relaxation engine, not derived from experimental
  density maps. The relaxation uses steric, bond, contact, and
  cylindrical envelope potentials with parameters from published
  filovirus biophysics.
- **Copy numbers**: derived from spacing assumptions (GP ~10 nm, VP40
  ~6 nm, NP ~7 nm) and the 50 nm segment length, not from stoichiometric
  measurements of a real BDBV virion.
- **Glycan geometry**: entirely speculative. No structural template.
- **NP-RNA component**: coarse-grained bead only. No experimental
  structure has been identified for this component.

## What Is Speculative

- **Filamentous morphology assumption**: the packing model assumes a
  uniform filamentous cylinder. Real filoviruses show variable
  morphology (straight, bent, "shepherd's crook", branched) that is not
  captured in the 50 nm segment.
- **Cross-species extrapolation**: 4LDB is from EBOV (Zaire ebolavirus),
  not BDBV (Bundibugyo ebolavirus). VP40 sequence identity between
  species is ~60%. Structural differences in matrix assembly are expected.
- **GP spike atomic detail**: 6N7J provides only a 16-residue stalk
  peptide, not a full GP trimer. The scene's GP spike positions are
  coarse-grained; atomic-detail rendering at each position requires a
  full GP trimer template (e.g., EBOV GP from PDB 3CSY or 5JQ3 as a
  further homologous template).
- **Nucleocapsid helix parameters**: the helix radius (6 nm) and pitch
  are assumed from published filovirus biophysics, not from a BDBV-
  specific cryo-ET reconstruction. No NP-RNA structure is available.
- **Assembly completeness**: the scene includes GP, VP40, NP, and
  glycan. It does not include VP24, VP30, VP35, L polymerase, or host
  membrane components (lipids, host-derived proteins).

## Transformations Applied

1. **Sobolev graph relaxation**: the SobolevMacro engine relaxes the
   packed configuration under bond, steric, contact, and cylindrical
   envelope potentials using a graph Laplacian spectral filter for
   gradient preconditioning (JAX, x64 precision).
2. **Instance transform export**: relaxed coordinates are converted to
   4×4 homogeneous transform matrices and exported as OpenUSD ASCII
   (`.usda`) via `sobolev_visualization.write_usda_instance_transforms()`.
3. **Unit conversion**: USDA coordinates are in nanometers. The ChimeraX
   visualization script (`build_chimerax_virion.py`) converts to Angstroms
   (×10) for PDB/ChimeraX compatibility.
4. **PDB model placement**: PDB structures (6N7J, 4LDB) are fetched live
   from RCSB and duplicated at the computed positions via ChimeraX's
   `combine` + `move` commands. Only translation is applied (the current
   USDA matrices have identity rotation).
5. **UE5 spline instancing**: the UE5 script constructs a shepherd's-crook
   spline and instances the 50 nm segment along it to build a full-length
   (~1081 nm) virion. Each instance is aligned to the spline tangent.

## Source Accessions

| Component | Accession | Database | URL | Species | Notes |
|---|---|---|---|---|---|
| GP stalk peptide + Fab | 6N7J | RCSB PDB | https://www.rcsb.org/structure/6N7J | Bundibugyo virus (peptide) + Homo sapiens (antibody) | 16-residue GP stalk peptide, NOT full GP trimer |
| VP40 dimer | 4LDB | RCSB PDB | https://www.rcsb.org/structure/4LDB | Zaire ebolavirus (EBOV), NOT BDBV | Homologous template; ~60% identity to BDBV VP40 |
| NP-RNA complex | None identified | N/A | N/A | N/A | 7ZPE was incorrectly assigned; it is a human enzyme. Coarse-grained bead only. |
| BDBV genome metadata | NC_014373.1 | NCBI GenBank | https://www.ncbi.nlm.nih.gov/nuccore/NC_014373.1 | Orthoebolavirus bundibugyoense | Metadata pointer only (organism, genome length). No sequence stored. |
| BDBV 2007 outbreak | FJ217161.1 | NCBI GenBank | https://www.ncbi.nlm.nih.gov/nuccore/FJ217161.1 | Orthoebolavirus bundibugyoense | Metadata pointer only. No sequence stored. |

## Deposition Target

If this model is ever deposited, the appropriate target is **PDB-IHM**
(integrative/hybrid models), not ordinary PDB deposition and not a claim
of a solved structure. PDB-IHM is specifically for integrative/hybrid
models that combine experimental and computational information, and it
is now unified with the PDB archive for integrative structures.