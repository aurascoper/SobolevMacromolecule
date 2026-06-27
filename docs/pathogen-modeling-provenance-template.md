# Pathogen-Associated Model Provenance Template

Copy this template when publishing a structural scene that includes
pathogen-associated components. Fill in every row. If a field is unknown,
write "unknown" — do not leave it blank or guess.

See [SAFETY.md](../SAFETY.md) for the full safety and scope statement.

## Model Title

[One-line title. Use "non-infectious integrative structural visualization"
or "hypothesis-generating model" — not "reconstruction" or "complete model"
unless experimentally validated.]

## Scope Statement

[1-3 sentences. State that this is a non-infectious computational model.
State that it is hypothesis-generating. State what it is NOT (not a
validated reconstruction, not infectious material, not a genome).]

## Provenance Table

| Component | Source Accession | Organism / Species | Experimental or Prediction Method | Resolution / Confidence | Chain IDs / Entities Used | Copy Number | Placement Rule | Transformations Applied | Assumptions and Known Limitations |
|---|---|---|---|---|---|---|---|---|---|

## What Is Experimental

[List each component whose structure is experimentally determined.
Cite the accession, organism, method, and resolution. Note any
mismatches between the source organism and the target organism.]

## What Is Predicted / Synthetic

[List each component whose coordinates are computed (not from
experimental density). Describe the method (e.g., graph relaxation,
AlphaFold, coarse-grained packing).]

## What Is Speculative

[List each assumption that is not supported by direct experimental
evidence. Cross-species extrapolation, copy-number estimates, glycan
composition, morphology assumptions, etc.]

## Source Accessions

| Component | Accession | Database | URL | Species | Notes |
|---|---|---|---|---|---|

## Safety Check

- [ ] No nucleotide or amino acid sequences of any pathogen are stored
- [ ] No synthesis-ready construct designs or cloning vectors
- [ ] No reverse-genetics systems, minigenomes, replicons, or rescue protocols
- [ ] No wet-lab protocols of any kind
- [ ] No "one-click" pipeline that turns a sequence into a buildable pathogen
- [ ] All PDB IDs verified against RCSB annotations
- [ ] All cross-species templates labeled as homologous, not species-matched
- [ ] Provenance table complete with organism, method, resolution, and limitations

## Deposition Target (if applicable)

[If the model will be deposited, the appropriate target is PDB-IHM
(integrative/hybrid models), not ordinary PDB deposition.]