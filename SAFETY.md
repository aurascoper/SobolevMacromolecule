# Safety, Scope, and Responsible Use

SobolevMacromolecule is a non-infectious computational modeling,
coarse-grained relaxation, and visualization repository.

## In scope

- Structural visualization using public PDB/mmCIF or other public structural data.
- Coarse-grained geometry, packing, relaxation, and rendering experiments.
- Integrative modeling workflows that document provenance, assumptions, uncertainty,
  and validation limits.
- Educational and defensive scientific communication.

## Out of scope

This repository must not contain or facilitate:

- infectious-virus reconstruction;
- reverse-genetics or rescue workflows;
- synthesis-ready viral genomes, infectious clones, or plasmid maps;
- wet-lab propagation, transfection, culture, infection, or recovery protocols;
- actionable protocols for producing virus-like particles or infectious material;
- optimization steps intended to improve infectivity, assembly, tropism, immune escape,
  pathogenicity, environmental stability, or transmissibility.

## Pathogen-associated models

Any pathogen-associated visualization in this repository is a non-infectious
computational model. Such models are hypothesis-generating and may combine
experimental structures, homologous templates, predicted structures, coarse-grained
placements, and rendering transforms.

They should not be described as experimentally validated, coordinate-exact,
infectious, or complete unless supported by appropriate primary experimental
evidence and independent expert review.

## Filovirus / select-agent caution

Ebolaviruses are regulated select agents in some jurisdictions. This repository
does not host biological material, infectious nucleic acids, rescue systems, or
laboratory methods. Contributors should consult institutional biosafety,
biosecurity, legal, and export-control guidance before sharing pathogen-associated
datasets or workflows.

## Provenance requirement

Any public structural scene should include a provenance table listing:

- component name;
- source accession or model source;
- organism/species;
- experimental method or prediction method;
- resolution/confidence when available;
- chain IDs or entities used;
- transformations or placement rules;
- assumptions and known limitations.

See [PROVENANCE.md](PROVENANCE.md) for the BDBV virion scene provenance table
and [docs/pathogen-modeling-provenance-template.md](docs/pathogen-modeling-provenance-template.md)
for a blank template.

## Reporting concerns

Please open a private security advisory or contact the maintainer if you find
content that appears to enable unsafe biological reconstruction or misuse.