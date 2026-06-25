# Whole-Cell Visualization and Streaming Architecture

SobolevMacro should not try to become a desktop molecular viewer. PyMOL and
similar tools are excellent for inspecting atomic structures, but a whole-cell
microenvironment is a different scale problem: billions of beads, repeated
asset instances, coarse-grained far fields, and live coordinate updates from a
headless solver.

The production shape is a decoupled simulation/rendering stack. SobolevMacro
owns the Hamiltonian and graph updates; a renderer owns camera movement,
instancing, lighting, and video delivery.

## Data Stack

Whole-cell scenes combine four data layers:

1. Cryo-electron tomography supplies the macro-scale cellular landscape:
   membranes, organelle boundaries, ribosome clusters, and density-derived
   spatial constraints. EMDB stores reconstructed cryo-EM volumes and
   representative tomograms; EMPIAR stores the raw image data behind many
   3D maps and tomograms; the CryoET Data Portal adds standardized metadata
   and annotations for ML-ready tomograms.
2. Structural assets come from PDB, AlphaFold-class tools, Boltz/Protenix, and
   local graph/coarse-grain templates. One asset is stored per molecular species,
   not per copy. In practice these arrive as `.pdb`, `.cif`, `.mmcif`, or
   renderer-native USD assets.
3. Proteomics and transcriptomics provide copy numbers and relative abundance.
   These counts determine how many instances of each protein, RNA, lipid, glycan,
   or coarse bead enter the scene. PaxDB-style abundance tables and UniProt
   identifiers are matching layers, not geometry sources.
4. Packing tools such as cellPACK turn boundaries, copy numbers, and asset
   envelopes into a non-overlapping starting coordinate graph. SobolevMacro then
   relaxes that graph under contact, steric, bonded, slab, and graph-Sobolev
   constraints.

The synthesis workflow is:

```
[1. Macro mesh]       parse .mrc/.rec tomogram -> segment membranes and organelles
[2. Asset library]    fetch PDB/mmCIF or predicted structures -> assign asset ids
[3. Composition]      read abundance/copy-number tables -> expand instance counts
[4. Packing]          cellPACK or local packer -> seed transforms inside meshes
[5. Relaxation]       SobolevMacro JAX loop -> resolve collisions and docking
[6. Visualization]    binary frames or OpenUSD -> WebGPU / UE5 / Pixel Streaming
```

## Rendering Stack

The renderer should consume instance transforms, not unique atom meshes.

- WebGPU is the browser-first target for interactive scientific dashboards,
  point sprites, impostors, and coarse-grained views.
- OpenUSD is the interchange target for cinematic and digital-twin pipelines.
  USD can represent one asset plus many transforms instead of duplicating
  geometry.
- Unreal Engine 5 is the high-end realtime target for Nanite-scale instancing,
  cinematic lighting, and cloud-rendered navigation through crowded biology.

The repository-level adapter is `sobolev_visualization.py`. It emits compact
binary coordinate frames, `(N, 4, 4)` instance matrices, deterministic far-field
coarse-graining results, JSON scene manifests, and minimal OpenUSD ASCII
PointInstancer scenes. It does not import Unreal, USD, WebGPU, gRPC, or
WebSocket libraries; those belong in renderer-specific clients.

## Cloud Topology

Use separate compute and render roles when the scene gets large:

```
+-------------------------------------------------------------+
| HPC / Cloud Compute                                         |
| [Asset Library] -> [JAX Solver: SobolevMacro]               |
+-------------------------------------------------------------+
                              |
                              | binary X_next frames
                              | over SHM, gRPC, or WebSockets
                              v
+-------------------------------------------------------------+
| Render Client                                                |
| [WebGPU Browser] or [Unreal Engine 5 Pixel Streaming Node]   |
+-------------------------------------------------------------+
```

For the JAX solver, A100/H100-class accelerators are appropriate when the graph
is enormous and eigensolvers or sparse approximations dominate runtime. For
interactive rendering and browser delivery, use a graphics-capable GPU tier
with hardware video encoding.

## Pixel Streaming Path

Unreal Engine's Pixel Streaming path runs the UE application on a cloud or
workstation server and streams rendered frames and audio to browsers over
WebRTC. Browser input is sent back to the engine, so a local Mac only needs to
decode video and send camera controls.

A typical UE launch for a packaged visualizer is:

```bash
./SobolevVisualizer \
  -AudioMixer \
  -PixelStreamingIP=localhost \
  -PixelStreamingPort=8888 \
  -RenderOffscreen \
  -ForceRes \
  -ResX=1920 \
  -ResY=1080
```

The important split is GPU selection:

- AWS G5 instances use NVIDIA A10G GPUs and are designed for graphics workloads,
  with RTX drivers, CUDA, NVENC, DirectX, Vulkan, and OpenGL support.
- NVIDIA L4 GPUs are a modern rendering and video tier: ray tracing, DLSS-class
  graphics features, and low-latency encode density make them a strong
  Pixel Streaming target.
- NVIDIA A100 cards have no NVENC encoder in NVIDIA's own support matrix. Use
  them for SobolevMacro compute, not as the final Pixel Streaming render/encode
  tier.

## Coordinate Frame ABI

`encode_coordinate_frame()` writes one simulation frame as:

```
header: <8sHHIqd
  magic        = b"SMACRO1\0"
  version      = 1
  flags        = optional asset-id bit
  n_nodes      = number of coordinate rows
  frame_index  = signed 64-bit frame counter
  time_seconds = float64 simulation time

body:
  coords       = n_nodes * 3 little-endian float32 values
  asset_ids    = optional n_nodes little-endian uint32 values
```

The ABI is intentionally boring: a WebSocket server can forward it as one binary
message, a gRPC schema can store it in a `bytes` field, and a shared-memory ring
buffer can use the same payload without translation.

`instance_matrices()` converts coordinates into homogeneous transforms for
renderer instancing. `coarse_grain_far_field()` keeps the active-site sphere at
full resolution and collapses far-field nodes into centroid beads while
preserving a source-index mapping.

`asset_id_vector_from_counts()` turns abundance records into an instance asset
vector after a caller has mapped biological identifiers to asset ids.
`write_scene_manifest()` records the tomogram, segmentation mesh, coordinate
frame, source databases, and asset prototypes in a JSON handoff file.
`write_usda_point_instancer()` writes a dependency-free `.usda` PointInstancer
for renderer smoke tests and early Unreal/USD integration.

## Current Scope

This repository now owns the simulation-facing side of whole-cell visualization:

- graph and chain Sobolev physics in `sobolev_macromolecule.py`;
- guarded RNA candidate polish in `sobolev_polish_gate.py`;
- transport and renderer-adapter primitives in `sobolev_visualization.py`.

A full Unreal project, USD asset library, Pixel Streaming signaling service, or
WebGPU renderer should live in a separate frontend repository that depends on
these frame primitives.

## References

- Epic Games, Pixel Streaming in Unreal Engine:
  https://dev.epicgames.com/documentation/en-us/unreal-engine/pixel-streaming-in-unreal-engine
- EMDB:
  https://www.ebi.ac.uk/emdb/
- EMPIAR:
  https://www.ebi.ac.uk/empiar/
- CryoET Data Portal:
  https://cryoetdataportal.czscience.com/
- RCSB PDB:
  https://www.rcsb.org/
- PaxDB:
  https://pax-db.org/
- UniProt:
  https://www.uniprot.org/
- NVIDIA Video Encode and Decode GPU Support Matrix:
  https://developer.nvidia.com/video-encode-decode-support-matrix
- AWS EC2 G5 instance family:
  https://aws.amazon.com/ec2/instance-types/g5/
- NVIDIA L4 Tensor Core GPU:
  https://www.nvidia.com/en-us/data-center/l4/
