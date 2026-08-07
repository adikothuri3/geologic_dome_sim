# `sims/isaac/terrain/`

Phase 4b lands here: terrain converters for Isaac —
`cloud_clean.ply` (from `recon/`) → Open3D mesh → OBJ → USD via Isaac Lab's `MeshConverterCfg`
(`collision_approximation="triangleMesh"`; PLY is not accepted directly), plus procedural
mountain-terrain configs (`TerrainGeneratorCfg`) as the fallback when LingBot-Map output
isn't usable. Empty on purpose until that work starts.
