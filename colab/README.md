# `colab/` — running the heavy stages on a rented GPU

Two notebooks:

| Notebook | What it runs |
| --- | --- |
| **`isaac_g1_flat_colab.ipynb`** | Phase 4a — the full-collision G1 velocity policy in Isaac Lab, three variants, scored and rendered to video |
| **`lingbot_map_colab.ipynb`** | Phase 3 — LingBot-Map reconstruction |

Both follow the same rule: **shell out to unmodified repo scripts.** If a notebook forked the
pipeline, a difference between cloud and local would not mean anything.

---

# `isaac_g1_flat_colab.ipynb` — Phase 4a training

## Why it exists

The Isaac policy does not walk yet. Two full-size runs are in `notes/experiments.md` and
neither translates; the second, at upstream's own 4096 envs, learned to *turn on the spot*.
`dome_g1/mdp.py::feet_air_time_joystick` is the diagnosed fix and is **committed and
unvalidated**. This notebook validates it against a fallback lever and — the part that makes
the result readable — a **positive control** running upstream's own task definition.

The second reason is narrower and decisive: `play_g1_flat.py --video` **crashes the 8 GB dev
card at 687 ms** bringing the RTX renderer up. Training fits locally; video does not. The
video deliverable exists only here.

## Using it

1. **Runtime → Change runtime type → L4.** *Not* A100 — NVIDIA lists GPUs without RT cores as
   unsupported for Isaac Sim 5.1, and the renderer is exactly the part that needs them. Colab
   has been seen substituting L4 for a requested A100 anyway, so cell 1 checks what you got.
2. Run top to bottom. Cell 1 is a hard gate (driver ≥ 580.65.06, ≥ 40 GB free, RT cores); the
   install cell is ~25 GB and 20–40 minutes and is idempotent.
3. The three training cells are independent — run, skip or re-run any of them. Each retries
   with `--resume` on a disconnect, so a dropped session costs at most 25 iterations.

**Nothing is uploaded**: the notebook `git clone`s this repo at a pinned `GIT_REF`, so the
commit hash in every experiments row is real. Set `GH_TOKEN` in Colab Secrets if the repo is
private.

> **Honest status.** Nobody has published a working Isaac Sim **5.1** install on Colab; the one
> documented recipe (`j3soon/isaac-sim-colab`, which the Vulkan plumbing follows) targets 4.5.
> Treat the first run as unproven. The fallback is a rented L40S/A10, where
> `sims/isaac/setup_isaac_cloud.sh` runs unchanged — see `sims/isaac/README.md`.

## Bringing results back

`runs/isaac/` is symlinked onto Drive, so checkpoints survive a reclaimed VM. The last cells
embed the videos inline, print the `notes/experiments.md` rows and zip the results.

The rows need care: the trainer appends them inside the Colab checkout, and the driver
**harvests them and restores the file** after every attempt — otherwise run A's row would
leave the tree dirty and the clean-tree gate would refuse to start run C. They accumulate in
`MyDrive/GeologicDome/results/experiments_rows.md`; paste them into the real repo. **Rows are
never deleted**, including the ones recording a run that did not walk.

---

# `lingbot_map_colab.ipynb` — Phase 3 reconstruction

## Why it exists

The 8 GB RTX 4060 Ti cannot run upstream's own default reconstruction config. Measured Aug 6
(`notes/open-questions.md`): `demo.py` defaults to `kv_cache_sliding_window=64`, this box tops out
near **24**, and `--stride 2` does not buy a bigger cache — so the ceiling is cache size, not
sequence length. At the 16 we can afford, upstream's `example/courthouse` reconstructs with
`traj_length_over_extent` **24.9** against `example/loop`'s healthy 3.36.

That leaves one question the local box cannot answer: is the card the reason, or is something else
wrong? The notebook answers it by running the **same repo code** at **upstream's exact settings**
on a card that can hold the cache.

## What it runs

`recon/reconstruct.py`, `recon/calibrate_scale.py`, `recon/clean_cloud.py`, `recon/inspect_cloud.py`
— unmodified, shelled out to. That is the point: if the notebook forked the pipeline, a difference
in the result would not mean anything.

1. Both demo scenes (`example/loop`, `example/courthouse`) at upstream's config —
   `streaming`, `kvsw 64`, `keyframe_interval 1`, `num_scale_frames 8`, 518 px crop,
   `--mask_sky` on courthouse only, matching upstream's README invocations.
2. A `kv_cache_sliding_window` ladder — 16 → 128 — to locate the cliff. Includes the
   `kvsw 16 / kfi 2` vs `kvsw 32 / kfi 1` pair, which holds temporal span fixed while doubling
   the cached view count: the one comparison that separates "needs more views" from
   "needs a longer horizon".
3. Heavy Open3D cleanup to a metric, ground-aligned `cloud_clean.ply` per scene.
4. A printed verdict (compute-bound / not-compute / partial) and pasteable
   `notes/experiments.md` rows.

## Using the LingBot notebook

1. Upload the notebook to Colab. **Runtime → Change runtime type → A100 or L4.** A T4 works but
   drops to fp16 (`reconstruct.py` picks bf16 only on sm_80+), so two variables change at once and
   the comparison is muddier.
2. Run cells top to bottom — **nothing to upload**. The `recon/` scripts are **baked into the
   notebook** as a base64 blob (cell 8) by `colab/embed_recon.py`, so the notebook can never run
   stale copies of the pipeline.
3. Total runtime on an A100 is roughly 20–30 minutes including the sweep; the 4.63 GB checkpoint
   download is the slowest single step.

## Keeping the blob fresh

After **any** change to `recon/*.py`, re-bake and verify before committing:

```powershell
python colab\embed_recon.py          # regenerates the RECON_BLOB cell
python colab\embed_recon.py --check  # exits 1 if the notebook is stale
```

(`cloud_to_hfield.py` is no longer part of the blob — terrain conversion is per-sim and runs
locally, not on the rented GPU.)

## Bringing results back

The last cell zips `cloud_clean.ply` + `run.json` + `scale.json` + `clean_stats.json` + renders +
logs per run and downloads it. Unzip into `runs/recon/` and the legacy MuJoCo chain
(`sims/mujoco/terrain/cloud_to_hfield.py` → `sims/mujoco/terrain/drop_test.py` →
`sims/mujoco/scripts/settle_g1_recon.py`) runs locally against it unchanged. The Isaac terrain
path (cloud → OBJ → USD, `sims/isaac/terrain/`) will consume the same `cloud_clean.ply` once
built.

Every run still needs a row in `notes/experiments.md` — the notebook prints them formatted; fill in
the commit hash of the code that was baked into the blob.
