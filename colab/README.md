# `colab/` — running LingBot-Map on a rented GPU

One notebook: **`lingbot_map_colab.ipynb`**.

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

## Using it

1. Upload the notebook to Colab. **Runtime → Change runtime type → A100 or L4.** A T4 works but
   drops to fp16 (`reconstruct.py` picks bf16 only on sm_80+), so two variables change at once and
   the comparison is muddier.
2. Run cells top to bottom. Cell 4 needs `recon/` — easiest route is a zip:

   ```powershell
   Compress-Archive -Path recon -DestinationPath recon.zip -Force
   ```

   It also accepts a mounted Drive copy, or a private clone if you store a GitHub PAT in Colab
   Secrets as `GD_TOKEN`.
3. Everything else is automatic. Total runtime on an A100 is roughly 20–30 minutes including the
   sweep; the 4.63 GB checkpoint download is the slowest single step.

## Bringing results back

The last cell zips `cloud_clean.ply` + `run.json` + `scale.json` + `clean_stats.json` + renders +
logs per run and downloads it. Unzip into `runs/recon/` and the rest of the chain
(`recon/cloud_to_hfield.py` → `terrain/drop_test.py` → `scripts/settle_g1_recon.py`) runs locally
against it unchanged.

Every run still needs a row in `notes/experiments.md` — the notebook prints them formatted; fill in
the commit hash of the `recon/` you uploaded.
