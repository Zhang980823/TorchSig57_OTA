#!/usr/bin/env python3
"""Create a reproducible, balanced Sig57 HDF5 subset without modifying the source.

Dependencies: numpy, h5py. No TorchSig, PyTorch, CUDA, or USRP is required.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("X", "Y", "Z", "BW", "FC")
BATCH_SIZE = 256


def select_indices(source, samples_per_combo, seed):
    """Sample each declared pair without assuming source rows are grouped."""
    if samples_per_combo <= 0:
        raise ValueError("samples_per_combo must be positive")
    if set(source.keys()) != set(FIELDS):
        raise ValueError(f"Expected exactly these source datasets: {FIELDS}")
    total = len(source["X"])
    if total != int(source.attrs["total_samples"]):
        raise ValueError("total_samples does not match X")
    if source["X"].shape != (total, 2, int(source.attrs["num_iq_samples"])):
        raise ValueError("Expected X shape (samples, 2, num_iq_samples)")
    for name in FIELDS[1:]:
        if source[name].shape != (total,):
            raise ValueError(f"{name} must have one scalar per sample")

    labels = source["Y"][:]
    snr_values = source["Z"][:]
    classes = np.arange(len(source.attrs["modulation_types"]))
    snrs = np.sort(np.asarray(source.attrs["snr_range"]))
    if not len(classes) or not len(snrs) or len(np.unique(snrs)) != len(snrs):
        raise ValueError("Class/SNR metadata must be nonempty; SNRs must be unique")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer, zero-based modulation labels")
    if not np.array_equal(np.unique(labels), classes):
        raise ValueError("Observed labels do not match modulation_types")
    if not np.array_equal(np.unique(snr_values), snrs):
        raise ValueError("Observed SNRs do not match snr_range")

    group_ids = labels.astype(np.int64) * len(snrs) + np.searchsorted(snrs, snr_values)
    counts = np.bincount(group_ids, minlength=len(classes) * len(snrs))
    for group in np.flatnonzero(counts < samples_per_combo):
        label, snr_index = divmod(int(group), len(snrs))
        raise ValueError(
            f"Class {label}, SNR {snrs[snr_index]} has {counts[group]} samples; "
            f"need {samples_per_combo}"
        )
    # Stable grouping ensures a seed selects the same source rows on every run.
    order = np.argsort(group_ids, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = [
        rng.choice(order[offsets[g]:offsets[g + 1]], samples_per_combo, replace=False)
        for g in range(len(counts))
    ]
    indices = np.sort(np.concatenate(selected)).astype(np.int64)
    return indices, snrs, counts.reshape(len(classes), len(snrs))


def write_subset(source, path, indices, snrs, samples_per_combo, seed):
    """Copy selected rows losslessly, preserving field names, types and attrs."""
    with h5py.File(path, "x") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["total_samples"] = len(indices)
        output.attrs["samples_per_combo"] = samples_per_combo
        output.attrs["snr_range"] = np.asarray(snrs, dtype=source.attrs["snr_range"].dtype)
        output.attrs["source_file"] = Path(source.filename).name
        output.attrs["source_total_samples"] = int(source.attrs["total_samples"])
        output.attrs["source_samples_per_combo"] = int(source.attrs["samples_per_combo"])
        output.attrs["source_snr_range"] = source.attrs["snr_range"]
        output.attrs["subset_seed"] = seed
        output.attrs["subset_method"] = "PCG64; uniform without replacement per class/SNR"
        output.attrs["subset_order"] = "ascending source row index"
        provenance = output.create_dataset(
            "source_indices", data=indices, compression="gzip", shuffle=True
        )
        provenance.attrs["description"] = "Zero-based row indices in source_file"
        for name in FIELDS:
            original = source[name]
            copied = output.create_dataset(
                name,
                shape=(len(indices),) + original.shape[1:],
                dtype=original.dtype,
                chunks=(min(BATCH_SIZE, len(indices)),) + original.shape[1:],
                compression="gzip", compression_opts=4, shuffle=True,
            )
            for key, value in original.attrs.items():
                copied.attrs[key] = value
            for start in range(0, len(indices), BATCH_SIZE):
                rows = indices[start:start + BATCH_SIZE]
                copied[start:start + len(rows)] = original[rows]


def verify_subset(source, path, indices, snrs, samples_per_combo):
    """Check every copied value against the source, including all IQ samples."""
    with h5py.File(path, "r") as output:
        if not np.array_equal(output["source_indices"][:], indices):
            raise ValueError(f"Incorrect provenance in {path}")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate source rows selected")
        if int(output.attrs["total_samples"]) != len(indices):
            raise ValueError("Incorrect output total_samples")
        if int(output.attrs["samples_per_combo"]) != samples_per_combo:
            raise ValueError("Incorrect output samples_per_combo")
        if not np.array_equal(output.attrs["snr_range"], snrs):
            raise ValueError("Incorrect output snr_range")
        for key, value in source.attrs.items():
            if key not in {"total_samples", "samples_per_combo", "snr_range"}:
                if not np.array_equal(output.attrs[key], value):
                    raise ValueError(f"Changed source attribute {key}")
        y, z = output["Y"][:], output["Z"][:]
        class_count = len(source.attrs["modulation_types"])
        if len(indices) != class_count * len(snrs) * samples_per_combo:
            raise ValueError("Incorrect output sample count")
        for label in range(class_count):
            for snr in snrs:
                if np.count_nonzero((y == label) & (z == snr)) != samples_per_combo:
                    raise ValueError(f"Unbalanced output: class={label}, SNR={snr}")
        for name in FIELDS:
            if output[name].dtype != source[name].dtype:
                raise ValueError(f"Changed dtype for {name}")
            for start in range(0, len(indices), BATCH_SIZE):
                rows = indices[start:start + BATCH_SIZE]
                expected = source[name][rows]
                actual = output[name][start:start + len(rows)]
                if actual.tobytes() != expected.tobytes():
                    raise ValueError(f"Data mismatch: {path.name}, {name}, row {start}")


def file_record(path, directory, count, snrs):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(directory)),
        "samples": count,
        "snr_db": np.asarray(snrs).tolist(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original Sig57 HDF5 file (read-only)")
    parser.add_argument("output_dir", type=Path, help="New directory; must not exist")
    parser.add_argument("--samples-per-combo", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-snr", action="store_true", help="Also write one HDF5 per SNR")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    source_stat = args.source.stat()
    with h5py.File(args.source, "r") as source:
        indices, snrs, counts = select_indices(source, args.samples_per_combo, args.seed)
        print(f"Source: {counts.shape[0]} classes x {len(snrs)} SNRs, "
              f"{counts.min()}..{counts.max()} rows/group", flush=True)
        print(f"Selected {len(indices):,} unique rows; seed={args.seed}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        combined = args.output_dir / args.source.name
        print(f"Writing and verifying {combined.name} ...", flush=True)
        write_subset(source, combined, indices, snrs, args.samples_per_combo, args.seed)
        verify_subset(source, combined, indices, snrs, args.samples_per_combo)
        records = [file_record(combined, args.output_dir, len(indices), snrs)]
        # Partition the same selected rows by SNR, so the shards and merged
        # file contain exactly the same samples and original row indices.
        if args.split_by_snr:
            shard_dir = args.output_dir / "github_shards"
            shard_dir.mkdir()
            with h5py.File(combined, "r") as small:
                selected_snrs = small["Z"][:]
            for snr in snrs:
                shard_indices = indices[selected_snrs == snr]
                shard = shard_dir / f"sig57_snr_{snr:g}dB.h5"
                print(f"Writing and verifying {shard.name} ...", flush=True)
                write_subset(source, shard, shard_indices, [snr], args.samples_per_combo, args.seed)
                verify_subset(source, shard, shard_indices, [snr], args.samples_per_combo)
                records.append(file_record(shard, args.output_dir, len(shard_indices), [snr]))
        after = args.source.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed during processing; do not publish this subset")
        manifest = {
            "description": "Sig57-OTA synthetic example subset (not OTA measurements)",
            "source_file": args.source.name,
            "source_bytes": source_stat.st_size,
            "source_total_samples": int(source.attrs["total_samples"]),
            "source_samples_per_combo": int(source.attrs["samples_per_combo"]),
            "source_counts_per_class_snr": counts.tolist(),
            "samples_per_combo": args.samples_per_combo,
            "total_samples": len(indices),
            "num_classes": counts.shape[0],
            "snr_db": snrs.tolist(),
            "seed": args.seed,
            "rng": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "h5py_version": h5py.__version__,
            "verification": "All rows of X/Y/Z/BW/FC compared bit-for-bit with original; all groups balanced",
            "files": records,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Done: {len(indices):,} samples; merged file "
              f"{records[0]['bytes'] / 1024**2:.2f} MiB. All checks passed.", flush=True)


if __name__ == "__main__":
    main()
