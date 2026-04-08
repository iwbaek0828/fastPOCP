#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

DIAMOND = "diamond"

PERCENT_IDENTITY_CUTOFF = 40.0
QUERY_COVERAGE_CUTOFF = 50.0   # DIAMOND --query-cover expects percentage
EVALUE_CUTOFF = 1e-5


def get_label_file_info(label_file):
    """
    Read label mapping file.
    Expected format (tab-separated):
        file_prefix<TAB>label
    """
    label_dict = {}
    with open(label_file, mode="r") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            sline = line.split("\t")
            if len(sline) < 2:
                continue
            label_dict[sline[0]] = sline[1]
    return label_dict


def count_total_proteins(faa_path):
    total_gene_count = 0
    with open(faa_path, mode="r") as reader:
        for line in reader:
            if line.startswith(">"):
                total_gene_count += 1
    return float(total_gene_count)


def infer_label_from_filename(faa_name):
    """
    Default label inference from filename.
    Example:
        GCF_00012345.faa -> GCF_00012345
        abc_def_xxx.faa  -> abc_def
    """
    stem = os.path.splitext(faa_name)[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return stem


def build_mode_args(mode):
    mode = mode.lower()
    if mode == "default":
        return []
    allowed = {
        "faster",
        "fast",
        "mid-sensitive",
        "sensitive",
        "more-sensitive",
        "very-sensitive",
        "ultra-sensitive",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported DIAMOND mode: {mode}")
    return [f"--{mode}"]



def make_diamond_db_if_needed(faa_path, threads):
    dmnd_path = str(faa_path) + ".dmnd"
    if os.path.exists(dmnd_path):
        return
    command_list = [
        DIAMOND, "makedb",
        "--in", str(faa_path),
        "-d", str(faa_path),
        "--threads", str(threads),
        "--quiet"
    ]
    subprocess.run(command_list, check=True)



def run_diamond_count(query_faa, db_faa, threads, mode="default", tmpdir=None,
                      block_size=None, index_chunks=None):
    """
    Run DIAMOND BLASTP and return the number of query proteins that pass
    POCP-style filters.

    Key speedups vs. the original code:
    - no temporary output files
    - filtering pushed into DIAMOND (--id, --query-cover, --evalue)
    - minimal output format (one line per accepted query hit)
    """
    command_list = [
        DIAMOND, "blastp",
        "--query", str(query_faa),
        "--db", str(db_faa),
        "--max-target-seqs", "1",
        "--threads", str(threads),
        "--quiet",
        "--evalue", str(EVALUE_CUTOFF),
        "--id", str(PERCENT_IDENTITY_CUTOFF),
        "--query-cover", str(QUERY_COVERAGE_CUTOFF),
        "--outfmt", "6", "qseqid",
    ]
    command_list.extend(build_mode_args(mode))

    if tmpdir:
        command_list.extend(["--tmpdir", str(tmpdir)])
    if block_size is not None:
        command_list.extend(["--block-size", str(block_size)])
    if index_chunks is not None:
        command_list.extend(["--index-chunks", str(index_chunks)])

    proc = subprocess.run(
        command_list,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = proc.stdout.strip()
    if not stdout:
        return 0
    return stdout.count("\n") + 1



def compute_pair(task):
    i, j, faa1, faa2, total1, total2, threads_per_job, mode, tmpdir, block_size, index_chunks = task

    homologous_gene_count1 = run_diamond_count(
        faa1, faa2, threads_per_job, mode=mode,
        tmpdir=tmpdir, block_size=block_size, index_chunks=index_chunks
    )
    homologous_gene_count2 = run_diamond_count(
        faa2, faa1, threads_per_job, mode=mode,
        tmpdir=tmpdir, block_size=block_size, index_chunks=index_chunks
    )

    pocp = 100.0 * (float(homologous_gene_count1) + float(homologous_gene_count2)) / (total1 + total2)
    pocp = round(pocp, 2)
    return i, j, pocp



def build_tasks(faa_paths, total_gene_count_list, threads_per_job, mode,
                tmpdir=None, block_size=None, index_chunks=None):
    tasks = []
    for i in range(len(faa_paths)):
        for j in range(i + 1, len(faa_paths)):
            tasks.append((
                i, j,
                str(faa_paths[i]),
                str(faa_paths[j]),
                total_gene_count_list[i],
                total_gene_count_list[j],
                threads_per_job,
                mode,
                tmpdir,
                block_size,
                index_chunks,
            ))
    return tasks

def execute(faa_dir, output_file, threads=1, label_file=None, mode="default",
            jobs=1, tmpdir=None, block_size=None, index_chunks=None, keep_db=True):
    warnings.filterwarnings(action="ignore")

    faa_dir = Path(faa_dir)
    if not faa_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {faa_dir}")

    label_dict = get_label_file_info(label_file) if label_file else {}

    faa_name_list = sorted([
        f.name for f in faa_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".faa", ".fasta", ".fa"}
    ])

    if len(faa_name_list) < 2:
        raise ValueError("At least two protein FASTA files are required.")

    faa_paths = [faa_dir / f for f in faa_name_list]

    label_name_list = []
    total_gene_count_list = []

    print(f"[1/4] Preparing {len(faa_paths)} DIAMOND databases and counting proteins...")
    for faa_path in faa_paths:
        make_diamond_db_if_needed(faa_path, threads=max(1, min(threads, jobs if jobs > 1 else threads)))
        total_gene_count_list.append(count_total_proteins(faa_path))

        inferred_label = infer_label_from_filename(faa_path.name)
        label_name = label_dict.get(inferred_label, inferred_label)
        label_name_list.append(label_name)

    np_pocp = np.eye(len(faa_paths)) * 100.0

    total_pairs = len(faa_paths) * (len(faa_paths) - 1) // 2
    threads_per_job = max(1, threads // max(1, jobs))
    if threads_per_job == 0:
        threads_per_job = 1

    print(f"[2/4] Calculating POCP for {total_pairs} genome pairs...")
    print(f"       DIAMOND mode: {mode}; jobs: {jobs}; threads per job: {threads_per_job}")

    tasks = build_tasks(
        faa_paths, total_gene_count_list, threads_per_job, mode,
        tmpdir=tmpdir, block_size=block_size, index_chunks=index_chunks
    )

    if jobs == 1:
        completed = 0
        for task in tasks:
            i, j, pocp = compute_pair(task)
            np_pocp[i, j] = pocp
            np_pocp[j, i] = pocp
            completed += 1
            if completed % 100 == 0 or completed == total_pairs:
                print(f"       Completed {completed}/{total_pairs} pairs")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            future_to_pair = {ex.submit(compute_pair, task): (task[0], task[1]) for task in tasks}
            for future in as_completed(future_to_pair):
                i, j, pocp = future.result()
                np_pocp[i, j] = pocp
                np_pocp[j, i] = pocp
                completed += 1
                if completed % 100 == 0 or completed == total_pairs:
                    print(f"       Completed {completed}/{total_pairs} pairs")

    print("[3/4] Writing POCP matrix...")
    df_pocp = pd.DataFrame(np_pocp, index=label_name_list, columns=label_name_list)
    df_pocp.to_csv(output_file)

    print("[4/4] Heatmap should be drawn in fastPOCP_plot.py.")

    if not keep_db:
        print("Cleaning up .dmnd files...")
        for faa_path in faa_paths:
            dmnd_path = str(faa_path) + ".dmnd"
            if os.path.exists(dmnd_path):
                os.remove(dmnd_path)



def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pairwise Percentage of Conserved Proteins (POCP) "
            "among protein FASTA files using DIAMOND.\n\n"
            "Speed-oriented improvements vs. the original script:\n"
            "  - Reuses existing .dmnd databases\n"
            "  - Pushes filters into DIAMOND\n"
            "  - Avoids temporary outfmt6 files\n"
            "  - Supports parallel pairwise jobs\n\n"
            "Conserved proteins are filtered as:\n"
            "  - E-value < 1e-5\n"
            "  - sequence identity > 40%\n"
            "  - query coverage > 50%\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-d", "--faa_dir",
        required=True,
        help="Directory containing protein FASTA files (*.faa, *.fasta, *.fa)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file for the pairwise POCP matrix"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Total CPU threads budget across all jobs (default: 1)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of genome-pair jobs to run in parallel (default: 1)"
    )
    parser.add_argument(
        "-m", "--mode",
        default="default",
        choices=[
            "default", "faster", "fast", "mid-sensitive",
            "sensitive", "more-sensitive", "very-sensitive", "ultra-sensitive"
        ],
        help="DIAMOND sensitivity mode (default: default)"
    )
    parser.add_argument(
        "-f", "--label_file",
        default=None,
        help=(
            "Optional tab-delimited label file.\n"
            "Format: inferred_id<TAB>custom_label\n"
            "Example:\n"
            "  GCF_000005845\tDeinococcus radiodurans R1"
        )
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="Optional DIAMOND temporary directory (recommend local SSD/NVMe)"
    )
    parser.add_argument(
        "--block-size",
        type=float,
        default=None,
        help="Optional DIAMOND --block-size value"
    )
    parser.add_argument(
        "--index-chunks",
        type=int,
        default=None,
        help="Optional DIAMOND --index-chunks value"
    )
    parser.add_argument(
        "--cleanup-db",
        action="store_true",
        help="Delete .dmnd files after completion (default: keep for reuse)"
    )

    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.jobs > args.threads:
        print(
            f"Warning: jobs ({args.jobs}) > threads ({args.threads}); "
            f"each DIAMOND job will fall back to 1 thread.",
            file=sys.stderr,
        )

    execute(
        faa_dir=args.faa_dir,
        output_file=args.output,
        threads=args.threads,
        label_file=args.label_file,
        mode=args.mode,
        jobs=args.jobs,
        tmpdir=args.tmpdir,
        block_size=args.block_size,
        index_chunks=args.index_chunks,
        keep_db=not args.cleanup_db,
    )


if __name__ == "__main__":
    main()
