#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIAMOND = "diamond"

PERCENT_IDENTITY_CUTOFF = 40.0
QUERY_COVERAGE_CUTOFF = 0.5
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


def parse_diamond_outfmt6(blast_outfile):
    """
    Parse DIAMOND tabular output and count homologous/conserved proteins.

    Output fields expected:
    qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen

    Criteria from Qin et al. (2014):
    - E-value < 1e-5
    - sequence identity > 40%
    - alignable region of the query protein sequence > 50%
    """
    homologous_gene_count = 0

    with open(blast_outfile, mode="r") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue

            sline = line.split("\t")
            if len(sline) < 13:
                raise ValueError(
                    f"Unexpected DIAMOND output format in {blast_outfile}. "
                    f"Expected 13 columns, got {len(sline)}."
                )

            percent_identity = float(sline[2])
            alignment_length = int(sline[3])
            query_start = int(sline[6])
            query_end = int(sline[7])
            evalue = float(sline[10])
            query_length = int(sline[12])

            # aligned span on query coordinates
            aligned_query_span = abs(query_end - query_start) + 1
            query_coverage = aligned_query_span / query_length

            # Apply strict interpretation of original paper wording
            if percent_identity <= PERCENT_IDENTITY_CUTOFF:
                continue

            if evalue >= EVALUE_CUTOFF:
                continue

            if query_coverage <= QUERY_COVERAGE_CUTOFF:
                continue

            homologous_gene_count += 1

    return homologous_gene_count


def make_diamond_db(faa_path, threads):
    command_list = [
        DIAMOND, "makedb",
        "--in", faa_path,
        "-d", faa_path,
        "--threads", str(threads),
        "--quiet"
    ]
    subprocess.run(command_list, check=True)


def run_diamond_blastp(query_faa, db_faa, output_file, threads):
    """
    Run DIAMOND BLASTP with output fields sufficient for POCP filtering.
    """
    command_list = [
        DIAMOND, "blastp",
        "--query", query_faa,
        "--db", db_faa,
        "--out", output_file,
        "--max-target-seqs", "1",
        "--threads", str(threads),
        "--quiet",
        "--outfmt", "6",
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen"
    ]
    subprocess.run(command_list, check=True)


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


def execute(faa_dir, output_file, threads=1, label_file=None):
    warnings.filterwarnings(action="ignore")

    if not os.path.isdir(faa_dir):
        raise FileNotFoundError(f"Input directory does not exist: {faa_dir}")

    label_dict = get_label_file_info(label_file) if label_file else {}

    faa_name_list = sorted(os.listdir(faa_dir))
    faa_name_list = [
        f for f in faa_name_list
        if f.endswith(".faa") or f.endswith(".fasta") or f.endswith(".fa")
    ]

    if len(faa_name_list) < 2:
        raise ValueError("At least two protein FASTA files are required.")

    label_name_list = []
    total_gene_count_list = []

    for faa_name in faa_name_list:
        faa_path = os.path.join(faa_dir, faa_name)
        make_diamond_db(faa_path, threads)
        total_gene_count_list.append(count_total_proteins(faa_path))

        inferred_label = infer_label_from_filename(faa_name)
        label_name = label_dict.get(inferred_label, inferred_label)
        label_name_list.append(label_name)

    np_pocp = np.eye(len(faa_name_list)) * 100.0
    pocp_list = list(np_pocp)

    for i in range(len(faa_name_list)):
        faa_name1 = os.path.join(faa_dir, faa_name_list[i])

        for j in range(i + 1, len(faa_name_list)):
            faa_name2 = os.path.join(faa_dir, faa_name_list[j])

            blast_outfile1 = f"diamond_{i}_against_{j}.txt"
            blast_outfile2 = f"diamond_{j}_against_{i}.txt"

            run_diamond_blastp(faa_name1, faa_name2, blast_outfile1, threads)
            homologous_gene_count1 = parse_diamond_outfmt6(blast_outfile1)

            run_diamond_blastp(faa_name2, faa_name1, blast_outfile2, threads)
            homologous_gene_count2 = parse_diamond_outfmt6(blast_outfile2)

            pocp = 100.0 * (
                float(homologous_gene_count1) + float(homologous_gene_count2)
            ) / (total_gene_count_list[i] + total_gene_count_list[j])
            pocp = round(pocp, 2)

            pocp_list[i][j] = pocp
            pocp_list[j][i] = pocp

            os.remove(blast_outfile1)
            os.remove(blast_outfile2)

        print(f"Finished row {i + 1}/{len(faa_name_list)}")

    for faa_name in faa_name_list:
        dmnd_path = os.path.join(faa_dir, faa_name + ".dmnd")
        if os.path.exists(dmnd_path):
            os.remove(dmnd_path)

    df_pocp = pd.DataFrame(pocp_list, index=label_name_list, columns=label_name_list)
    df_pocp.to_csv(output_file)

    plt.figure(figsize=(max(8, len(df_pocp) * 1.2), max(8, len(df_pocp) * 1.2)))
    plt.pcolor(df_pocp, vmin=0.0, vmax=100.0)

    for i in range(df_pocp.shape[1]):
        for j in range(df_pocp.shape[0]):
            plt.text(
                i + 0.5, j + 0.5,
                round(df_pocp.iloc[j, i], 2),
                ha="center", va="center", fontsize=8
            )

    plt.xticks(np.arange(0.5, len(df_pocp.columns), 1), df_pocp.columns, rotation=90)
    plt.yticks(np.arange(0.5, len(df_pocp.index), 1), df_pocp.index)
    plt.title("Pairwise POCP values heatmap", fontsize=16)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_file + "_heatmap.pdf", dpi=300)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pairwise Percentage of Conserved Proteins (POCP) "
            "among protein FASTA files using DIAMOND.\n\n"
            "Conserved proteins are defined following Qin et al. (2014):\n"
            "  - E-value < 1e-5\n"
            "  - sequence identity > 40%%\n"
            "  - aligned region on the query protein > 50%% of query length\n"
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
        help="Number of CPU threads for DIAMOND (default: 1)"
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    execute(
        faa_dir=args.faa_dir,
        output_file=args.output,
        threads=args.threads,
        label_file=args.label_file
    )


if __name__ == "__main__":
    main()