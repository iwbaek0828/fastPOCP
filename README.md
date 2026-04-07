# fastPOCP

A software for calculating the pairwise percentage of conserved proteins (POCP), an index for delineating the cutoff of genera.

Compared with the ordinary published POCP paper (Qin et al., 2014), this software uses DIAMOND to perform ultrafast pairwise POCP calculations for large genome groups. 

J Bacteriol. 2014 Jun;196(12):2210-5. doi: 10.1128/JB.01688-14. Epub 2014 Apr 4.
A proposed genus boundary for the prokaryotes based on genomic insights. PMID 24706738


The options for fastPOCP

-h, --help                   Show this help message and exit

-d, --faa_dir FAA_DIR        Directory containing protein fasta files (*.faa, *.fasta, *.fa)

-o, --output OUTPUT          Output CSV file for the pairwise POCP matrix

-t, --threads THREADS        Number of CPU threads for DIAMOND (default: 1)

-f, --label_file LABEL_FILE  Optional tab-delimited label file.
                             Format: inferred_id<tab>custom_label
                             Example: GCF_000005845<tab>Deinococcus radiodurans Ri
