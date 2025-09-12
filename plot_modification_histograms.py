#!/usr/bin/env python3
"""
plot_modification_histograms.py

Reads a CSV of triplet modification counts and plots histograms
of raw and precomputed structure‑normalized modification, insertion, and deletion counts.
Also prints consistency checks for insertions+deletions vs. modifications.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of raw and pre‑normalized modification counts from CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (can have commented header lines).",
    )
    parser.add_argument(
        "--output_png",
        default="modification_histograms.png",
        help="Filename for the output PNG (default: %(default)s).",
    )
    args = parser.parse_args()

    # Read CSV, skipping metadata/commented header lines
    df = pd.read_csv(args.input_csv, comment="#")

    # Raw‑count columns
    mod_cols = [
        "total_modifications",
        "stem_modifications",
        "hloop_modifications",
        "iloop_modifications",
        "bulge_modifications",
        "mloop_modifications",
    ]
    ins_cols = [
        "total_insertions",
        "stem_insertions",
        "hloop_insertions",
        "iloop_insertions",
        "bulge_insertions",
        "mloop_insertions",
    ]
    del_cols = [
        "total_deletions",
        "stem_deletions",
        "hloop_deletions",
        "iloop_deletions",
        "bulge_deletions",
        "mloop_deletions",
    ]

    # Pre‑normalized ("f_") columns provided in the CSV
    norm_mod_cols = [
        "f_total_modifications",
        "f_stem_modifications",
        "f_hloop_modifications",
        "f_iloop_modifications",
        "f_bulge_modifications",
        "f_mloop_modifications",
    ]
    norm_ins_cols = [
        "f_total_insertions",
        "f_stem_insertions",
        "f_hloop_insertions",
        "f_iloop_insertions",
        "f_bulge_insertions",
        "f_mloop_insertions",
    ]
    norm_del_cols = [
        "f_total_deletions",
        "f_stem_deletions",
        "f_hloop_deletions",
        "f_iloop_deletions",
        "f_bulge_deletions",
        "f_mloop_deletions",
    ]

    # Length columns (singular form in this CSV)
    struct_len_cols = [
        "total_len_stem",
        "total_len_hloop",
        "total_len_iloop",
        "total_len_bulge",
        "total_len_mloop",
    ]

    # Sanity check: all expected columns present
    missing = (
        [c for c in mod_cols if c not in df.columns],
        [c for c in ins_cols if c not in df.columns],
        [c for c in del_cols if c not in df.columns],
        [c for c in norm_mod_cols if c not in df.columns],
        [c for c in norm_ins_cols if c not in df.columns],
        [c for c in norm_del_cols if c not in df.columns],
        [c for c in struct_len_cols if c not in df.columns],
    )
    if any(missing):
        msgs = []
        if missing[0]:
            msgs.append(f"modifications: {missing[0]}")
        if missing[1]:
            msgs.append(f"insertions:    {missing[1]}")
        if missing[2]:
            msgs.append(f"deletions:     {missing[2]}")
        if missing[3]:
            msgs.append(f"f_mods:        {missing[3]}")
        if missing[4]:
            msgs.append(f"f_ins:         {missing[4]}")
        if missing[5]:
            msgs.append(f"f_dels:        {missing[5]}")
        if missing[6]:
            msgs.append(f"lengths:       {missing[6]}")
        raise ValueError("Missing expected columns – " + "; ".join(msgs))

    # Consistency checks
    total_mismatch = (df["total_insertions"] + df["total_deletions"] != df["total_modifications"])
    if total_mismatch.any():
        bad = df.index[total_mismatch].tolist()
        print(f"ERROR: Rows {bad} fail total_insertions + total_deletions == total_modifications")
    else:
        print("PASS: total_insertions + total_deletions == total_modifications for all rows")

    for struct in ["stem", "hloop", "iloop", "bulge", "mloop"]:
        ins_col = f"{struct}_insertions"
        del_col = f"{struct}_deletions"
        mod_col = f"{struct}_modifications"
        mismatch = (df[ins_col] + df[del_col] != df[mod_col])
        if mismatch.any():
            bad = df.index[mismatch].tolist()
            print(f"ERROR: Rows {bad} fail {ins_col} + {del_col} == {mod_col}")
        else:
            print(f"PASS: {ins_col} + {del_col} == {mod_col} for all rows")

    # Plotting
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12))
    axes = axes.flatten()

    # Row 0: raw modification counts
    for i, col in enumerate(mod_cols):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, edgecolor="black")
        ax.set_title(col.replace("_", " ").capitalize())
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")

    # Row 1: pre‑normalized modifications
    for i, col in enumerate(norm_mod_cols):
        ax = axes[6 + i]
        ax.hist(df[col].dropna(), bins=30, edgecolor="black")
        title = col.replace("f_", "").replace("_", " ").capitalize()
        ax.set_title(f"{title} (normalized)")
        ax.set_xlabel("Count / structure length")
        ax.set_ylabel("Frequency")

    # Row 2: pre‑normalized insertions (green)
    for i, col in enumerate(norm_ins_cols):
        ax = axes[12 + i]
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", color="green")
        title = col.replace("f_", "").replace("_", " ").capitalize()
        ax.set_title(f"{title} (normalized)")
        ax.set_xlabel("Insertions / structure length")
        ax.set_ylabel("Frequency")

    # Row 3: pre‑normalized deletions (red)
    for i, col in enumerate(norm_del_cols):
        ax = axes[18 + i]
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", color="red")
        title = col.replace("f_", "").replace("_", " ").capitalize()
        ax.set_title(f"{title} (normalized)")
        ax.set_xlabel("Deletions / structure length")
        ax.set_ylabel("Frequency")

    # Match y‑axis within each insertion/deletion pair
    for i in range(6):
        ai = axes[12 + i]
        ad = axes[18 + i]
        top = max(ai.get_ylim()[1], ad.get_ylim()[1])
        ai.set_ylim(0, top)
        ad.set_ylim(0, top)

    # Match x‑axis across the three normalized rows (mods, ins, del)
    for i in range(6):
        grp = [axes[6 + i], axes[12 + i], axes[18 + i]]
        xmin = min(ax.get_xlim()[0] for ax in grp)
        xmax = max(ax.get_xlim()[1] for ax in grp)
        for ax in grp:
            ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved histogram PNG to {args.output_png}")

if __name__ == "__main__":
    main()

