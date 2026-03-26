"""Pandas / PyRanges helpers."""

import pyranges as pr


def df_to_pyranges(
    df,
    start_col="start",
    end_col="end",
    chr_col="chr",
    start_slop=0,
    end_slop=0,
):
    df = df.copy()
    df["Chromosome"] = df[chr_col]
    df["Start"] = df[start_col] - start_slop
    df["End"] = df[end_col] + end_slop
    return pr.PyRanges(df)
