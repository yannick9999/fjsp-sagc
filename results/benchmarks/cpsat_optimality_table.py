"""Table: share of instances solved to optimality by CP-SAT, per instance size."""

from pathlib import Path

import pandas as pd

CPSAT_DIR = Path(__file__).parent / "CPSAT"

# (file stem, label) in display order
SIZES = [
    ("1005", "10x5"),
    ("1010", "10x10"),
    ("1510", "15x10"),
    ("2005", "20x5"),
    ("2010", "20x10"),
    ("2020", "20x20"),
    ("2030", "20x30"),
    ("3010", "30x10"),
    ("4010", "40x10"),
    ("5010", "50x10"),
    ("10010", "100x10"),
    ("20010", "200x10"),
    ("brandimarte", "Brandimarte"),
    ("edata", "edata"),
    ("rdata", "rdata"),
    ("vdata", "vdata"),
]


def main():
    row = {}
    for stem, label in SIZES:
        df = pd.read_csv(CPSAT_DIR / f"{stem}.csv")
        row[label] = 100 * (df["status"] == "OPTIMAL").mean()

    table = pd.DataFrame([row], index=["CP-SAT optimal (%)"])
    table.columns.name = "Size"

    print(table.to_string(float_format="%.1f"))
    print()
    print(table.to_latex(float_format="%.1f"))

    table.to_csv(Path(__file__).parent / "cpsat_optimality_table.csv", float_format="%.1f")


if __name__ == "__main__":
    main()
