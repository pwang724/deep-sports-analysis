"""Convert every source and print the row counts: python -m dsa.data.convert"""
import pandas as pd

from dsa.data.convert import NAMES, run

rows = {name: run(name) for name in NAMES}
print()
print(pd.DataFrame(rows).T.to_string())
