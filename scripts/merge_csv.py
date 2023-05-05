from pathlib import Path

import pandas as pd
from XrayTo3DShape import get_anatomy_from_path
csv_dir = 'results/challengeR/'
csv_files = Path(csv_dir).glob('./**/*.csv')


dataframes = []
for f in csv_files:
    anatomy = get_anatomy_from_path(str(f))
    print(anatomy, f)
    df = pd.read_csv(f)
    df['anatomy'] = anatomy
    dataframes.append(df)
print(f'{len(dataframes)} dataframes read')
merged_df = pd.concat(dataframes, join='inner').sort_index()
MERGED_DF_FILENAME = Path('results/challengeR/merged-metric-log.csv')
MERGED_DF_FILENAME.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(MERGED_DF_FILENAME,header=True,index=False)
