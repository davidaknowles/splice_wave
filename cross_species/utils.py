import pandas as pd

def merge_intervals(df):
    
    df.sort_values(by=['chrom', 'start'], inplace=True)
    
    merged = []

    for index, row in df.iterrows():
        # If merged list is empty, add the first interval
        if not merged:
            merged.append((row['chrom'], row['start'], row['end']))
        else:
            last = merged[-1]

            # Check if current interval overlaps or is adjacent to the last merged interval
            if row['chrom'] == last[0] and row['start'] <= last[2]:
                merged[-1] = (last[0], last[1], max(last[2], row['end']))
            else:
                merged.append((row['chrom'], row['start'], row['end']))

    return pd.DataFrame(merged, columns=['chrom', 'start', 'end'])
