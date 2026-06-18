import pandas as pd
import numpy as np

# 1. Load the existing features file you showed me
df = pd.read_csv("uclass_processed.csv")

# 2. Extract just the filename column and rename it slightly for the script
labels_df = df[['filename']].copy()
labels_df.rename(columns={'filename': 'FileName'}, inplace=True)

# 3. Assign a random dummy label (0 to 4) to every single file
# 0: NoStutter, 1: Prolongation, 2: Block, 3: SoundRep, 4: WordRep
np.random.seed(42) # Keeps it consistent
labels_df['label'] = np.random.randint(0, 5, size=len(labels_df))

# 4. Save as the new labels file
labels_df.to_csv("uclass_dummy_labels.csv", index=False)
print("✅ Created uclass_dummy_labels.csv with random labels for testing!")
print(labels_df.head())