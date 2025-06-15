import wfdb
import os
import numpy as np

from wfdb import rdrecord, rdann

# Paths
data_path = r"/workspaces/ecg-beat-classifier/Data/mit-bih-arrhythmia-database"
normal_beats = []
abnormal_beats = []

# Beat types
NORMAL_LABEL = 'N'

# Read all record names
records = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209',
    '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

print("🔍 Extracting beats...")

for record_name in records:
    try:
        record = rdrecord(os.path.join(data_path, record_name))
        annotation = rdann(os.path.join(data_path, record_name), 'atr')
        signal = record.p_signal[:, 0]  # Take first channel

        for idx, label in zip(annotation.sample, annotation.symbol):
            # Get beat window: 100 samples before and after (total 200)
            if idx > 100 and idx + 100 < len(signal):
                beat = signal[idx-100:idx+100]
                if label == NORMAL_LABEL:
                    normal_beats.append(beat)
                else:
                    abnormal_beats.append(beat)

        print(f"✅ Processed record {record_name} - Normals: {len(normal_beats)}, Abnormals: {len(abnormal_beats)}")

    except Exception as e:
        print(f"⚠️ Skipped {record_name} due to error: {e}")

np.save(r"/workspaces/ecg-beat-classifier/Data/normal_beats.npy", normal_beats)
np.save(r"/workspaces/ecg-beat-classifier/Data/abnormal_beats.npy", abnormal_beats)

print("✅ Beat extraction complete!")
print(f"📦 Saved {len(normal_beats)} normal beats")
print(f"📦 Saved {len(abnormal_beats)} abnormal beats")
