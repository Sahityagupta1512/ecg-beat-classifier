import wfdb
import os

# Create a folder to store data
os.makedirs(r"F:\2409029\Part 1 - Sem 2\FDS - Kalpesh Sir\ECG_Streamlitapp\Data/mit-bih-arrhythmia-database", exist_ok=True)

# List of record names
records = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209',
    '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

for record in records:
    print(f"⬇️ Downloading record {record}...")
    wfdb.dl_database(
        'mitdb',
        os.path.join('Data', 'mit-bih-arrhythmia-database'),
        [record]
    )

print("✅ Download complete!")
