import os
import pandas as pd

from utils.helpers import *
from utils.params import *



# Converting txt files to csv and fixing formatting
for vehicle in vehicles:
    vehicleFolder = os.path.join(datasetFolder, vehicle)
    csvPaths = []
    for _, _, list in os.walk(vehicleFolder):
        for txt in sorted(list):
            txtPath = os.path.join(vehicleFolder, txt)
            csv_output = txtPath.replace('.txt', '.csv')
            if csv_output not in csvPaths:
                csvPaths.append(csv_output)

            lines = []

            if not os.path.exists(csv_output):
                # Creating dataset
                df = pd.DataFrame(columns=columns)

                with open(txtPath, 'r') as file:
                    for i, line in enumerate(file):
                        print(f'[💾 {txtPath}] {i+1}', end='\r')

                        strip = line.strip().split(',')
                        if 'no-attack' in txtPath:
                            while len(strip) < 11:
                                strip.append('00')
                            strip.append('R')
                        else:
                            while len(strip) < 12:
                                strip.insert(-1, '00')

                        line = pd.Series(strip, index=columns)
                        lines.append(line)
                df = pd.DataFrame(lines, columns=columns)
                df.to_csv(csv_output, index=False)
                print()

    # Merging and processing datasets
    dfs = []
    merged = f'./dataset/{vehicle}.csv'
    if not os.path.exists(merged):
        for csv in csvPaths:
            print(f'[⚙️ PROCESSING] {csv}')
            df = pd.read_csv(csv)

            # Convert hexadecimal values to binary
            hex_columns = ['DATA [0]', 'DATA [1]', 'DATA [2]', 'DATA [3]', 'DATA [4]', 'DATA [5]', 'DATA [6]', 'DATA [7]']

            for i, col in enumerate(hex_columns):
                    # Middle step by converting as int
                    int_col = df[col].apply(int, base=16)
                    # Convert the byte feature to binary string representation
                    binary_representation = int_col.apply(lambda x: format(x, '08b'))
                    # Split the binary string into separate bit features
                    split_bits = binary_representation.apply(lambda x: [int(bit) for bit in x])
                    # Add the split bits as separate columns to the DataFrame
                    for j in range(8):
                        df[f'Bit_{i*8+j}'] = split_bits.apply(lambda x: x[j])

            df.drop(columns=hex_columns, inplace=True)

            # Convert timestamp to datetime and calculate delta time
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            df['Interval'] = df.groupby('CAN ID')['Timestamp'].diff().dt.total_seconds().fillna(0)
            df.drop('Timestamp', axis=1, inplace=True)

            df['Flag'] = df['Flag'].map({'R': 0, 'T': 1})

            dfs.append(df)

        vehicle_df = pd.concat(dfs, ignore_index=True)

        df.to_csv(merged, index=False)