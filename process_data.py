import os
import pandas as pd
import numpy as np

input_folder = os.path.join(os.getcwd(), 'input_csv')
output_folder = os.path.join(os.getcwd(), 'output_ladybug')
os.makedirs(output_folder, exist_ok=True)

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

def fill_missing_with_previous_day(df):
    for column in df.columns:
        mask = df[column].isna()
        df.loc[mask, column] = df[column].shift(24)
    return df

def create_ladybug_file(data, column_name, output_file):
    with open(output_file, 'w') as f:
        for value in data[column_name]:
            f.write(f"{value}\n")

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, sep=';', skiprows=1)
    
    df['datetime(UTC+02)'] = pd.to_datetime(df['datetime(UTC+02)'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('datetime(UTC+02)', inplace=True)
    
    # Remove February 29th
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    
    start_date = '2024-04-23'
    end_date = '2024-06-09'
    filtered_df = df[start_date:end_date]
    
    # Resample to hourly data
    hourly_df = filtered_df.resample('H').mean()
    
    # Fill missing data with values from the previous day
    hourly_df = fill_missing_with_previous_day(hourly_df)
    
    # Generate a full year's hourly timestamps without February 29th
    full_year = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='H')
    full_year = full_year[~((full_year.month == 2) & (full_year.day == 29))]
    
    # Reindex to full year, filling missing values with data from the same hour of the previous day
    hourly_avg_full = hourly_df.reindex(full_year)
    hourly_avg_full = fill_missing_with_previous_day(hourly_avg_full)
    
    # Fill any remaining NaN values with 0
    hourly_avg_full = hourly_avg_full.fillna(0)
    
    base_name = os.path.splitext(file)[0]
    create_ladybug_file(hourly_avg_full, 'temperature(C)', os.path.join(output_folder, f'{base_name}_temperature_ladybug.txt'))
    create_ladybug_file(hourly_avg_full, 'humidity(%)', os.path.join(output_folder, f'{base_name}_humidity_ladybug.txt'))
    create_ladybug_file(hourly_avg_full, 'co2(ppm)', os.path.join(output_folder, f'{base_name}_co2_ladybug.txt'))

print(f"Ladybug input files have been created in the {output_folder} folder.")

# Verify the number of lines in each output file
for output_file in os.listdir(output_folder):
    file_path = os.path.join(output_folder, output_file)
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    print(f"{output_file}: {line_count} lines")