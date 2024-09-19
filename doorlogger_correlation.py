import os
import pandas as pd
import numpy as np
import math
from scipy import stats

# Define thermal comfort parameters
AIR_SPEED = 0.1  # m/s
CLOTHING_LEVEL = 0.6  # clo
METABOLIC_RATE = 1.0  # met
EXTERNAL_WORK = 0  # met

def calculate_pmv(ta, tr, vel, rh, met, clo, wme):
    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    fcl = 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(vel)
    taa = ta + 273
    tra = tr + 273
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015
    n = 0
    hc = hcf  # Initialize hc with hcf
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            return 1000
    tcl = 100 * xn - 273
    hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15)
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100.0, 4))
    hl6 = fcl * hc * (tcl - ta)
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return pmv

def calculate_ppd(pmv):
    return 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0))

def calculate_ppd_from_temp_rh(temperature, relative_humidity, air_speed, clothing_level, metabolic_rate, external_work):
    pmv = calculate_pmv(temperature, temperature, air_speed, relative_humidity, metabolic_rate, clothing_level, external_work)
    ppd = calculate_ppd(pmv)
    return ppd

def load_data(file_path):
    return pd.read_csv(file_path, header=None, names=['value'])

def load_door_logger_data(file_path):
    with open(file_path, 'r') as f:
        door_data = [1 - (float(line.strip()) > 0) for line in f]  # Invert values: 0 becomes 1 (open), any positive value becomes 0 (closed)
    return door_data

def process_data(temp_file, humidity_file, co2_file, door_file=None):
    temp_data = pd.read_csv(temp_file, header=None, names=['temperature'])
    humidity_data = pd.read_csv(humidity_file, header=None, names=['humidity'])
    co2_data = pd.read_csv(co2_file, header=None, names=['co2'])

    date_range = pd.date_range(start='2024-01-01', periods=8760, freq='H')
    df = pd.DataFrame({
        'temperature': temp_data['temperature'].values,
        'humidity': humidity_data['humidity'].values,
        'co2': co2_data['co2'].values
    }, index=date_range)

    df['ppd'] = df.apply(lambda row: calculate_ppd_from_temp_rh(
        row['temperature'], row['humidity'], AIR_SPEED, CLOTHING_LEVEL, METABOLIC_RATE, EXTERNAL_WORK
    ), axis=1)

    if door_file:
        door_data = load_door_logger_data(door_file)
        df['door_open'] = door_data

    return df

def analyze_correlations(df, door_data):
    df = df.copy()  
    df['door_open'] = door_data
    
    # Calculate correlations
    corr_ppd_door = df['ppd'].corr(df['door_open'])
    corr_co2_door = df['co2'].corr(df['door_open'])
    
    # Calculate average PPD and CO2 when door is open vs closed
    ppd_door_open = df[df['door_open'] == 1]['ppd'].mean()
    ppd_door_closed = df[df['door_open'] == 0]['ppd'].mean()
    co2_door_open = df[df['door_open'] == 1]['co2'].mean()
    co2_door_closed = df[df['door_open'] == 0]['co2'].mean()
    
    # Perform t-tests
    ppd_ttest = stats.ttest_ind(df[df['door_open'] == 1]['ppd'], df[df['door_open'] == 0]['ppd'])
    co2_ttest = stats.ttest_ind(df[df['door_open'] == 1]['co2'], df[df['door_open'] == 0]['co2'])
    
    return {
        'corr_ppd_door': corr_ppd_door,
        'corr_co2_door': corr_co2_door,
        'ppd_door_open': ppd_door_open,
        'ppd_door_closed': ppd_door_closed,
        'co2_door_open': co2_door_open,
        'co2_door_closed': co2_door_closed,
        'ppd_ttest': ppd_ttest,
        'co2_ttest': co2_ttest
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'analysis_data')
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    sensor_files = {}
    for file in os.listdir(input_folder):
        if file.endswith('_ladybug.txt'):
            parts = file.split('_')
            sensor_name = '_'.join(parts[:-3])  # Group by everything before the date
            data_type = parts[-2]  # co2, humidity, or temperature
            
            if sensor_name not in sensor_files:
                sensor_files[sensor_name] = {}
            
            sensor_files[sensor_name][data_type] = file

    for sensor_name, files in sensor_files.items():
        if not all(data_type in files for data_type in ['temperature', 'humidity', 'co2']):
            print(f"Error: Missing data files for sensor {sensor_name}")
            continue

        door_file = os.path.join(input_folder, f'{sensor_name}_door_logger.txt')
        
        df = process_data(
            os.path.join(input_folder, files['temperature']),
            os.path.join(input_folder, files['humidity']),
            os.path.join(input_folder, files['co2']),
            door_file if os.path.exists(door_file) else None
        )

        start_date = '2024-04-23'
        end_date = '2024-06-09'
        df_filtered = df[start_date:end_date]

        if 'door_open' in df_filtered.columns:
            door_data_filtered = df_filtered['door_open'].values
            correlations = analyze_correlations(df_filtered, door_data_filtered)

            print(f"\nCorrelations for {sensor_name}:")
            print(f"Correlation between PPD and door opening: {correlations['corr_ppd_door']:.4f}")
            print(f"Correlation between CO2 and door opening: {correlations['corr_co2_door']:.4f}")
            print(f"Average PPD when door is open: {correlations['ppd_door_open']:.2f}%")
            print(f"Average PPD when door is closed: {correlations['ppd_door_closed']:.2f}%")
            print(f"Average CO2 when door is open: {correlations['co2_door_open']:.2f} ppm")
            print(f"Average CO2 when door is closed: {correlations['co2_door_closed']:.2f} ppm")
            print(f"PPD t-test p-value: {correlations['ppd_ttest'].pvalue:.4f}")
            print(f"CO2 t-test p-value: {correlations['co2_ttest'].pvalue:.4f}")
        else:
            print(f"Warning: No door logger data available for {sensor_name}")

if __name__ == "__main__":
    main()