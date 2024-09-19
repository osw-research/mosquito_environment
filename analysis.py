import os
import pandas as pd
import numpy as np
import math

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

def process_data(temp_file, humidity_file, co2_file):
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

    return df

def analyze_data(df, start_date, end_date):
    df_filtered = df[start_date:end_date]
    
    stats = {
        'temperature_mean': df_filtered['temperature'].mean(),
        'humidity_mean': df_filtered['humidity'].mean(),
        'co2_mean': df_filtered['co2'].mean(),
        'ppd_mean': df_filtered['ppd'].mean(),
        'comfort_percentage_20': (df_filtered['ppd'] <= 20).mean() * 100,
        'comfort_percentage_50': (df_filtered['ppd'] <= 50).mean() * 100,
        'high_co2_percentage': (df_filtered['co2'] > 700).mean() * 100
    }
    
def calculate_hourly_averages(df):
    hours = [0, 6, 12, 18]
    averages = {}
    for hour in hours:
        hour_data = df[df.index.hour == hour]
        averages[hour] = {
            'ppd': hour_data['ppd'].mean(),
            'temperature': hour_data['temperature'].mean()
        }
    return averages

def analyze_data(df, start_date, end_date):
    try:
        df_filtered = df[start_date:end_date]
        
        if df_filtered.empty:
            print(f"No data found between {start_date} and {end_date}")
            return None

        stats = {
            'temperature': df_filtered['temperature'].describe(),
            'humidity': df_filtered['humidity'].describe(),
            'co2': df_filtered['co2'].describe(),
            'ppd': df_filtered['ppd'].describe(),
            'comfort_percentage_20': (df_filtered['ppd'] <= 20).mean() * 100,
            'comfort_percentage_50': (df_filtered['ppd'] <= 50).mean() * 100,
            'high_co2_percentage': (df_filtered['co2'] > 700).mean() * 100,
            'high_co2_percentage_530': (df_filtered['co2'] > 530).mean() * 100
        }

        # Calculate averages for specific hours
        hours = [0, 6, 12, 18]
        for hour in hours:
            hour_data = df_filtered[df_filtered.index.hour == hour]
            stats[f'ppd_{hour:02d}'] = hour_data['ppd'].mean()
            stats[f'temperature_{hour:02d}'] = hour_data['temperature'].mean()
        
        return stats
    except Exception as e:
        print(f"Error in analyze_data: {str(e)}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'analysis_data')
    output_folder = os.path.join(script_dir, 'output')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Group files by sensor
    sensor_files = {}
    for file in os.listdir(input_folder):
        if file.endswith('_ladybug.txt'):
            parts = file.split('_')
            sensor_name = '_'.join(parts[:-3])  # Group by everything before the date
            data_type = parts[-2]  # co2, humidity, or temperature
            
            if sensor_name not in sensor_files:
                sensor_files[sensor_name] = {}
            
            sensor_files[sensor_name][data_type] = file

    all_stats = {}
    all_hourly_averages = {}

    for sensor_name, files in sensor_files.items():
        if not all(data_type in files for data_type in ['temperature', 'humidity', 'co2']):
            print(f"Error: Missing data files for sensor {sensor_name}")
            continue

        try:
            df = process_data(
                os.path.join(input_folder, files['temperature']),
                os.path.join(input_folder, files['humidity']),
                os.path.join(input_folder, files['co2'])
            )

            print(f"Data range for {sensor_name}: {df.index.min()} to {df.index.max()}")

            start_date = '2024-04-25'
            end_date = '2024-06-09'
            stats = analyze_data(df, start_date, end_date)
            
            if stats is None:
                print(f"Error: Failed to analyze data for sensor {sensor_name}")
                continue

            all_stats[sensor_name] = stats

            hourly_averages = calculate_hourly_averages(df[start_date:end_date])
            all_hourly_averages[sensor_name] = hourly_averages

            print(f"\nStatistics for {sensor_name}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}:\n{value}")

        except Exception as e:
            print(f"Error processing data for sensor {sensor_name}: {str(e)}")

    if not all_stats:
        print("No valid data to create Excel file.")
        return

    # Create Excel file
    excel_file = os.path.join(output_folder, 'all_sensors_statistics.xlsx')
    with pd.ExcelWriter(excel_file) as writer:
        overall_stats = pd.DataFrame(all_stats).T
        overall_stats.to_excel(writer, sheet_name='Overall Statistics')

        # Hourly averages
        hourly_data = pd.DataFrame()
        for sensor, averages in all_hourly_averages.items():
            sensor_data = pd.DataFrame({
                f'{sensor} PPD': [f"{averages[hour]['ppd']:.2f}%" for hour in [0, 6, 12, 18]],
                f'{sensor} Temperature': [f"{averages[hour]['temperature']:.2f}°C" for hour in [0, 6, 12, 18]]
            }, index=['00:00', '06:00', '12:00', '18:00'])
            hourly_data = pd.concat([hourly_data, sensor_data], axis=1)
        
        hourly_data.to_excel(writer, sheet_name='Hourly Averages')

    print(f"\nExcel file with statistics for all sensors has been saved: {excel_file}")

if __name__ == "__main__":
    main()