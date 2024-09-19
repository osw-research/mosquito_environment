import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib as mpl
import math
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import PyPDF2

# Set font properties for editable text in PDF
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Helvetica'

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

def read_door_logger_data(file_path):
    print(f"Reading door logger data from: {file_path}")
    with open(file_path, 'r') as f:
        door_data = []
        for i, line in enumerate(f):
            value = float(line.strip())
            door_data.append(value == 0)
            if i < 5:  # Print first 5 lines for debugging
                print(f"Line {i}: {line.strip()} -> {value > 0}")
    print(f"Total lines read: {len(door_data)}")
    return door_data

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'output_ladybug')
output_folder = os.path.join(script_dir, 'charts')
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
    exit(1)

files = [f for f in os.listdir(input_folder) if f.endswith('_temperature_ladybug.txt')]
if not files:
    print(f"Error: No temperature files found in '{input_folder}'.")
    exit(1)

for file in files:
    base_name = file.replace('_temperature_ladybug.txt', '')
    temp_file = os.path.join(input_folder, file)
    humidity_file = os.path.join(input_folder, f'{base_name}_humidity_ladybug.txt')
    co2_file = os.path.join(input_folder, f'{base_name}_co2_ladybug.txt')

    if not all(os.path.exists(f) for f in [temp_file, humidity_file, co2_file]):
        print(f"Error: Missing data files for {base_name}")
        continue

    temp_data = pd.read_csv(temp_file, header=None, names=['temperature'])
    humidity_data = pd.read_csv(humidity_file, header=None, names=['humidity'])
    co2_data = pd.read_csv(co2_file, header=None, names=['co2'])

    date_range = pd.date_range(start='2023-01-01', periods=8760, freq='H')
    df = pd.DataFrame({
        'temperature': temp_data['temperature'].values,
        'humidity': humidity_data['humidity'].values,
        'co2': co2_data['co2'].values
    }, index=date_range)

    df['ppd'] = df.apply(lambda row: calculate_ppd_from_temp_rh(
        row['temperature'], row['humidity'], AIR_SPEED, CLOTHING_LEVEL, METABOLIC_RATE, EXTERNAL_WORK
    ), axis=1)

    start_date = '2023-04-23 00:00:00'
    end_date = '2023-06-09 23:00:00'
    df_filtered = df[start_date:end_date]

    # Load door logger data
    door_logger_file = os.path.join(input_folder, 'ladybug_door_open_data.txt')
    if os.path.exists(door_logger_file):
        door_open_data = read_door_logger_data(door_logger_file)
        door_open_data = door_open_data[df_filtered.index[0].dayofyear * 24:df_filtered.index[-1].dayofyear * 24 + 24]
    else:
        door_open_data = None

    if door_open_data is not None:
        print(f"Number of door open hours: {sum(door_open_data)}")
        print(f"First few door open values: {door_open_data[:10]}")
    else:
        print("No door data found")

    ppd_heatmap = df_filtered['ppd'].values.reshape(-1, 24).T[::-1]
    co2_heatmap = df_filtered['co2'].values.reshape(-1, 24).T[::-1]

    # Define the new color scheme
    colors = ['white', '#FFE5E5', '#FFCCCC', '#FFB2B2', '#FF9999', '#C11414']
    n_bins = 6
    ppd_cmap = mcolors.LinearSegmentedColormap.from_list('custom_ppd', colors, N=n_bins)

    # Create boundaries for the color bins
    bounds = [0, 50, 60, 70, 80, 90, 100]
    norm = mcolors.BoundaryNorm(bounds, ppd_cmap.N)

    fig = plt.figure(figsize=(25, 12))  # Reduced overall height
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 2])  # Adjusted height ratio
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

# Daily Mean Temperature and CO2 Levels (ax1)
daily_temp = df_filtered['temperature'].resample('D').mean()
daily_co2 = df_filtered['co2'].resample('D').mean()
daily_temp_std = df_filtered['temperature'].resample('D').std()
daily_co2_std = df_filtered['co2'].resample('D').std()

ax1_temp, ax1_co2 = ax1, ax1.twinx()
ax1_temp.plot(daily_temp.index, daily_temp.values, color='#C11414', label='Temperature')
ax1_co2.plot(daily_co2.index, daily_co2.values, color='#9a9a9a', label='CO2')
ax1_temp.fill_between(daily_temp.index, daily_temp - daily_temp_std, daily_temp + daily_temp_std, color='#C11414', alpha=0.25)
ax1_co2.fill_between(daily_co2.index, daily_co2 - daily_co2_std, daily_co2 + daily_co2_std, color='#9a9a9a', alpha=0.25)

ax1_temp.set_ylabel('Temperature (°C)', color='#C11414')
ax1_co2.set_ylabel('CO2 (ppm)', color='#9a9a9a')
ax1_temp.set_ylim(25, 35)
ax1_co2.set_ylim(400, 1000)
ax1.set_title('DAILY MEAN TEMPERATURE AND CO2 LEVELS', color='black')

date_range = pd.date_range(start=df_filtered.index[0], end=df_filtered.index[-1], freq='D')
ax1.set_xlim(date_range[0], date_range[-1])
ax1.set_xticks(date_range)
ax1.xaxis.set_visible(False)

lines1, labels1 = ax1_temp.get_legend_handles_labels()
lines2, labels2 = ax1_co2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Hourly Mean PPD and CO2 Levels (ax2)
hourly_ppd = df_filtered['ppd'].groupby(df_filtered.index.hour).mean()
hourly_co2 = df_filtered['co2'].groupby(df_filtered.index.hour).mean()
hourly_ppd_std = df_filtered['ppd'].groupby(df_filtered.index.hour).std()
hourly_co2_std = df_filtered['co2'].groupby(df_filtered.index.hour).std()

ax2_ppd, ax2_co2 = ax2, ax2.twiny()
ax2_ppd.plot(hourly_ppd.values, range(23, -1, -1), color='#C11414', label='PPD')
ax2_co2.plot(hourly_co2.values, range(23, -1, -1), color='#9a9a9a', label='CO2')
ax2_ppd.fill_betweenx(range(23, -1, -1), hourly_ppd - hourly_ppd_std, hourly_ppd + hourly_ppd_std, color='#C11414', alpha=0.25)
ax2_co2.fill_betweenx(range(23, -1, -1), hourly_co2 - hourly_co2_std, hourly_co2 + hourly_co2_std, color='#9a9a9a', alpha=0.25)

ax2_ppd.set_xlabel('PPD (%)', color='#C11414')
ax2_co2.set_xlabel('CO2 (ppm)', color='#9a9a9a')
ax2_ppd.set_xlim(0, 100)
ax2_co2.set_xlim(400, 1000)
ax2.set_title('HOURLY MEAN PPD AND CO2 LEVELS', color='black')
ax2.set_ylabel('Hour of Day', color='black')
ax2.set_ylim(23, 0)
ax2.set_yticks(range(23, -1, -1))
ax2.set_yticklabels(range(0, 24))

lines1, labels1 = ax2_ppd.get_legend_handles_labels()
lines2, labels2 = ax2_co2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Heatmap (ax3)
im = sns.heatmap(ppd_heatmap, ax=ax3, cmap=ppd_cmap, norm=norm, cbar_kws={'label': 'PPD (%)', 'ticks': bounds}, linewidths=0.5, linecolor='white')
ax3.set_title(f'THERMAL COMFORT (PPD) - {base_name.upper()}\n(APRIL 23 TO JUNE 09)', color='black')
ax3.set_xlabel('Date', color='black')
ax3.set_ylabel('Hour of Day', color='black')
ax3.set_xticks(np.arange(0, ppd_heatmap.shape[0], 1))
ax3.set_xticklabels([(pd.to_datetime(start_date) + pd.Timedelta(days=i)).day for i in range(ppd_heatmap.shape[0])], ha='center')
ax3.set_yticks(np.arange(0.5, 24.5, 1))
ax3.set_yticklabels(range(23, -1, -1))

# Add CO2 rectangles
for i in range(co2_heatmap.shape[0]):
    for j in range(co2_heatmap.shape[1]):
        if co2_heatmap[i, j] > 700:
            rect = Rectangle((j, co2_heatmap.shape[0] - i - 1), 1, 1, fill=False, edgecolor='#9a9a9a', lw=1.5, hatch='...', alpha=0.7)
            ax3.add_patch(rect)

# Add door open indicators
if door_open_data is not None:
    door_open_heatmap = np.array(door_open_data).reshape(-1, 24).T[::-1]
    for i in range(door_open_heatmap.shape[0]):
        for j in range(door_open_heatmap.shape[1]):
            if door_open_heatmap[i, j]:  # This now checks for 0 values
                ax3.plot(j + 0.5, i + 0.5, 'kx', markersize=5, markeredgewidth=2)

# Adjust colorbar
cbar = ax3.collections[0].colorbar
cbar.ax.set_ylabel("PPD (%)", rotation=-90, va="bottom")
cbar.ax.yaxis.set_label_coords(2.0, 0.5)
pos = ax3.get_position()
cbar.ax.set_position([pos.x1 + 0.0001, pos.y0, pos.width * 0.01, pos.height])

plt.tight_layout()

# Adjust the position of ax1 to match the width of ax3
pos1 = ax1.get_position()
pos3 = ax3.get_position()
ax1.set_position([pos3.x0, pos1.y0, pos3.width, pos1.height])

# Remove unnecessary spines and ticks
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1_temp.tick_params(axis='y', which='both', left=True, right=False, colors='#C11414')
ax1_co2.tick_params(axis='y', which='both', left=False, right=True, colors='#9a9a9a')
ax2.tick_params(axis='y', which='both', left=True, right=False)
ax2_ppd.tick_params(axis='x', which='both', top=False, bottom=True, colors='#C11414')
ax2_co2.tick_params(axis='x', which='both', top=True, bottom=False, colors='#9a9a9a')

plt.draw()
output_file_path = os.path.join(output_folder, f'{base_name}_ppd_temp_co2_chart_Apr23_Jun09.pdf')
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
plt.close()

# Overlay PDF
overlay_file = os.path.join(script_dir, 'charts', 'overlay.pdf')
if os.path.exists(overlay_file):
    with open(output_file_path, 'rb') as file1, open(overlay_file, 'rb') as file2:
        pdf1 = PyPDF2.PdfReader(file1)
        pdf2 = PyPDF2.PdfReader(file2)
        page1 = pdf1.pages[0]
        page2 = pdf2.pages[0]
        page1.merge_page(page2)
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(page1)
        with open(output_file_path, 'wb') as output_file:
            pdf_writer.write(output_file)

print(f"\nPDF charts have been saved in the {output_folder} folder.")