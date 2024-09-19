import pandas as pd

def create_door_open_file(door_file, output_file, datetime_column, motorseconds_column):
    # Read the CSV file, skipping the first row (title) and using semicolon as separator
    door_data = pd.read_csv(door_file, skiprows=1, sep=';')
    
    # Parse the datetime column
    door_data[datetime_column] = pd.to_datetime(door_data[datetime_column])
    
    # Convert motorseconds to numeric, replacing any non-numeric values with NaN
    door_data[motorseconds_column] = pd.to_numeric(door_data[motorseconds_column], errors='coerce')
    
    # Set the datetime as index
    door_data.set_index(datetime_column, inplace=True)
    
    # Define a function to determine if the door is open
    def is_door_open(motor_seconds):
        return 0 <= motor_seconds <= 10
    
    # Add a column indicating whether the door is open
    door_data['door_open'] = door_data[motorseconds_column].apply(is_door_open)
    
    # Calculate the time difference between consecutive readings
    door_data['time_diff'] = door_data.index.to_series().diff().dt.total_seconds() / 60
    
    # Calculate the open duration for each reading
    door_data['open_duration'] = door_data['door_open'] * door_data['time_diff']
    
    # Resample the data to hourly intervals and sum the open duration
    door_hourly = door_data.resample('H').agg({
        'open_duration': 'sum'
    })
    
    # Create a new column indicating if the door was open for more than 10 minutes in that hour
    door_hourly['open_more_than_10min'] = (door_hourly['open_duration'] > 10).astype(int)
    
    # Print some diagnostic information
    print(f"\nTotal hours with door open > 10 minutes: {door_hourly['open_more_than_10min'].sum()}")
    print("\nTop 5 hours with longest door open time:")
    print(door_hourly.sort_values('open_duration', ascending=False).head())
    
    # Create a full year of data (non-leap year)
    start_date = door_hourly.index.min().replace(month=1, day=1, hour=0)
    full_year = pd.date_range(start=start_date, freq='H', periods=8760)  # Force 8760 hours
    
    # Reindex to fill the entire year
    door_hourly_full = door_hourly.reindex(full_year).fillna(0)

    # Print values for specific hours
    specific_hours = [
        '2024-04-23 20:00:00',
        '2024-04-24 13:00:00',
        '2024-04-24 19:00:00',
        '2024-05-14 06:00:00',
        '2024-05-25 20:00:00',
        '2024-05-24 06:00:00'
    ]

    print("\nValues for specific hours:")
    for hour in specific_hours:
        if pd.to_datetime(hour) in door_hourly_full.index:
            open_duration = door_hourly_full.loc[hour, 'open_duration']
            is_open_more_than_10min = door_hourly_full.loc[hour, 'open_more_than_10min']
            print(f"{hour}: Open duration = {open_duration:.2f} minutes, Open > 10 min = {is_open_more_than_10min}")
        else:
            print(f"{hour}: No data available")
    
    # Write the results to a text file
    with open(output_file, 'w') as f:
        for value in door_hourly_full['open_more_than_10min']:
            f.write(f"{value}\n")

    print(f"\nDoor open data has been saved to '{output_file}'")
    print(f"Total lines in output file: {len(door_hourly_full)}")

# Example usage
door_file = 'door_logger.csv'  # Update this to the actual path of your CSV file
output_file = 'door_open_data.txt'

# First, let's print the column names
temp_data = pd.read_csv(door_file, skiprows=1, sep=';')
print("Column names in the CSV file:")
for i, column in enumerate(temp_data.columns):
    print(f"{i}: {column}")

# Now, ask the user to specify which columns to use
datetime_column = input("Enter the name of the datetime column: ")
motorseconds_column = input("Enter the name of the motorseconds column: ")

create_door_open_file(door_file, output_file, datetime_column, motorseconds_column)