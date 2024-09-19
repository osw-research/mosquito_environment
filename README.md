# mosquito_environment
Python Scripts to take environmental data from Aranet Pro4 data loggers and create mosquito risk heatmaps
How to use:
Download the csv files for the period you want to analyse from Aranet Cloud
Create a folder in the same directory as the python scripts called 'input_csv' and put the csv files in there
run the 'process_data.py' script to create the text files for temp, humidity and co2 that will be used to make the charts. Change the start and end date depending on the study period you want to analyse. The script has been designed to create text files with 8760 values, corresponding to each hour of the year and compatable with Ladybug.
The 'doorlog.py' script can be used to create doorlog text files for every hour the door was open for over 10 minutes. 
Put the overlay pdf in the chart folder. 
run the 'chart_maker.py' script to create the visualisations from this data. 'chart_maker_door.py' to add the door logger data. 
run the 'door_cori.py' script to run a statistical analysis between the PPD and CO2 data. 
run the 'analysis.py' to create an excel spreadsheet to compare data from data loggers. 
