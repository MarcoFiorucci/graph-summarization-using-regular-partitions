#!/usr/bin/bash



echo "-.-.-.-.-.-.- MAP EXPERIMENT -.-.-.-.-.-.-"
# Performs a single experiment with the specified n
# It creates: 
# - a folder csv used to compute the result of the queries and the performances
# - a folder db, the database

# If you want to repeat the experiment, you should delete the folder ./db in order to reconstruct the database

n=1000

echo "DB GENERATION"
python3 gen_db.py "$n"
clear

echo ".CSV GENERATION (performing queries)"
python3 gen_csv.py
clear

echo "PLOT GENERATION"
python3 map_plot.py "$n"
clear

echo "-.-.-.-.-.-.- ROBUSTNESS EXPERIMENT -.-.-.-.-.-.-"
# Generates a .csv in folder csv and then plot it
python3 robustness_experiment.py

