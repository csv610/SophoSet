import csv
import sys
import os
from tabulate import tabulate

# Check if the input file is provided as a command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <input_csv_file>")
    sys.exit(1)

# Get the input filename from command line argument
input_filename = sys.argv[1]

# Check if the input file exists
if not os.path.isfile(input_filename):
    print(f"Error: The file '{input_filename}' does not exist.")
    sys.exit(1)

# Generate the output filename
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}.txt"

# Read the CSV file
try:
    with open(input_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read the first row as headers
        data = list(csvreader)  # Read the rest of the data
except csv.Error as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Create the table
table = tabulate(data, headers=headers, tablefmt="grid")

# Write the table to a text file
try:
    with open(output_filename, 'w') as output_file:
        output_file.write(table)
    print(f"Table has been read from '{input_filename}' and saved to '{output_filename}'")
except IOError as e:
    print(f"Error writing to output file: {e}")
    sys.exit(1)
