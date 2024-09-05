import csv

# List of companies to filter
companies = [
    "Carlton & United Breweries",
    "United States Golf Association (USGA)",
    "United States Naval Academy",
    "United Tractors",
    "United States Institute of Peace",
    "United Robotics Group",
    "United Family Healthcare",
    "United Talent Agency",
    "United Surgical Partners International, Inc",
    "United Way",
    "United Launch Alliance (ULA)",
    "Manchester United",
    "United Colors of Benetton India",
    "Leeds United",
    "Royal United Services Institute",
    "United Biscuits",
    "United Rentals",
    "United Natural Foods",
    "United Microelectronics Corporation (UMC)",
    "West Ham United FC",
    "United Airlines",
    "UnitedHealth Group",
    "United Soccer Coaches",
    "United Utilities",
    "United Nations Volunteers",
    "United States Military Academy at West Point",
    "United Nations Global Compact",
    "United States Postal Service",
    "United Nations Office for Disaster Risk Reduction (UNDRR)",
    "United States Tennis Association (USTA)"
]

# Columns to extract
columns = [
    "Company",
    "Industry",
    "Website",
    "Company Linkedin Url",
    "Facebook Url",
    "Twitter Url",
    "Company Street",
    "Company City",
    "Company State",
    "Company Country",
    "Company Postal Code",
    "Company Address",
    "Keywords",
    "Company Phone",
    "SEO Description",
    "Short Description",
    "Founded Year"
]

# Open the CSV file
with open('apollo.csv', mode='r') as infile:
    reader = csv.DictReader(infile)

    # Filtered data list
    filtered_data = []

    # Iterate over the rows in the CSV
    for row in reader:
        if row['Company'] in companies:
            # Extract only the specified columns
            filtered_row = {col: row[col] for col in columns}
            filtered_data.append(filtered_row)

# Write the filtered data to a new CSV file
output_filename = 'filtered_companies.csv'
with open(output_filename, mode='w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=columns)

    # Write header
    writer.writeheader()

    # Write filtered rows
    writer.writerows(filtered_data)

print(f"Filtered data has been written to '{output_filename}' successfully.")
