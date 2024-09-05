import pandas as pd


def move_data(i):
    # Load the CSV files
    filtered_companies = pd.read_csv('csvs/filtered_companies.csv')
    company_data = pd.read_csv('csvs/company_data.csv')

    # Drop any columns that have 'Unnamed' in their name
    filtered_companies = filtered_companies.loc[:, ~filtered_companies.columns.str.contains('^Unnamed')]
    company_data = company_data.loc[:, ~company_data.columns.str.contains('^Unnamed')]

    # Select the first row from filtered_companies
    first_row = filtered_companies.iloc[[i]]  # Make sure you use the correct index (0 is the first row)

    # Concatenate the first row to the company_data DataFrame
    company_data = pd.concat([company_data, first_row], ignore_index=True)

    # Save the updated company_data DataFrame back to CSV
    company_data.to_csv('csvs/company_data.csv', index=False)


