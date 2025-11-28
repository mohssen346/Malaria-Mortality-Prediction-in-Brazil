import pandas as pd

# Brazilian state abbreviations to full names
sigla_to_name = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapa',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceara',
    'DF': 'Distrito Federal',
    'ES': 'Espirito Santo',
    'GO': 'Goias',
    'MA': 'Maranhao',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Para',
    'PB': 'Paraiba',
    'PR': 'Parana',
    'PE': 'Pernambuco',
    'PI': 'Piaui',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondonia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'Sao Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}


# Function to load and melt yearly data from excel files
def load_and_melt_yearly(file_name, value_name, sheet_name=0, county_col='Year'):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    # Rename the first column to 'County' since it's mislabeled as 'Year' but contains county codes
    df.rename(columns={county_col: 'County'}, inplace=True)
    # Melt to long format: County, Year, Value
    df_melted = df.melt(id_vars=['County'], var_name='Year', value_name=value_name)
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
    return df_melted.dropna(subset=['Year'])


# Load Building Surface Area
df_building = load_and_melt_yearly('Data/Building_surface_area.xlsx', 'Building_Area')

# Load Gross Domestic Product
df_gdp = load_and_melt_yearly('Data/Gross_Domestic_Product.xlsx', 'GDP')

# Merge on County and Year
df_combined_yearly = pd.merge(df_building, df_gdp, on=['County', 'Year'], how='outer')

# Load updated Subnational Unit data
df_subnational = pd.read_excel('Data/Subnational Unit-data_Edit_1.xlsx')
# Extract metrics
df_metrics = df_subnational[['Name', 'Year', 'Deaths', 'Clinical Cases']].rename(columns={'Name': 'State', 'Clinical Cases': 'Clinical_Cases'})







# Load microregions mapping
df_micro = pd.read_excel('Data/Microregions mapping.xlsx', sheet_name='br_microrregioes_2021')
# Create county_to_sigla
county_to_sigla = dict(zip(df_micro['CD_MICRO'], df_micro['SIGLA']))
# Create county_to_state with full names
county_to_state = {county: sigla_to_name.get(sigla, 'Unknown') for county, sigla in county_to_sigla.items()}
# Create county_to_name
county_to_name = dict(zip(df_micro['CD_MICRO'], df_micro['NM_MICRO']))

# Add State and County_Name to combined yearly
df_combined_yearly['State'] = df_combined_yearly['County'].map(county_to_state)
df_combined_yearly['County_Name'] = df_combined_yearly['County'].map(county_to_name)

# Merge with metrics on State and Year
df_combined = pd.merge(df_combined_yearly, df_metrics, on=['State', 'Year'], how='outer')

# For environmental data (weekly/monthly), aggregate to yearly
# Palmer Drought Severity Index (monthly)
df_pdsi = pd.read_excel('Data/Palmer_Drought_Severity_Index.xlsx', sheet_name='Palmer_Drought_Severity_Index')
df_pdsi.rename(columns={'Month': 'County'}, inplace=True)
date_columns = df_pdsi.columns[1:]
numeric_dates = pd.to_numeric(date_columns, errors='coerce')
dates = pd.to_datetime(numeric_dates, origin='1899-12-30', unit='d', errors='coerce')
df_pdsi.columns = ['County'] + dates.tolist()
df_pdsi_melted = df_pdsi.melt(id_vars=['County'], var_name='Date', value_name='PDSI')
df_pdsi_melted['Year'] = df_pdsi_melted['Date'].dt.year
df_pdsi_yearly = df_pdsi_melted.groupby(['County', 'Year'])['PDSI'].mean().reset_index()

# Function for weekly data
def aggregate_weekly(file_name, value_name, agg_func='mean', sheet_name=0):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    df.rename(columns={'Epi Weeks': 'County'}, inplace=True)
    week_columns = df.columns[1:]
    df_melted = df.melt(id_vars=['County'], var_name='EpiWeek', value_name=value_name)
    df_melted['Year'] = df_melted['EpiWeek'].apply(lambda x: int(str(x)[:4]))
    df_agg = df_melted.groupby(['County', 'Year'])[value_name].agg(agg_func).reset_index()
    return df_agg

# Aggregate all weekly/monthly data
df_precip = aggregate_weekly('Data/Sum_precipitation.xlsx', 'Precipitation', agg_func='sum', sheet_name='Sum_precipitation')
df_mean_temp = aggregate_weekly('Data/Mean_temperature.xlsx', 'Mean_Temperature', agg_func='mean', sheet_name='Mean_temperature')
df_max_temp = aggregate_weekly('Data/Maximum_temperature.xlsx', 'Max_Temperature', agg_func='mean', sheet_name='Maximum_temperature')
df_min_temp = aggregate_weekly('Data/Minimum_temperature.xlsx', 'Min_Temperature', agg_func='mean', sheet_name='Minimum_temperature')
df_wind = aggregate_weekly('Data/Mean_wind_speed.xlsx', 'Wind_Speed', agg_func='mean', sheet_name='Mean_wind_speed')
df_rh = aggregate_weekly('Data/Mean_relative_humidity.xlsx', 'Rel_Humidity', agg_func='mean', sheet_name='Mean_relative_humidity')
df_atm = aggregate_weekly('Data/Mean_atmospheric_pressure.xlsx', 'Atm_Pressure', agg_func='mean', sheet_name='Mean_atmospheric_pressure')
df_ndvi = aggregate_weekly('Data/Mean_Normalized_Difference_Vegetation_Index.xlsx', 'NDVI', agg_func='mean', sheet_name='Mean_Normalized_Difference_Vege')

# Merge all to df_combined
df_combined = pd.merge(df_combined, df_pdsi_yearly, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_precip, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_mean_temp, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_max_temp, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_min_temp, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_wind, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_rh, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_atm, on=['County', 'Year'], how='outer')
df_combined = pd.merge(df_combined, df_ndvi, on=['County', 'Year'], how='outer')

# Sort by County, Year
df_combined.sort_values(['County', 'Year'], inplace=True)

# Save to new file
df_combined.to_csv('combined_data.csv', index=False)

print("Combined data saved to combined_data.csv")
print(df_combined.head())