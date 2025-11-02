import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

try:
    
    df_stations_world = pd.read_csv('C:/Users/Avaneesh/Downloads/archive (2)/charging_stations_2025_world.csv',encoding='latin1')
    df_country_summary = pd.read_csv('C:/Users/Avaneesh/Downloads/archive (2)/country_summary_2025.csv')
    df_stations_ml = pd.read_csv('C:/Users/Avaneesh/Downloads/archive (2)/charging_stations_2025_ml.csv')
    print("All datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: One or more files not found. Please ensure all CSVs are in the directory. Error: {e}")
    exit()


print("DATA SET'S INFORMATION")
print(df_stations_world.info())
print(df_country_summary.info())
print(df_stations_ml.info())

print("DATA SET'S STATISTICS")
print(df_stations_world.describe())
print(df_country_summary.describe())
print(df_stations_ml.describe())

print("TOP 5 ROWS OF DATA SETS")
print("Country Summary of 2025")
print(df_country_summary.head())
print("Charging stations ML of 2025")
print(df_stations_ml.head())
print("World Charging stations of 2025")
print(df_stations_world.head())

print("BOTTOM 3 ROWS OF DATA SET")
print("Country summary")
print(df_country_summary.tail())
print("Charging Station ML")
print(df_stations_ml.tail())
print("Charging Station World")
print(df_stations_world.tail())

df_stations_world['state_province'] = df_stations_world['state_province'].fillna('Unknown')
df_stations_world['city'] = df_stations_world['city'].fillna('Unknown')
df_stations_world['power_kw'] = df_stations_world['power_kw'].fillna(0)

print("NULL VALUES OF THE DATA SETS")
print(df_country_summary.isna().sum())
print(df_stations_ml.isna().sum())
print(df_stations_world.isna().sum())

print("HANDLING MISSING VALUES")
print(df_country_summary.fillna(0))
print(df_stations_ml.fillna(0))
print(df_stations_world.fillna(0))

df_top_10 = df_country_summary.sort_values(by='stations', ascending=False).head(10)
print(f"Top 10 Countries by Stations:\n{df_top_10[['country_code', 'stations']]}")

average_ports=df_stations_world['ports'].mean()
print("Average Charging ports in the World : ",average_ports)
total_ports=df_stations_world['ports'].sum()
print("Total no. of Ports in the World : ",total_ports)

df_stations_world['state_province'] = df_stations_world['state_province'].str.lower().str.strip()
corrections = {
    'keral': 'kerala',
    'keraka': 'kerala',
    'lerala': 'kerala',
    'ka': 'karnataka',
    'karnatak': 'karnataka',
    'karnataka': 'karnataka',
    'tamilnadu': 'tamil nadu',
    'tamilnadu ': 'tamil nadu',
    'tamil nadu ': 'tamil nadu',
    'tamil nasdu': 'tamil nadu',
    'mahrashtra': 'maharashtra',
    'west bengal': 'west bengal',
    'villupuram': 'tamil nadu',
    'chennai': 'tamil nadu',
    'bangalore urban': 'karnataka',
    'uttar pradesh': 'uttar pradesh',
    'uttar  pradesh': 'uttar pradesh',
    'india': pd.NA,
    'unknown': pd.NA
}
df_stations_world['state_province'] = df_stations_world['state_province'].replace(corrections).str.capitalize()
indian_ports=len(df_stations_world[df_stations_world['country_code']=='IN'])
print("ports in india : ",indian_ports)
indian_ports=df_stations_world[df_stations_world['country_code']=='IN']
ports_by_state = indian_ports['state_province'].value_counts()
print("Number of ports by Indian states: ",ports_by_state)

plt.figure(figsize=(10,6))
sns.lineplot(
    x='country_code',
    y='stations',
    data=df_top_10,
    marker='o',              
    linestyle='-',           
    color='steelblue'        
)
plt.title('Top 10 Countries - Charging Stations ')
plt.xlabel('Country Code')
plt.ylabel('Total Charging Stations')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(
    x=ports_by_state.values,
    y=ports_by_state.index,                       
    palette='Blues_d',       
)
plt.title('Indian Charging stations ')
plt.xlabel('No. of Ports')
plt.ylabel('States')
plt.tight_layout()
plt.show()

power_class_counts = df_stations_world['power_class'].value_counts().reset_index()
power_class_counts.columns = ['Power_Class', 'Count']
print(power_class_counts.head())
plt.figure(figsize=(10,6))
sns.barplot(
    x='Power_Class',
    y='Count',
    data=power_class_counts,
    palette='coolwarm',
    hue='Power_Class',
    legend=False

)
plt.title('Global Distribution of Charging Station Power Classes')
plt.xlabel('Power Class')
plt.ylabel('Total Number of Stations')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    x='power_kw', 
    y='ports', 
    data=df_stations_world,
    alpha=0.5
)
plt.xscale('log')
plt.title('Power vs Number of Ports')
plt.grid(True,linestyle='-', alpha=0.5)
plt.show()


country_stats = df_stations_world.groupby('country_code')['power_kw'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='country_code',
    y='power_kw',
    data=country_stats.head(20),
    s=100,
    color='dodgerblue'
)
plt.title('Average Charging Power (kW) by Country')
plt.xlabel('Country Code')
plt.ylabel('Average Power (kW)')
plt.xticks(rotation=45)
plt.grid(True,linestyle='-', alpha=0.5)
plt.show()

print("")
print("")
df_stations_world = df_stations_world.dropna(subset=['is_fast_dc', 'ports', 'latitude', 'longitude', 'country_code'])
label_enc = LabelEncoder() #converting text to number 
df_stations_world['country_encode'] = label_enc.fit_transform(df_stations_world['country_code'].astype(str)) #fit -> groups similar rows as single row
df_stations_world['is_fast_dc_encoded'] = label_enc.fit_transform(df_stations_world['is_fast_dc'].astype(str)) #transform -> asigning dense rank for each row
df_stations_world['power_class_encoded'] = label_enc.fit_transform(df_stations_world['power_class'])
X = df_stations_world[['ports', 'latitude', 'longitude', 'is_fast_dc_encoded', 'country_encode']] #features for training
y = df_stations_world['power_class_encoded']#target prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% test and 80% training
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(" Accuracy percentage:", accuracy_score(y_test, y_pred)*100)
print("\n Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

df_stations_world = df_stations_world.dropna(subset=['country_code', 'is_fast_dc'])
df_stations_world['is_fast_dc'] = df_stations_world['is_fast_dc'].astype(str).str.lower().isin(['true', 'yes', '1'])
region_stats = df_stations_world.groupby('country_code').agg(
    total_stations=('ports', 'sum'),
    fast_stations=('is_fast_dc', 'sum')
).reset_index()
region_stats['fast_ratio'] = (region_stats['fast_stations'] / region_stats['total_stations']) * 100
lagging_regions = region_stats.sort_values(by='fast_ratio', ascending=True)
print("Regions lagging in fast charger deployment:")
print(lagging_regions['country_code'].head(10))
print("Total number of stations per Country : ",region_stats['fast_stations'].value_counts().sum())
print("Number of lagging stations per Country : ",region_stats['fast_stations'].value_counts()[0])

df_stations_world = df_stations_world.merge(region_stats[['country_code', 'fast_ratio']], on='country_code', how='left')
X = df_stations_world[['latitude', 'longitude', 'fast_ratio']]
y = df_stations_world['is_fast_dc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Prediction Accuracy(Fast charging based on Country):", accuracy_score(y_test, y_pred)*100)
