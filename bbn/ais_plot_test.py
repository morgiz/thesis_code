import json
import matplotlib.pyplot as plt

#Denne koden plotter AIS data fra en JSON fil for Bastø Electric sin ferd fra Hørten til Moss

with open('aistrack_points_257122880_2025-09-23T12_38_53+00_00-2025-09-24T12_38_40+00_00.json', 'r') as f:
    ais_data = json.load(f)

features = ais_data['features']
lons = [feature['geometry']['coordinates'][0] for feature in features]
lats = [feature['geometry']['coordinates'][1] for feature in features]

plt.figure(figsize=(8, 6))
plt.plot(lons, lats, marker='o', linestyle='-', color='b')
plt.title('AIS Ship Track')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()