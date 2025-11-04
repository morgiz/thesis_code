import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyproj import Transformer

# Load AIS data
with open('aistrack_points_257122880_2025-09-23T12_38_53+00_00-2025-09-24T12_38_40+00_00.json', 'r') as f:
    ais_data = json.load(f)
features = ais_data['features']
lons = [feature['geometry']['coordinates'][0] for feature in features]
lats = [feature['geometry']['coordinates'][1] for feature in features]

# Load map image
img = mpimg.imread(r'1f8f38ac05af2bcfb5dde22d526416857e0ed60e\printout copy.jpg')
img_height, img_width = img.shape[0], img.shape[1]

# World file parameters (from your file)
A = 18.75
C = 240781.96588019
F = 6601721.1390248
E = -18.75

# Transform lon/lat to map projection (assume UTM zone 32N, EPSG:32632 for SE Norway)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
xs, ys = transformer.transform(lons, lats)

# Convert projected coordinates to pixel positions
x_pixels = [(x - C) / A for x in xs]
y_pixels = [(y - F) / E for y in ys]

# Plot
plt.imshow(img)
plt.plot(x_pixels, y_pixels, marker='o', linestyle='-', color='b')
plt.title('AIS Ship Track on Map')
plt.axis('off')
plt.show()