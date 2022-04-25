from pydicom import dcmread
import matplotlib.pyplot as plt
import pandas as pd
import os

home_dir = '/Users/jacobazoulay/CS231N_Project/'
image_path = home_dir + 'Images/JUPITERP065L/SER00001/IMG00001.dcm'

ds = dcmread(image_path)
image = ds.pixel_array  # pixel data is stored in 'pixel_array' element

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# All x-rays should look like this, though perhaps flipped/rotated.You might expect a sideways x-ray but not an
# upside-down one.
# Images will have different brightnesses, contrasts, etc.

# pd.read_excel('/Users/jacobazoulay/CS231N_Project/labels.xlsx')
labels = pd.read_excel(home_dir + 'labels.xlsx')
print(labels)

image_path = home_dir + 'Images/' + labels.iloc[1]['lateral x-ray']
print(labels.iloc[1]['lateral x-ray'])
ds = dcmread(image_path)
image = ds.pixel_array

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)

plt.imshow(image, cmap='gray') # x-ray
plt.scatter(labels.iloc[0]['superior_patella_x'], labels.iloc[0]['superior_patella_y']) # superior patella loc in blue
plt.scatter(labels.iloc[0]['inferior_patella_x'], labels.iloc[0]['inferior_patella_y']) # inferior patella loc in orange
plt.scatter(labels.iloc[0]['tibial_plateau_x'], labels.iloc[0]['tibial_plateau_y']) # tibial_plateau loc in green

ax.set_xticks([])
ax.set_yticks([])
plt.show()