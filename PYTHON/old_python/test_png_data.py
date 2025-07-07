import pandas as pd
from PIL import Image, ExifTags

def extract_exif(path):
    img = Image.open(path)
    exif_data = img._getexif() or {}
    tags = {}
    for tag_id, value in exif_data.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        if tag == 'GPSInfo' and isinstance(value, dict):
            gps = {}
            for gps_id, gps_val in value.items():
                gps_tag = ExifTags.GPSTAGS.get(gps_id, gps_id)
                gps[gps_tag] = gps_val
            tags['GPSInfo'] = gps
        else:
            tags[tag] = value
    return pd.DataFrame.from_dict(tags, orient='index', columns=['Value']).reset_index().rename(columns={'index': 'Tag'})

# Paths to the images
path1 = 'IMG_A47E4345-1880-4B15-99D8-579427396596.jpeg'
path2 = 'IMG_987453BF-D944-4158-8C1D-E602038EF56C.jpeg'

# Extract EXIF and display
df1 = extract_exif(path1)
df2 = extract_exif(path2)

print("EXIF Metadata for First Image:")
print(df1.to_string())
print("\nEXIF Metadata for Second Image:")
