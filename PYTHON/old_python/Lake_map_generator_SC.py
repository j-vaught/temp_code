import os
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

# ─── CONFIGURE OSMnx & LOCAL CACHE ───────────────────────
ox.settings.use_cache    = True
ox.settings.cache_folder = "cache"       # where Overpass responses are cached
ox.settings.log_console  = False

def load_or_fetch_geojson(path, fetch_fn):
    """
    Load a GeoDataFrame from `path` if it exists;
    otherwise call `fetch_fn()`, save to `path`, and return it.
    """
    if os.path.exists(path):
        print(f"[CACHE] Loading        : {path}")
        gdf = gpd.read_file(path)
    else:
        print(f"[FETCH] Downloading    : {path}")
        gdf = fetch_fn()
        gdf.to_file(path, driver="GeoJSON")
    print(f" → {os.path.basename(path)}: {gdf.shape[0]} rows, CRS={gdf.crs}")
    return gdf

# ─── 1. SOUTH CAROLINA BOUNDARY ────────────────────────────
def fetch_sc_boundary():
    gdf = ox.geocode_to_gdf("South Carolina, USA")
    print("Fetched SC boundary; preview:")
    print("  columns:", list(gdf.columns))
    print("  sample:", gdf.head(1))
    return gdf.to_crs(epsg=3857)

sc = load_or_fetch_geojson("cache/sc_boundary.geojson", fetch_sc_boundary)

# ─── 2. ALL SC LAKES ───────────────────────────────────────
def fetch_sc_lakes():
    print("Fetching all lakes via Overpass…")
    tags = {"natural": "water", "water": "lake"}
    gdf = ox.features_from_place("South Carolina, USA", tags=tags)
    print("  raw columns:", list(gdf.columns))
    print("  sample head:\n", gdf.head(3))
    return gdf.to_crs(epsg=3857)

# ← Load & cache your full lakes layer here
lakes = load_or_fetch_geojson("cache/sc_lakes.geojson", fetch_sc_lakes)

# ─── 2.1 FILTER OUT TINY “DOT” WATERBODIES ────────────────
# compute area (in m²) and drop anything below your threshold
lakes["area_m2"] = lakes.geometry.area
print(f"[DEBUG] lake areas (min→max): {lakes.area_m2.min():.0f} m² → {lakes.area_m2.max():.0f} m²")
# e.g. keep only lakes larger than 0.5 km² (500 000 m²)
threshold = 1_000_000
large_lakes = lakes[lakes.area_m2 > threshold].copy()
print(f"[DEBUG] kept {len(large_lakes)} lakes > {threshold/1e6:.1f} km²")

# ─── 3. ALL SC CITIES/TOWNS ─────────────────────────────────
def fetch_sc_cities():
    print("Fetching SC cities/towns via Overpass…")
    tags = {"place": ["city", "town"]}
    gdf = ox.features_from_place("South Carolina, USA", tags=tags)
    print(f"  fetched {len(gdf)} features")
    gdf = gdf.to_crs(epsg=3857)
    # keep only Point geometries
    pts = gdf[gdf.geometry.geom_type == "Point"]
    print(f"  points only: {len(pts)}")
    # drop duplicate names so each town appears once
    unique = pts.drop_duplicates(subset="name")
    print(f"  unique names: {len(unique)} (after dropping dup names)")
    return unique

cities = load_or_fetch_geojson("cache/sc_cities.geojson", fetch_sc_cities)

# ─── 4. FILTER YOUR POI LAKES ───────────────────────────────
poi_names = [
    "Lake Murray", "Lake Moultrie", "Lake Marion", "Lake Greenwood",
    "Lake Jocassee", "Lake Wylie", "Lake Monticello", "Lake Hartwell",
    "Lake Wateree", "Lake Keowee", "Lake Strom Thurmond",
    "Richard B. Russell Lake", "Clarks Hill Lake",
]
poi_lakes = lakes[lakes["name"].isin(poi_names)]
print(f"[FILTER] POI lakes found: {len(poi_lakes)}")
print("  matched names:", poi_lakes["name"].unique())

# ─── 5. FILTER TO YOUR SELECTED CITIES ─────────────────────
selected_cities = [
    "Aiken", "Greenville", "Beaufort", "Orangeburg", "Greenwood", 
    "Rock Hill", "Union", "Anderson", "Chester", "Florence", 
    "Sumter", "Chesterfield", "Myrtle Beach", "Charleston", 
    "Columbia", "Newberry"
]
cities = cities[cities["name"].isin(selected_cities)]
print(f"[FILTER] Selected cities found: {len(cities)}")
print("  matched names:", cities["name"].unique())

# ─── 6. PLOT EVERYTHING ─────────────────────────────────────
print("\n[ PLOTTING ]")
fig, ax = plt.subplots(figsize=(20, 24))

print(" • plotting state boundary…")
sc.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

print(" • plotting all lakes (light blue)…")
lakes.plot(ax=ax, facecolor="lightblue", edgecolor="none", alpha=0.5)

print(" • highlighting POI lakes (red)…")
poi_lakes.plot(ax=ax, facecolor="red", edgecolor="darkred", linewidth=0.7)

print(" • plotting selected cities…")
cities.plot(ax=ax, marker="o", color="black", markersize=60)

print(" • adding city labels…")
for _, row in cities.iterrows():
    ax.text(
        row.geometry.x,
        row.geometry.y + 3_000,
        row["name"],
        fontsize=9,
        ha="center",
        va="bottom"
    )

ax.set_axis_off()
ax.set_title(
    "South Carolina: Selected Cities & POI Lakes\n"
    "(Lakes in Red, Cities as Black Dots)",
    pad=25
)
plt.tight_layout()
#plt.show()
plt.savefig("sc_selected_lakes_cities.png", dpi=300, bbox_inches="tight")



