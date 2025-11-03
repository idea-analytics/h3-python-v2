import pandas as pd
import geopandas as gpd
import h3
import numpy as np
from shapely.geometry import Polygon, Point
from shiny import App, ui, render, reactive
import plotly.graph_objects as go

# -----------------------------
# Define metro centers
# -----------------------------
metro_centers = {
    "El Paso": (-106.49, 31.76),
    "Austin": (-97.74, 30.27),
    "Dallas": (-96.80, 32.78),
    "Houston": (-95.37, 29.76),
    "Midland-Odessa": (-102.0, 31.84),
    "Lubbock": (-101.85, 33.58),
    "Corpus Christi": (-97.40, 27.80),
    "San Antonio": (-98.49, 29.42)
}

radius_deg = 0.6  # ~40 miles ≈ 0.6 degrees

# -----------------------------
# Load population and tracts
# -----------------------------
def load_population_csv(path='Population-2023-filtered.csv'):
    pop_df = pd.read_csv(path)
    pop_df['Geography_clean'] = pop_df['Geography'].str.replace('1400000US', '', regex=False)
    pop_df['population'] = pop_df['Estimate!!Total:']
    return pop_df[['Geography_clean','population']]

def load_tracts():
    from load_data import load_texas_tracts
    tracts = load_texas_tracts()
    tracts['GEOID'] = tracts['GEOID'].astype(str)
    tracts = tracts.to_crs(epsg=4326)
    return tracts

# -----------------------------
# Filter tracts to metro areas (~40 mile radius)
# -----------------------------
def filter_metro_tracts(tracts_gdf, metro_centers, radius_deg=0.6):
    selected = []
    for name, (lon, lat) in metro_centers.items():
        # bounding box around metro center
        minx, maxx = lon - radius_deg, lon + radius_deg
        miny, maxy = lat - radius_deg, lat + radius_deg
        bbox = gpd.GeoDataFrame(geometry=[Polygon([
            (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
        ])], crs='EPSG:4326')
        intersects = tracts_gdf[tracts_gdf.intersects(bbox.iloc[0].geometry)]
        selected.append(intersects)
    return pd.concat(selected).drop_duplicates().reset_index(drop=True)

# -----------------------------
# Generate H3 hexes
# -----------------------------
def generate_hexes(tracts_gdf, resolution=6):
    hex_ids = set()
    for geom in tracts_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            if geom.geom_type == 'Polygon':
                hex_ids.update(h3.geo_to_cells(geom.__geo_interface__, resolution))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    hex_ids.update(h3.geo_to_cells(poly.__geo_interface__, resolution))
        except:
            continue
    return list(hex_ids)

def hex_to_polygon(hex_id):
    boundary = h3.cell_to_boundary(hex_id)
    return Polygon(boundary)

# -----------------------------
# Fast population distribution
# -----------------------------
def distribute_population_fast(tracts_gdf, resolution=6):
    """
    Filter tracts to metro areas, generate hexes, and distribute population.
    """
    # Filter to metro areas (~40 mi radius)
    metros_geom = []
    for lat, lon in METROS.values():
        point = Point(lon, lat)
        circle = point.buffer(RADIUS_DEG)  # rough circular area
        metros_geom.append(circle)
    
    metro_union = gpd.GeoSeries(metros_geom).unary_union
    tracts_gdf = tracts_gdf[tracts_gdf.intersects(metro_union)].copy()
    
    # Generate H3 hexes covering all tracts
    hex_ids_set = set()
    for idx, tract in tracts_gdf.iterrows():
        geom = tract.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            hexes = h3.geo_to_cells(geom.__geo_interface__, resolution)
            hex_ids_set.update(hexes)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                hexes = h3.geo_to_cells(poly.__geo_interface__, resolution)
                hex_ids_set.update(hexes)
    
    hex_ids = list(hex_ids_set)
    
    # Convert hexes to GeoDataFrame
    hex_polygons = []
    valid_hex_ids = []
    for h in hex_ids:
        try:
            poly = hex_to_polygon(h)
            if poly.is_valid and not poly.is_empty:
                hex_polygons.append(poly)
                valid_hex_ids.append(h)
        except:
            continue
    
    hexes_gdf = gpd.GeoDataFrame({
        "hex_id": valid_hex_ids,
        "geometry": hex_polygons
    }, crs="EPSG:4326")
    
    # Project to planar CRS for area
    hexes_gdf_proj = hexes_gdf.to_crs(epsg=3083)
    hexes_gdf["hex_area_m2"] = hexes_gdf_proj.geometry.apply(lambda g: g.area if g.is_valid and not g.is_empty else 0)
    
    # Initialize population column
    hexes_gdf["population"] = 0.0
    
    # Spatial index for faster intersection
    hex_sindex = hexes_gdf.sindex
    
    # Distribute tract population to hexes
    for idx, tract in tracts_gdf.iterrows():
        tract_geom = tract.geometry
        tract_pop = tract["Estimate!!Total:"]
        if tract_geom is None or tract_geom.is_empty or pd.isna(tract_pop):
            continue
        
        # Find intersecting hexes
        possible_idx = list(hex_sindex.intersection(tract_geom.bounds))
        intersecting_hexes = hexes_gdf.iloc[possible_idx]
        
        if len(intersecting_hexes) == 0:
            continue
        
        # Compute intersection areas
        intersection_areas = []
        hex_indices = []
        for h_idx, hrow in intersecting_hexes.iterrows():
            inter = tract_geom.intersection(hrow.geometry)
            if inter.is_empty or not inter.is_valid:
                continue
            intersection_areas.append(inter.area)
            hex_indices.append(h_idx)
        
        total_inter_area = sum(intersection_areas)
        if total_inter_area == 0:
            continue
        
        # Distribute population proportionally
        for i, h_idx in enumerate(hex_indices):
            weight = intersection_areas[i] / total_inter_area
            hexes_gdf.at[h_idx, "population"] += tract_pop * weight
    
    return hexes_gdf

# -----------------------------
# Score calculation
# -----------------------------
def calculate_scores(hexes_gdf):
    hexes_gdf['score'] = hexes_gdf['population'] / hexes_gdf['hex_area_m2']
    return hexes_gdf

# -----------------------------
# Plotly map
# -----------------------------
def create_map_zoomable(hexes_gdf):
    fig = go.Figure()

    min_score, max_score = hexes_gdf['score'].min(), hexes_gdf['score'].max()

    for idx, row in hexes_gdf.iterrows():
        coords = list(row.geometry.exterior.coords)
        lons, lats = zip(*coords)
        norm = (row['score'] - min_score) / (max_score - min_score) if max_score > min_score else 0.5
        color = f'rgb({int(255*norm)},{int(255*(1-norm)*0.5)},{int(255*(1-norm))})'

        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(width=0.5, color='white'),
            hovertemplate=(
                f"<b>Hex ID:</b> {row['hex_id']}<br>"
                f"<b>Population:</b> {row['population']:.0f}<br>"
                f"<b>Density Score:</b> {row['score']:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center={"lat": 31.5, "lon": -99.5},
            zoom=5  # initial zoom level
        ),
        margin={"l":0,"r":0,"t":0,"b":0},
        height=800
    )

    return fig

# -----------------------------
# Shiny app
# -----------------------------
app_ui = ui.page_fluid(
    ui.h2("Texas Metro Population Hex Map"),
    ui.input_action_button("refresh","Refresh Map"),
    ui.output_ui("map_plot")
)

def server(input, output, session):
    pop_df = load_population_csv()
    tracts_gdf = load_tracts()
    tracts_gdf = tracts_gdf.merge(pop_df, left_on='GEOID', right_on='Geography_clean', how='inner')
    tracts_gdf = filter_metro_tracts(tracts_gdf, metro_centers, radius_deg=0.6)
    hex_ids = generate_hexes(tracts_gdf, resolution=6)
    hexes_gdf = distribute_population_fast(tracts_gdf, hex_ids)
    hexes_gdf = calculate_scores(hexes_gdf)

    @reactive.Calc
    def processed_hexes():
        return hexes_gdf

    @output
    @render.ui
    def map_plot():
        hexes = processed_hexes()
        fig = create_map_zoomable(hexes)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

app = App(app_ui, server)




