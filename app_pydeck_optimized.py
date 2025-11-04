"""
Folium-Based Shiny App for Texas Metro Population Hex Map
Ultra-high performance visualization using precomputed hex centroids
"""

import math
import json
import time
import pandas as pd
from shiny import App, ui, render, reactive
import folium
import branca.colormap as cm

# -----------------------------
# Utility Functions
# -----------------------------
def hex_corners(lat, lng, radius=0.02):
    """Return approximate hexagon corners around a centroid (lat, lng)"""
    corners = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.radians(angle_deg)
        dlat = radius * math.sin(angle_rad)
        dlng = radius * math.cos(angle_rad) / math.cos(math.radians(lat))
        corners.append([lat + dlat, lng + dlng])
    return corners

def get_zoom_level_bounds(lat, lng, zoom, width_pixels=1200, height_pixels=800):
    """Calculate approximate lat/lng bounds for viewport (for filtering)"""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    lat_deg_per_pixel = 360 / (256 * n)
    lng_deg_per_pixel = 360 / (256 * n * math.cos(lat_rad))
    lat_offset = (height_pixels / 2) * lat_deg_per_pixel
    lng_offset = (width_pixels / 2) * lng_deg_per_pixel
    min_lat = max(-85, lat - lat_offset)
    max_lat = min(85, lat + lat_offset)
    min_lng = lng - lng_offset
    max_lng = lng + lng_offset
    return (min_lat, min_lng, max_lat, max_lng)

def filter_hexes_by_viewport(df, bounds, zoom, buffer_factor=1.5):
    """Filter hexes by viewport bounds with buffer"""
    if df.empty:
        return df
    min_lat, min_lng, max_lat, max_lng = bounds
    lat_buffer = (max_lat - min_lat) * (buffer_factor - 1) / 2
    lng_buffer = (max_lng - min_lng) * (buffer_factor - 1) / 2
    min_lat -= lat_buffer
    max_lat += lat_buffer
    min_lng -= lng_buffer
    max_lng += lng_buffer
    mask = (df['lat'] >= min_lat) & (df['lat'] <= max_lat) & (df['lng'] >= min_lng) & (df['lng'] <= max_lng)
    return df[mask].copy()

# -----------------------------
# Cached Data Loader
# -----------------------------
@reactive.Calc
def load_hex_data():
    """Load hex_data.feather and summary"""
    try:
        df = pd.read_feather('hex_data.feather')
        df = df[df['population'] > 0].copy()
        summary = {
            'total_hexes': len(df),
            'total_population': float(df['population'].sum()),
            'score_min': float(df['score'].min()),
            'score_max': float(df['score'].max()),
            'score_mean': float(df['score'].mean()),
            'center_lat': 30.0,
            'center_lon': -99.0
        }
        return {'data': df, 'summary': summary}
    except Exception as e:
        print(f"Error loading hex data: {e}")
        return None

# -----------------------------
# Summary Stats and Legend
# -----------------------------
def create_summary_stats(hex_data, filtered_count=0):
    if not hex_data:
        return "No data available"
    summary = hex_data['summary']
    efficiency = f" • Showing {filtered_count:,} ({filtered_count/summary['total_hexes']*100:.1f}%)" if filtered_count else ""
    return f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;margin-bottom:15px;">
        <h4>Dataset Summary</h4>
        <div style="display:flex;gap:20px;">
            <div><strong>Total Hexes:</strong> {summary['total_hexes']:,}</div>
            <div><strong>Total Population:</strong> {summary['total_population']:,.0f}</div>
            <div><strong>Score Range:</strong> {summary['score_min']:.4f} - {summary['score_max']:.4f}</div>
            <div><strong>Mean Score:</strong> {summary['score_mean']:.4f}</div>
        </div>
        <div style="margin-top:10px;font-size:12px;color:#666;">
            Powered by Folium Hex Polygons{efficiency}
        </div>
    </div>
    """

def create_color_legend(hex_data):
    if not hex_data:
        return ""
    summary = hex_data['summary']
    return f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;">
        <h5>Population Density Scale</h5>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:12px;">Low</span>
            <div style="width:200px;height:20px;background:linear-gradient(to right, green,yellow,red);border:1px solid #ccc;"></div>
            <span style="font-size:12px;">High</span>
        </div>
        <div style="display:flex;justify-content:space-between;width:220px;font-size:11px;margin-top:5px;">
            <span>{summary['score_min']:.4f}</span>
            <span>{summary['score_max']:.4f}</span>
        </div>
    </div>
    """

# -----------------------------
# Shiny UI
# -----------------------------
app_ui = ui.page_fluid(
    ui.h2("Texas Metro Population Hex Map", style="text-align:center;color:#2c3e50"),
    ui.output_ui("loading_status"),
    ui.output_ui("summary_stats"),
    ui.output_ui("color_legend"),
    ui.input_action_button("refresh", "Refresh Map", class_="btn-primary"),
    ui.input_checkbox("fast_mode", "Fast Loading Mode"),
    ui.input_numeric("zoom_level", "Zoom Level", value=6, min=0, max=12, step=1),
    ui.output_ui("map_plot")
)

# -----------------------------
# Shiny Server
# -----------------------------
def server(input, output, session):
    hex_data = reactive.Value(None)
    viewport_info = reactive.Value({'filtered_count': 0, 'total_count': 0})
    loading_state = reactive.Value("Loading map data...")

    @reactive.Calc
    @reactive.event(lambda: True)
    def load_data_reactive():
        data = load_hex_data()
        hex_data.set(data)
        if data:
            loading_state.set("Map data loaded")
        else:
            loading_state.set("Error loading data")
        return data

    @output
    @render.ui
    def loading_status():
        status = loading_state.get()
        color = "#007bff" if "Loading" in status else "#28a745" if "loaded" in status else "#dc3545"
        return ui.HTML(f"<p style='text-align:center;color:{color};font-weight:bold'>{status}</p>")

    @output
    @render.ui
    def summary_stats():
        data = hex_data.get()
        vinfo = viewport_info.get()
        return ui.HTML(create_summary_stats(data, vinfo.get('filtered_count', 0)))

    @output
    @render.ui
    def color_legend():
        data = hex_data.get()
        return ui.HTML(create_color_legend(data))

    @output
    @render.ui
    def map_plot():
        data = hex_data.get()
        if not data or data['data'].empty:
            return ui.HTML("<p>No data to display</p>")

        df = data['data']
        zoom = input.zoom_level()
        fast_mode = input.fast_mode()
        center_lat, center_lng = 30.0, -99.0

        bounds = get_zoom_level_bounds(center_lat, center_lng, zoom)
        df_filtered = filter_hexes_by_viewport(df, bounds, zoom)

        if fast_mode and len(df_filtered) > 5000:
            df_filtered = df_filtered.iloc[::5].copy()

        viewport_info.set({'filtered_count': len(df_filtered), 'total_count': len(df)})

        m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, tiles='CartoDB positron')
        min_pop, max_pop = df_filtered['population'].min(), df_filtered['population'].max()
        colormap = cm.LinearColormap(['green','yellow','red'], vmin=min_pop, vmax=max_pop)

        for _, row in df_filtered.iterrows():
            folium.Polygon(
                locations=hex_corners(row['lat'], row['lng'], radius=0.02),
                color='white',
                weight=0.5,
                fill=True,
                fill_color=colormap(row['population']),
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>Hex:</b> {row['hex_id']}<br>"
                    f"<b>Population:</b> {row['population']:,}<br>"
                    f"<b>Score:</b> {row.get('score',0):.4f}",
                    max_width=250
                )
            ).add_to(m)

        return ui.HTML(m._repr_html_())

app = App(app_ui, server)
