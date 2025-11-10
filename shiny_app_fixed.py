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
import os
from pathlib import Path

# Import H3 for accurate hex boundaries
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    print("Warning: H3 not available, using approximate hex shapes")

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_CENTER_LAT = 30.0
DEFAULT_CENTER_LON = -99.0
DEFAULT_ZOOM = 6
MAX_HEXES_FAST_MODE = 5000
HEX_RADIUS = 0.02

# -----------------------------
# Utility Functions
# -----------------------------
def get_h3_hex_boundary(hex_id):
    """Get actual H3 hex boundary coordinates (compatible with both H3 v3 and v4+)"""
    try:
        # Try new API first (H3 v4+)
        boundary = h3.cell_to_boundary(hex_id)
    except (AttributeError, NameError):
        try:
            # Fall back to old API (H3 v3)
            boundary = h3.h3_to_geo_boundary(hex_id)
        except (AttributeError, NameError):
            # If H3 not available, create approximate hex
            return hex_corners_fallback(0, 0, HEX_RADIUS)
    
    # Convert from (lat, lon) to [lat, lon] list format for Folium
    return [[lat, lon] for lat, lon in boundary]

def hex_corners_fallback(lat, lng, radius=HEX_RADIUS):
    """Fallback approximate hexagon corners (only used if H3 unavailable)"""
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
# Data Loading Functions
# -----------------------------
def load_hex_data_file(file_path='hex_data.feather'):
    """Load hex data from feather file"""
    try:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return None
        
        df = pd.read_feather(file_path)
        print(f"Loaded {len(df)} hexes from {file_path}")
        
        # Validate required columns
        required_cols = ['hex_id', 'population', 'score', 'lat', 'lng']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return None
        
        # Filter to hexes with population > 0
        df = df[df['population'] > 0].copy()
        print(f"Filtered to {len(df)} hexes with population > 0")
        
        if len(df) == 0:
            print("Warning: No hexes with population > 0")
            return None
        
        return df
        
    except Exception as e:
        print(f"Error loading hex data: {e}")
        return None

def load_tract_data_file(file_path='tract_data.feather'):
    """Load tract outline data from feather file"""
    try:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return None
        
        df = pd.read_feather(file_path)
        print(f"Loaded {len(df)} tract outlines from {file_path}")
        
        # Validate required columns for tract data
        required_cols = ['GEOID', 'geometry_coords']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing tract columns {missing_cols}")
            return None
        
        return df
        
    except Exception as e:
        print(f"Error loading tract data: {e}")
        return None

def create_sample_data():
    """Create sample hex data for testing when real data is not available"""
    print("Creating sample hex data for testing...")
    
    # Create sample hexes around Texas metros
    metros = {
        "Austin": (30.2672, -97.7431),
        "Dallas": (32.7767, -96.7970),
        "Houston": (29.7604, -95.3698),
        "San Antonio": (29.4241, -98.4936),
    }
    
    sample_data = []
    
    for metro_name, (lat, lon) in metros.items():
        # Create hexes in a grid around each metro
        for i in range(-3, 4):
            for j in range(-3, 4):
                offset_lat = i * 0.05
                offset_lon = j * 0.05
                
                hex_lat = lat + offset_lat
                hex_lon = lon + offset_lon
                
                # Generate a real H3 hex ID for this location if H3 is available
                if H3_AVAILABLE:
                    try:
                        # Try new API first (H3 v4+)
                        try:
                            hex_id = h3.latlng_to_cell(hex_lat, hex_lon, 6)
                        except AttributeError:
                            # Fall back to old API (H3 v3)
                            hex_id = h3.geo_to_h3(hex_lat, hex_lon, 6)
                    except Exception as e:
                        hex_id = f'{metro_name}_{i}_{j}'
                else:
                    hex_id = f'{metro_name}_{i}_{j}'
                
                # Distance from metro center (for population simulation)
                distance = math.sqrt(offset_lat**2 + offset_lon**2)
                
                # Simulate population (higher near center)
                population = max(0, int(10000 * math.exp(-distance * 2)))
                
                if population > 100:  # Only include significant population
                    sample_data.append({
                        'hex_id': hex_id,
                        'population': population,
                        'score': population / 1000000,  # Density score
                        'hex_area_m2': 1000000,  # 1 km²
                        'lat': hex_lat,
                        'lng': hex_lon
                    })
    
    df = pd.DataFrame(sample_data)
    print(f"Created {len(df)} sample hexes")
    return df

def calculate_summary_stats(df):
    """Calculate summary statistics from hex data"""
    if df is None or len(df) == 0:
        return {
            'total_hexes': 0,
            'total_population': 0,
            'score_min': 0,
            'score_max': 0,
            'score_mean': 0,
            'center_lat': DEFAULT_CENTER_LAT,
            'center_lon': DEFAULT_CENTER_LON
        }
    
    return {
        'total_hexes': len(df),
        'total_population': float(df['population'].sum()),
        'score_min': float(df['score'].min()),
        'score_max': float(df['score'].max()),
        'score_mean': float(df['score'].mean()),
        'center_lat': float(df['lat'].mean()),
        'center_lon': float(df['lng'].mean())
    }

# -----------------------------
# UI Components
# -----------------------------
def create_summary_stats_html(summary, filtered_count=0):
    """Create summary stats HTML"""
    if summary['total_hexes'] == 0:
        return "<div style='padding:15px;color:red;'>No data available</div>"
    
    efficiency = f" • Showing {filtered_count:,} ({filtered_count/summary['total_hexes']*100:.1f}%)" if filtered_count else ""
    
    return f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;margin-bottom:15px;">
        <h4>Dataset Summary</h4>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
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

def create_color_legend_html(summary):
    """Create color legend HTML"""
    if summary['total_hexes'] == 0:
        return ""
    
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
# Map Creation
# -----------------------------
def create_folium_map(df_filtered, tract_data, center_lat, center_lng, zoom, show_hexes=True, show_tracts=False):
    """Create Folium map with hex polygons and optional tract outlines"""
    
    # Create base map with specified center and zoom
    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=zoom, 
        tiles='CartoDB positron'
    )
    
    # Add tract outlines if requested and available
    if show_tracts and tract_data is not None and len(tract_data) > 0:
        print(f"Adding {len(tract_data)} tract outlines to map")
        for _, row in tract_data.iterrows():
            try:
                # Parse the geometry coordinates
                import json
                coords = json.loads(row['geometry_coords'])
                
                folium.Polygon(
                    locations=coords,
                    color='blue',
                    weight=1,
                    fill=False,
                    opacity=0.7,
                    popup=folium.Popup(
                        f"<div style='font-family:Arial;'>"
                        f"<b>Census Tract:</b> {row['GEOID']}<br>"
                        f"<b>Population:</b> {row.get('population', 'N/A'):,}<br>"
                        f"</div>",
                        max_width=250
                    )
                ).add_to(m)
            except Exception as e:
                print(f"Error adding tract {row.get('GEOID', 'unknown')}: {e}")
                continue
    
    # Add hexes if requested and available
    if show_hexes and df_filtered is not None and len(df_filtered) > 0:
        # Create colormap based on population
        min_pop = df_filtered['population'].min()
        max_pop = df_filtered['population'].max()
        
        if max_pop > min_pop:
            colormap = cm.LinearColormap(['green','yellow','red'], vmin=min_pop, vmax=max_pop)
        else:
            colormap = cm.LinearColormap(['green'], vmin=0, vmax=1)
        
        # Add hex polygons using actual H3 boundaries
        for _, row in df_filtered.iterrows():
            try:
                # Use actual H3 hex boundary if hex_id is available and H3 is installed
                if 'hex_id' in row and H3_AVAILABLE and pd.notna(row['hex_id']):
                    hex_boundary = get_h3_hex_boundary(row['hex_id'])
                else:
                    # Fallback to approximate hex around centroid
                    hex_boundary = hex_corners_fallback(row['lat'], row['lng'], radius=HEX_RADIUS)
                
                folium.Polygon(
                    locations=hex_boundary,
                    color='white',
                    weight=0.5,
                    fill=True,
                    fill_color=colormap(row['population']),
                    fill_opacity=0.6,
                    popup=folium.Popup(
                        f"<div style='font-family:Arial;'>"
                        f"<b>Population:</b> {int(row['population']):,}<br>"
                        f"<b>Median Income:</b> ${int(row.get('median_income', 0)):,}<br>"
                        f"<b>Income Level:</b> {row.get('target_status', 'Unknown')}"
                        f"</div>",
                        max_width=250
                    )
                ).add_to(m)
            except Exception as e:
                print(f"Error adding hex {row.get('hex_id', 'unknown')}: {e}")
                continue
    
    return m

# -----------------------------
# Shiny UI
# -----------------------------
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("Texas Metro Population Hex Map"),
        ui.tags.meta(charset="utf-8"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1")
    ),
    
    ui.div(
        ui.h2("Texas Metro Population Hex Map", style="text-align:center;color:#2c3e50;margin-bottom:20px"),
        
        # Status and controls
        ui.div(
            ui.output_ui("loading_status"),
            style="margin-bottom:15px"
        ),
        
        ui.div(
            ui.row(
                ui.column(2, ui.input_action_button("refresh", "Refresh Map", class_="btn-primary btn-block")),
                ui.column(2, ui.input_checkbox("fast_mode", "Fast Loading Mode", value=False)),
                ui.column(2, ui.input_numeric("zoom_level", "Zoom Level", value=DEFAULT_ZOOM, min=1, max=12, step=1)),
                ui.column(2, ui.input_checkbox("show_stats", "Show Statistics", value=True)),
                ui.column(2, ui.input_checkbox("show_hexes", "Show Hexes", value=True)),
                ui.column(2, ui.input_checkbox("show_tracts", "Show Census Tracts", value=False))
            ),
            style="margin-bottom:20px"
        ),
        
        # Summary statistics
        ui.div(
            ui.output_ui("summary_stats"),
            ui.output_ui("color_legend")
        ),
        
        # Map container
        ui.div(
            ui.output_ui("map_plot"),
            style="margin-top:20px"
        ),
        
        style="max-width:1200px;margin:0 auto;padding:20px"
    )
)

# -----------------------------
# Shiny Server
# -----------------------------
def server(input, output, session):
    # Reactive values
    hex_data = reactive.Value(None)
    tract_data = reactive.Value(None)
    summary_data = reactive.Value({})
    loading_message = reactive.Value("Initializing...")
    filtered_count = reactive.Value(0)
    
    # Initialize data on startup
    @reactive.Effect
    def load_initial_data():
        loading_message.set("Loading data...")
        
        # Try to load hex data first
        df_hex = load_hex_data_file('hex_data.feather')
        
        # Fall back to sample data if real data not available
        if df_hex is None:
            loading_message.set("Real hex data not found, creating sample data...")
            df_hex = create_sample_data()
        
        # Try to load tract data
        df_tract = load_tract_data_file('tract_data.feather')
        
        if df_hex is not None and len(df_hex) > 0:
            hex_data.set(df_hex)
            summary = calculate_summary_stats(df_hex)
            summary_data.set(summary)
            
            if df_tract is not None:
                tract_data.set(df_tract)
                loading_message.set(f"✅ Loaded {len(df_hex):,} hexes and {len(df_tract):,} tract outlines")
            else:
                loading_message.set(f"✅ Loaded {len(df_hex):,} hexes successfully (no tract data)")
        else:
            loading_message.set("❌ Failed to load data")
    
    # Refresh data when refresh button is clicked - call the effect trigger
    @reactive.Effect
    @reactive.event(input.refresh)
    def refresh_data():
        # Force reload by updating a reactive value that triggers load_initial_data
        loading_message.set("Refreshing data...")
        
        # Manually reload data (same logic as load_initial_data)
        df_hex = load_hex_data_file('hex_data.feather')
        
        if df_hex is None:
            loading_message.set("Real hex data not found, creating sample data...")
            df_hex = create_sample_data()
        
        df_tract = load_tract_data_file('tract_data.feather')
        
        if df_hex is not None and len(df_hex) > 0:
            hex_data.set(df_hex)
            summary = calculate_summary_stats(df_hex)
            summary_data.set(summary)
            
            if df_tract is not None:
                tract_data.set(df_tract)
                loading_message.set(f"✅ Refreshed {len(df_hex):,} hexes and {len(df_tract):,} tract outlines")
            else:
                loading_message.set(f"✅ Refreshed {len(df_hex):,} hexes successfully (no tract data)")
        else:
            loading_message.set("❌ Failed to refresh data")
    
    # Calculate filtered data based on zoom and viewport
    @reactive.Calc
    def get_filtered_data():
        df = hex_data.get()
        if df is None or len(df) == 0:
            filtered_count.set(0)
            return None
        
        zoom = input.zoom_level()
        fast_mode = input.fast_mode()
        summary = summary_data.get()
        
        # Get viewport bounds
        center_lat = summary.get('center_lat', DEFAULT_CENTER_LAT)
        center_lng = summary.get('center_lon', DEFAULT_CENTER_LON)
        bounds = get_zoom_level_bounds(center_lat, center_lng, zoom)
        
        # Filter by viewport
        df_filtered = filter_hexes_by_viewport(df, bounds, zoom)
        
        # Apply fast mode sampling if needed
        if fast_mode and len(df_filtered) > MAX_HEXES_FAST_MODE:
            sample_rate = MAX_HEXES_FAST_MODE / len(df_filtered)
            df_filtered = df_filtered.sample(frac=sample_rate, random_state=42).copy()
        
        filtered_count.set(len(df_filtered))
        return df_filtered
    
    # Output: Loading status
    @output
    @render.ui
    def loading_status():
        message = loading_message.get()
        if "✅" in message:
            color = "#28a745"
        elif "❌" in message:
            color = "#dc3545"
        else:
            color = "#007bff"
        
        return ui.div(
            ui.p(message, style=f"color:{color};font-weight:bold;text-align:center;margin:0;"),
            style="padding:10px;border-radius:5px;background-color:#f8f9fa;"
        )
    
    # Output: Summary statistics
    @output
    @render.ui
    def summary_stats():
        if not input.show_stats():
            return ui.div()
        
        summary = summary_data.get()
        count = filtered_count.get()
        html_content = create_summary_stats_html(summary, count)
        return ui.HTML(html_content)
    
    # Output: Color legend
    @output
    @render.ui
    def color_legend():
        if not input.show_stats():
            return ui.div()
        
        summary = summary_data.get()
        html_content = create_color_legend_html(summary)
        return ui.HTML(html_content)
    
    # Output: Map
    @output
    @render.ui
    def map_plot():
        try:
            df_filtered = get_filtered_data() if input.show_hexes() else None
            tract_df = tract_data.get() if input.show_tracts() else None
            summary = summary_data.get()
            zoom = input.zoom_level()
            
            if summary.get('total_hexes', 0) == 0 and not input.show_tracts():
                return ui.div(
                    ui.p("No data available to display", style="text-align:center;color:#666;"),
                    style="padding:50px;border:1px solid #ddd;border-radius:5px;"
                )
            
            # Map center from summary
            center_lat = summary.get('center_lat', DEFAULT_CENTER_LAT)
            center_lng = summary.get('center_lon', DEFAULT_CENTER_LON)
            
            # Create map
            m = create_folium_map(
                df_filtered, 
                tract_df, 
                center_lat,
                center_lng,
                zoom, 
                show_hexes=input.show_hexes(), 
                show_tracts=input.show_tracts()
            )
            
            # Return map HTML
            map_html = m._repr_html_()
            return ui.HTML(map_html)
            
        except Exception as e:
            error_msg = f"Error creating map: {str(e)}"
            print(error_msg)
            return ui.div(
                ui.p(error_msg, style="color:red;text-align:center;"),
                style="padding:50px;border:1px solid #ddd;border-radius:5px;"
            )

# -----------------------------
# Create App
# -----------------------------
app = App(app_ui, server)

# -----------------------------
# Additional Setup for Production
# -----------------------------
def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'folium', 'branca']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing required packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main entry point for the application"""
    print("🚀 Starting Texas Metro Population Hex Map...")
    
    if not check_requirements():
        print("❌ Cannot start app due to missing requirements")
        return
    
    print("📊 App ready! Access it in your browser.")
    print("💡 If you don't have hex_data.feather, the app will use sample data")
    
    # Note: In production, you would typically run the app with:
    # shiny run app.py
    # or similar command depending on your deployment setup

if __name__ == "__main__":
    main()