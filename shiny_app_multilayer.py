"""
Folium-Based Shiny App for Texas Metro Population Hex Map
Multi-layer visualization with toggleable data views
"""

import math
import json
import time
import pandas as pd
import geopandas as gpd
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
HEX_RADIUS = 0.002

# Layer configurations
LAYER_CONFIGS = {
    'population_change': {
        'name': 'Population Change %',
        'column': 'population_change_pct',
        'colors': ['red', 'white', 'green'],  # Negative=red, 0=white, positive=green
        'format': '{:.2f}%',
        'diverging': True  # Centered at 0
    },
    'median_income': {
        'name': 'Median Income',
        'column': 'median_income',
        'colors': ['red', 'yellow', 'green'],  # Low=red, high=green
        'format': '${:,.0f}',
        'diverging': False
    },
    'reading_3rd': {
        'name': '3rd Grade Reading Low Performance',
        'column': 'reading_3rd_low_performance_median',
        'colors': ['green', 'yellow', 'red'],  # Low%=green (good), high%=red (bad)
        'format': '{:.1f}%',
        'diverging': False
    },
    'math_7th': {
        'name': '7th Grade Math Low Performance',
        'column': 'math_7th_low_performance_median',
        'colors': ['green', 'yellow', 'red'],  # Low%=green (good), high%=red (bad)
        'format': '{:.1f}%',
        'diverging': False
    }
}

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

def create_colormap_for_layer(df_filtered, layer_key):
    """Create appropriate colormap for a given layer"""
    if df_filtered is None or len(df_filtered) == 0:
        return cm.LinearColormap(['gray'], vmin=0, vmax=1)
    
    config = LAYER_CONFIGS[layer_key]
    column = config['column']
    
    if column not in df_filtered.columns:
        return cm.LinearColormap(['gray'], vmin=0, vmax=1)
    
    # Get valid (non-NaN) values
    valid_data = df_filtered[column].dropna()
    
    if len(valid_data) == 0:
        return cm.LinearColormap(['gray'], vmin=0, vmax=1)
    
    min_val = float(valid_data.min())
    max_val = float(valid_data.max())
    
    if min_val == max_val:
        return cm.LinearColormap(config['colors'], vmin=0, vmax=1)
    
    # Handle diverging colormaps (centered at 0)
    if config['diverging']:
        # For population change, center at 0
        abs_max = max(abs(min_val), abs(max_val))
        return cm.LinearColormap(
            config['colors'],
            vmin=-abs_max,
            vmax=abs_max
        )
    else:
        # Regular colormap
        return cm.LinearColormap(
            config['colors'],
            vmin=min_val,
            vmax=max_val
        )

def format_value(value, layer_key):
    """Format a value according to its layer configuration"""
    if pd.isna(value):
        return "N/A"
    
    config = LAYER_CONFIGS[layer_key]
    try:
        return config['format'].format(value)
    except:
        return str(value)

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
        print(f"Columns: {list(df.columns)}")
        
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
        import traceback
        traceback.print_exc()
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
    
    stats = {
        'total_hexes': len(df),
        'total_population': float(df['population'].sum()),
        'score_min': float(df['score'].min()),
        'score_max': float(df['score'].max()),
        'score_mean': float(df['score'].mean()),
        'center_lat': float(df['lat'].mean()),
        'center_lon': float(df['lng'].mean())
    }
    
    # Add layer-specific stats
    for layer_key, config in LAYER_CONFIGS.items():
        col = config['column']
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                stats[f'{layer_key}_min'] = float(valid_data.min())
                stats[f'{layer_key}_max'] = float(valid_data.max())
                stats[f'{layer_key}_mean'] = float(valid_data.mean())
    
    return stats

# -----------------------------
# UI Components
# -----------------------------
def create_summary_stats_html(summary, filtered_count=0, active_layer='population_change'):
    """Create summary stats HTML with layer information"""
    if summary['total_hexes'] == 0:
        return "<div style='padding:15px;color:red;'>No data available</div>"
    
    efficiency = f" • Showing {filtered_count:,} ({filtered_count/summary['total_hexes']*100:.1f}%)" if filtered_count else ""
    
    # Get active layer stats
    layer_config = LAYER_CONFIGS[active_layer]
    layer_stats = ""
    if f'{active_layer}_min' in summary:
        min_val = format_value(summary[f'{active_layer}_min'], active_layer)
        max_val = format_value(summary[f'{active_layer}_max'], active_layer)
        mean_val = format_value(summary[f'{active_layer}_mean'], active_layer)
        layer_stats = f"""
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #ddd;">
            <strong>Active Layer: {layer_config['name']}</strong><br>
            <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:5px;">
                <div><strong>Range:</strong> {min_val} - {max_val}</div>
                <div><strong>Mean:</strong> {mean_val}</div>
            </div>
        </div>
        """
    
    return f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;margin-bottom:15px;">
        <h4>Dataset Summary</h4>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
            <div><strong>Total Hexes:</strong> {summary['total_hexes']:,}</div>
            <div><strong>Total Population:</strong> {summary['total_population']:,.0f}</div>
        </div>
        {layer_stats}
        <div style="margin-top:10px;font-size:12px;color:#666;">
            Powered by Folium Hex Polygons{efficiency}
        </div>
    </div>
    """

def create_color_legend_html(summary, active_layer='population_change'):
    """Create color legend HTML for active layer"""
    if summary['total_hexes'] == 0:
        return ""
    
    if f'{active_layer}_min' not in summary:
        return ""
    
    layer_config = LAYER_CONFIGS[active_layer]
    min_val = format_value(summary[f'{active_layer}_min'], active_layer)
    max_val = format_value(summary[f'{active_layer}_max'], active_layer)
    
    # Create gradient based on layer colors
    colors = layer_config['colors']
    gradient = f"linear-gradient(to right, {', '.join(colors)})"
    
    return f"""
    <div style="background-color:#f8f9fa;padding:15px;border-radius:5px;">
        <h5>{layer_config['name']} Scale</h5>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:12px;">Low</span>
            <div style="width:200px;height:20px;background:{gradient};border:1px solid #ccc;"></div>
            <span style="font-size:12px;">High</span>
        </div>
        <div style="display:flex;justify-content:space-between;width:220px;font-size:11px;margin-top:5px;">
            <span>{min_val}</span>
            <span>{max_val}</span>
        </div>
    </div>
    """

# -----------------------------
# Map Creation
# -----------------------------
def create_folium_map(df_filtered, tract_data, center_lat, center_lng, zoom, 
                      show_hexes=True, show_tracts=False, active_layer='population_change'):
    """Create Folium map with hex polygons colored by active layer"""
    
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
                continue
    
    # Add hexes if requested and available
    if show_hexes and df_filtered is not None and len(df_filtered) > 0:
        # Create colormap for active layer
        colormap = create_colormap_for_layer(df_filtered, active_layer)
        layer_config = LAYER_CONFIGS[active_layer]
        layer_column = layer_config['column']
        
        # Add hex polygons
        for _, row in df_filtered.iterrows():
            try:
                # Get hex boundary
                if 'hex_id' in row and H3_AVAILABLE and pd.notna(row['hex_id']):
                    hex_boundary = get_h3_hex_boundary(row['hex_id'])
                else:
                    hex_boundary = hex_corners_fallback(row['lat'], row['lng'], radius=HEX_RADIUS)
                
                # Get color value for this hex
                layer_value = row.get(layer_column, None)
                if pd.notna(layer_value):
                    fill_color = colormap(layer_value)
                    fill_opacity = 0.7
                else:
                    fill_color = 'gray'
                    fill_opacity = 0.3
                
                # Build comprehensive popup
                popup_html = f"""
                <div style='font-family:Arial; min-width:200px;'>
                    <h4 style='margin-top:0;'>Hex Information</h4>
                    
                    <div style='margin-bottom:10px;'>
                        <strong>📊 Demographics</strong><br>
                        <b>Population:</b> {int(row.get('population', 0)):,}<br>
                        <b>Median Income:</b> ${int(row.get('median_income', 0)):,}<br>
                        <b>Status:</b> {row.get('target_status', 'Unknown')}
                    </div>
                    
                    <div style='margin-bottom:10px;'>
                        <strong>📈 Population Change</strong><br>
                        <b>2022→2023:</b> {format_value(row.get('population_change_pct', None), 'population_change')}
                    </div>
                """
                
                # Add school density info if available
                if 'schools_in_hex' in row:
                    popup_html += f"""
                    <div style='margin-bottom:10px;'>
                        <strong>🏫 Schools</strong><br>
                        <b>In This Hex:</b> {int(row.get('schools_in_hex', 0))}<br>
                        <b>In K1 Ring (7 hexes):</b> {int(row.get('schools_in_k1_ring', 0))}<br>
                        <b>Public Schools (K1):</b> {int(row.get('public_schools_k1', 0))}<br>
                        <b>Charter Schools (K1):</b> {int(row.get('charter_schools_k1', 0))}<br>
                        <b>IDEA Schools (K1):</b> {int(row.get('idea_schools_k1', 0))}
                    </div>
                    """
                
                # Add academic performance if available
                if 'reading_3rd_low_performance_median' in row or 'math_7th_low_performance_median' in row:
                    popup_html += f"""
                    <div style='margin-bottom:10px;'>
                        <strong>📚 Academic Performance</strong><br>
                        <span style='font-size:11px;color:#666;'>(Did Not Meet + Approaches)</span><br>
                        <b>3rd Reading:</b> {format_value(row.get('reading_3rd_low_performance_median', None), 'reading_3rd')}<br>
                        <b>7th Math:</b> {format_value(row.get('math_7th_low_performance_median', None), 'math_7th')}
                    </div>
                    """
                
                # Highlight active layer value
                popup_html += f"""
                    <div style='background-color:#f0f0f0;padding:8px;border-radius:3px;margin-top:10px;'>
                        <strong>🎯 Active Layer</strong><br>
                        <b>{layer_config['name']}:</b> {format_value(layer_value, active_layer)}
                    </div>
                </div>
                """
                
                folium.Polygon(
                    locations=hex_boundary,
                    color='white',
                    weight=0.5,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
                
            except Exception as e:
                print(f"Error adding hex: {e}")
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
        ui.h2("Texas Metro Population Hex Map - Multi-Layer Analysis", 
              style="text-align:center;color:#2c3e50;margin-bottom:20px"),
        
        # Status and controls
        ui.div(
            ui.output_ui("loading_status"),
            style="margin-bottom:15px"
        ),
        
        # Layer selection
        ui.div(
            ui.h4("Select Data Layer", style="margin-bottom:10px;"),
            ui.input_radio_buttons(
                "active_layer",
                "",
                choices={
                    'population_change': '📈 Population Change (2022→2023)',
                    'median_income': '💰 Median Household Income',
                    'reading_3rd': '📖 3rd Grade Reading Performance',
                    'math_7th': '🔢 7th Grade Math Performance'
                },
                selected='population_change'
            ),
            style="background-color:#f8f9fa;padding:15px;border-radius:5px;margin-bottom:20px;"
        ),
        
        # Map controls
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
        
        style="max-width:1400px;margin:0 auto;padding:20px"
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
        
        # Load hex data
        df_hex = load_hex_data_file('hex_data.feather')
        
        if df_hex is None:
            loading_message.set("❌ No hex data found")
            return
        
        # Load tract data
        df_tract = load_tract_data_file('tract_data.feather')
        
        # Store data
        hex_data.set(df_hex)
        tract_data.set(df_tract)
        
        if df_hex is not None and len(df_hex) > 0:
            summary = calculate_summary_stats(df_hex)
            summary_data.set(summary)
            
            datasets_loaded = [f"{len(df_hex):,} hexes"]
            if df_tract is not None: 
                datasets_loaded.append(f"{len(df_tract):,} tracts")
            
            loading_message.set(f"✅ Loaded: {', '.join(datasets_loaded)}")
        else:
            loading_message.set("❌ Failed to load hex data")
    
    # Refresh data when refresh button is clicked
    @reactive.Effect
    @reactive.event(input.refresh)
    def refresh_data():
        loading_message.set("Refreshing data...")
        
        df_hex = load_hex_data_file('hex_data.feather')
        df_tract = load_tract_data_file('tract_data.feather')
        
        if df_hex is not None and len(df_hex) > 0:
            hex_data.set(df_hex)
            tract_data.set(df_tract)
            summary = calculate_summary_stats(df_hex)
            summary_data.set(summary)
            loading_message.set(f"✅ Refreshed {len(df_hex):,} hexes")
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
        active_layer = input.active_layer()
        html_content = create_summary_stats_html(summary, count, active_layer)
        return ui.HTML(html_content)
    
    # Output: Color legend
    @output
    @render.ui
    def color_legend():
        if not input.show_stats():
            return ui.div()
        
        summary = summary_data.get()
        active_layer = input.active_layer()
        html_content = create_color_legend_html(summary, active_layer)
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
            active_layer = input.active_layer()
            
            if summary.get('total_hexes', 0) == 0:
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
                show_tracts=input.show_tracts(),
                active_layer=active_layer
            )
            
            # Return map HTML
            map_html = m._repr_html_()
            return ui.HTML(map_html)
            
        except Exception as e:
            error_msg = f"Error creating map: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return ui.div(
                ui.p(error_msg, style="color:red;text-align:center;"),
                style="padding:50px;border:1px solid #ddd;border-radius:5px;"
            )

# -----------------------------
# Create App
# -----------------------------
app = App(app_ui, server)

if __name__ == "__main__":
    print("🚀 Starting Texas Metro Population Hex Map - Multi-Layer Analysis...")
    print("📊 App ready! Run with: shiny run app.py")
