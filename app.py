"""
Lightweight Shiny App for Texas Metro Population Hex Map Visualization
This app loads pre-processed hex data and displays it interactively
"""

import json
import pandas as pd
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# Data loading functions
# -----------------------------
def load_hex_data(feather_file='hex_data.feather', summary_file='hex_summary.json'):
    """Load pre-processed hex data from Feather and summary from JSON with caching"""
    try:
        # Load hex data from Feather
        print(f"Loading hex data from {feather_file}...")
        df = pd.read_feather(feather_file)
        print(f"Loaded {len(df)} hexes from Feather file")
        
        # Load summary from JSON
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {summary_file} not found, calculating summary from data")
            # Calculate summary from the data if JSON not available
            summary = {
                'total_hexes': len(df),
                'total_population': float(df['population'].sum()),
                'score_min': float(df['score'].min()),
                'score_max': float(df['score'].max()),
                'score_mean': float(df['score'].mean()),
                'center_lat': float(df['center_lat'].mean()),
                'center_lon': float(df['center_lon'].mean())
            }
        
        # Convert DataFrame to the format expected by the visualization
        # Only include hexes with population > 0 for better performance
        df_filtered = df[df['population'] > 0].copy()
        print(f"Filtered to {len(df_filtered)} hexes with population > 0")
        
        hexes_list = []
        for idx, row in df_filtered.iterrows():
            hexes_list.append({
                'hex_id': row['hex_id'],
                'population': row['population'],
                'score': row['score'],
                'hex_area_m2': row['hex_area_m2'],
                'coordinates': {
                    'lons': row['lons'],
                    'lats': row['lats']
                }
            })
        
        # Update summary with filtered data
        summary['total_hexes'] = len(hexes_list)
        
        return {
            'summary': summary,
            'hexes': hexes_list
        }
        
    except FileNotFoundError:
        print(f"Error: {feather_file} not found. Please run the data processor first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# -----------------------------
# Color mapping functions
# -----------------------------
def get_color_from_score(score, min_score, max_score):
    """Generate RGB color based on score value"""
    if max_score <= min_score:
        return 'rgb(128,128,128)'  # Gray for uniform values
    
    # Normalize score to 0-1 range
    norm = (score - min_score) / (max_score - min_score)
    
    # Color gradient: blue (low) to red (high)
    red = int(255 * norm)
    green = int(255 * (1 - norm) * 0.5)
    blue = int(255 * (1 - norm))
    
    return f'rgb({red},{green},{blue})'

# -----------------------------
# Plotly map creation
# -----------------------------
def create_interactive_map(hex_data):
    """Create interactive Plotly map from hex data using reliable individual traces"""
    if not hex_data:
        return go.Figure().add_annotation(
            text="No data available. Please upload hex_data.feather",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    summary = hex_data['summary']
    hexes = hex_data['hexes']
    
    min_score = summary['score_min']
    max_score = summary['score_max']
    
    # Create individual traces but with optimized settings
    for i, hex_data_point in enumerate(hexes):
        coords = hex_data_point['coordinates']
        lons = coords['lons']
        lats = coords['lats']
        
        score = hex_data_point['score']
        population = hex_data_point['population']
        hex_id = hex_data_point['hex_id']
        
        # Normalize score for color mapping
        if max_score > min_score:
            norm = (score - min_score) / (max_score - min_score)
        else:
            norm = 0.5
        
        # Generate color
        red = int(255 * norm)
        green = int(255 * (1 - norm) * 0.5)
        blue = int(255 * (1 - norm))
        color = f'rgba({red},{green},{blue},0.7)'
        
        # Add trace for this hex
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(width=0.3, color='rgba(255,255,255,0.6)'),
            hovertemplate=(
                f"<b>Hex:</b> {hex_id}<br>"
                f"<b>Population:</b> {int(population):,}<br>"
                f"<b>Density:</b> {score:.4f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
            name=""  # Empty name for cleaner legend
        ))
        
        # Limit number of traces for very large datasets
        if i >= 2000:  # Max 2000 hexes for performance
            print(f"Limiting display to first 2000 hexes out of {len(hexes)} for performance")
            break
    
    # Add color legend as separate element
    if len(hexes) > 0:
        # Create invisible scatter for colorbar
        legend_values = [0, 0.25, 0.5, 0.75, 1.0]
        fig.add_trace(go.Scattermapbox(
            lon=[summary['center_lon']] * len(legend_values),
            lat=[summary['center_lat']] * len(legend_values),
            mode='markers',
            marker=dict(
                size=0,  # Invisible
                color=legend_values,
                colorscale='RdYlBu_r',
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Population Density",
                        side="right"
                    ),
                    tickmode="array",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'],
                    len=0.7,
                    thickness=15
                )
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Set map center and zoom
    center_lat = summary['center_lat']
    center_lon = summary['center_lon']
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=6
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        showlegend=False
    )
    
    return fig

# -----------------------------
# Statistics functions
# -----------------------------
def create_summary_stats(hex_data):
    """Create summary statistics display"""
    if not hex_data:
        return "No data available"
    
    summary = hex_data['summary']
    
    stats_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <h4>Dataset Summary</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div><strong>Total Hexes:</strong> {summary['total_hexes']:,}</div>
            <div><strong>Total Population:</strong> {summary['total_population']:,.0f}</div>
            <div><strong>Score Range:</strong> {summary['score_min']:.6f} - {summary['score_max']:.6f}</div>
            <div><strong>Mean Score:</strong> {summary['score_mean']:.6f}</div>
        </div>
    </div>
    """
    return stats_html

def create_color_legend(hex_data):
    """Create a color legend for the map"""
    if not hex_data:
        return ""
    
    summary = hex_data['summary']
    min_score = summary['score_min']
    max_score = summary['score_max']
    
    # Create color gradient
    legend_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
        <h5>Population Density Scale</h5>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 12px;">Low</span>
            <div style="width: 200px; height: 20px; background: linear-gradient(to right, 
                rgb(0,128,255), rgb(128,64,128), rgb(255,0,0)); border: 1px solid #ccc;"></div>
            <span style="font-size: 12px;">High</span>
        </div>
        <div style="display: flex; justify-content: space-between; width: 220px; font-size: 11px; margin-top: 5px;">
            <span>{min_score:.6f}</span>
            <span>{max_score:.6f}</span>
        </div>
    </div>
    """
    return legend_html

# -----------------------------
# Shiny UI
# -----------------------------
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("Texas Metro Population Hex Map")
    ),
    
    ui.div(
        ui.h2("Texas Metro Population Hex Map", 
              style="text-align: center; color: #2c3e50; margin-bottom: 20px;"),
        
        ui.div(
            ui.output_ui("summary_stats"),
            style="margin-bottom: 15px;"
        ),
        
        ui.div(
            ui.output_ui("color_legend"),
            style="margin-bottom: 15px;"
        ),
        
        ui.div(
            ui.row(
                ui.column(6,
                    ui.input_action_button(
                        "refresh", 
                        "Refresh Map", 
                        class_="btn-primary",
                        style="margin-bottom: 15px; margin-right: 10px;"
                    )
                ),
                ui.column(6,
                    ui.input_checkbox(
                        "fast_mode",
                        "Fast Loading Mode (fewer hexes)",
                        value=False
                    )
                )
            ),
            ui.output_ui("map_plot")
        ),
        
        ui.div(
            ui.p([
                "Data shows population density across Texas metro areas using H3 hexagonal grid. ",
                "Hover over hexes to see population and density values. ",
                "Use mouse to zoom and pan the map. ",
                "Enable Fast Loading Mode for better performance with large datasets."
            ], style="font-size: 12px; color: #6c757d; text-align: center; margin-top: 15px;")
        ),
        
        style="max-width: 1200px; margin: 0 auto; padding: 20px;"
    )
)

# -----------------------------
# Shiny Server
# -----------------------------
def server(input, output, session):
    
    # Load data on startup
    hex_data = reactive.Value(None)
    
    @reactive.Effect
    def load_data():
        """Load hex data when app starts"""
        data = load_hex_data('hex_data.feather', 'hex_summary.json')
        hex_data.set(data)
        if data:
            print(f"Loaded data with {data['summary']['total_hexes']} hexes")
        else:
            print("Failed to load hex data")
    
    @output
    @render.ui
    def summary_stats():
        """Render summary statistics"""
        data = hex_data.get()
        return ui.HTML(create_summary_stats(data))
    
    @output
    @render.ui
    def color_legend():
        """Render color legend"""
        data = hex_data.get()
        return ui.HTML(create_color_legend(data))
    
    @output
    @render.ui
    def map_plot():
        """Render the interactive map"""
        data = hex_data.get()
        
        if not data:
            return ui.div(
                ui.h4("No Data Available", style="text-align: center; color: #dc3545;"),
                ui.p("Please ensure 'hex_data.feather' and 'hex_summary.json' are in the app directory.", 
                     style="text-align: center; color: #6c757d;"),
                style="padding: 50px; text-align: center;"
            )
        
        try:
            # Apply fast mode if enabled
            if input.fast_mode():
                # Sample every 3rd hex for faster loading
                hexes = data['hexes'][::3]
                fast_data = {
                    'summary': data['summary'],
                    'hexes': hexes
                }
                print(f"Fast mode: Rendering {len(hexes)} of {len(data['hexes'])} hexes")
                fig = create_interactive_map(fast_data)
            else:
                print(f"Full mode: Rendering {len(data['hexes'])} hexes")
                fig = create_interactive_map(data)
            
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        except Exception as e:
            print(f"Error creating map: {e}")
            return ui.div(
                ui.h4("Error Creating Map", style="text-align: center; color: #dc3545;"),
                ui.p(f"Error: {str(e)}", style="text-align: center; color: #6c757d;"),
                style="padding: 50px; text-align: center;"
            )
    
    @reactive.Effect
    @reactive.event(input.refresh)
    def refresh_data():
        """Refresh data when button is clicked"""
        print("Refreshing data...")
        data = load_hex_data('hex_data.feather', 'hex_summary.json')
        hex_data.set(data)
    
    @reactive.Effect
    @reactive.event(input.fast_mode)
    def toggle_fast_mode():
        """Re-render map when fast mode is toggled"""
        mode = "fast" if input.fast_mode() else "full"
        print(f"Switching to {mode} mode...")

# -----------------------------
# Create Shiny App
# -----------------------------
app = App(app_ui, server)

# -----------------------------
# Instructions for deployment
# -----------------------------
"""
DEPLOYMENT INSTRUCTIONS:

1. Run data_processor.py locally to generate hex_data.feather:
   python data_processor.py

2. Upload both files to your Posit Connect deployment:
   - app.py (this file)
   - hex_data.feather (generated by data_processor.py)
   - hex_summary.json (generated by data_processor.py)

3. Required packages for Posit Connect:
   - shiny
   - plotly
   - pandas
   - pyarrow (for reading Feather files)

4. The app will automatically load hex_data.feather on startup
   and display the interactive map with hover tooltips.

5. Users can zoom, pan, and hover to explore the data.

Benefits of Feather over JSON and Parquet:
- Fastest loading times (optimized for speed)
- Efficient with coordinate arrays
- Pandas native format
- Good compression with fast decompression
- Perfect for this use case with lots of arrays
"""
