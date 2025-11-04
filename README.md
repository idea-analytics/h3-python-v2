# Texas Metro Population Hex Map - Setup Guide

## Overview
This project creates an interactive hex map visualization of population data for Texas metro areas using Shiny for Python and Folium.

## Key Fixes Applied

### 1. Data Processing Script (`data_processor_fixed.py`)
- **H3 API Compatibility**: Updated to work with both H3 v3 and v4+ APIs
- **Error Handling**: Added comprehensive error handling and fallback options
- **Sample Data**: Creates sample data when real census data is unavailable
- **Improved Logging**: Better debug output to track processing steps
- **CRS Handling**: Improved coordinate reference system management

### 2. Shiny App (`shiny_app_fixed.py`)
- **Reactive System**: Fixed reactive dependencies and event handling
- **Error Resilience**: App continues to work even when data files are missing
- **Sample Data Fallback**: Automatically creates sample data for testing
- **Performance**: Added viewport filtering and fast mode for large datasets
- **UI Improvements**: Better layout and status indicators

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Alternative Installation (if requirements.txt fails)
```bash
pip install pandas geopandas h3 shapely folium branca shiny numpy pyarrow
```

## Usage

### Option A: With Real Census Data

1. **Prepare Your Data**:
   - Place `Population-2023-filtered.csv` in the project directory
   - Ensure you have a `load_data.py` module with `load_texas_tracts()` function
   - Or modify the script to load your tract data differently

2. **Generate Hex Data**:
   ```bash
   python data_processor_fixed.py
   ```
   This creates:
   - `hex_data.feather` - Main hex data file
   - `hex_summary.json` - Summary statistics

3. **Run the Shiny App**:
   ```bash
   shiny run shiny_app_fixed.py
   ```

### Option B: With Sample Data (for Testing)

1. **Run the Shiny App Directly**:
   ```bash
   shiny run shiny_app_fixed.py
   ```
   The app will automatically create sample data if `hex_data.feather` is not found.

## Features

### Data Processing
- ✅ H3 hexagonal grid generation at resolution 6
- ✅ Population distribution from census tracts to hexes
- ✅ Density score calculation
- ✅ Metro area filtering (40-mile radius around major Texas cities)
- ✅ Robust error handling and logging

### Visualization App
- ✅ Interactive Folium map with hex polygons
- ✅ Population-based color coding (green → yellow → red)
- ✅ Viewport-based filtering for performance
- ✅ Fast mode for large datasets
- ✅ Real-time statistics display
- ✅ Zoom-based hex filtering
- ✅ Popup information for each hex

### Performance Optimizations
- Viewport filtering: Only renders hexes visible in current map view
- Fast mode: Samples large datasets for quicker rendering
- Spatial indexing: Efficient intersection calculations
- Feather format: Fast data loading

## Troubleshooting

### Common Issues and Solutions

1. **"No module named 'load_data'"**
   - The script will create sample tract data automatically
   - Or provide your own tract loading function

2. **"H3 function not found"**
   - The script handles both H3 v3 and v4+ APIs automatically
   - Update H3: `pip install --upgrade h3`

3. **"No hex data file found"**
   - The app creates sample data automatically
   - Or run the data processor first

4. **"Empty map display"**
   - Check that hex_data.feather contains data with population > 0
   - Verify coordinate ranges are reasonable (Texas area)

5. **"Slow map rendering"**
   - Enable "Fast Loading Mode" checkbox
   - Increase zoom level to reduce visible area
   - Check that viewport filtering is working

### Data Requirements

#### For Real Data Processing:
- `Population-2023-filtered.csv` with columns:
  - `Geography`: Census tract identifiers
  - `Estimate!!Total:`: Population estimates
- Census tract geometries (via `load_data.py` or similar)

#### Data Format Expected:
```
Geography,Estimate!!Total:
1400000US48001020100,1250
1400000US48001020200,890
...
```

## File Structure
```
project/
├── data_processor_fixed.py     # Data processing script
├── shiny_app_fixed.py         # Shiny visualization app
├── requirements.txt           # Python dependencies
├── Population-2023-filtered.csv  # Census data (optional)
├── load_data.py              # Tract loading module (optional)
├── hex_data.feather          # Generated hex data
└── hex_summary.json          # Generated summary stats
```

## Configuration

### Metro Areas (can be modified in data_processor_fixed.py):
- El Paso, Austin, Dallas, Houston
- Midland-Odessa, Lubbock, Corpus Christi, San Antonio

### Adjustable Parameters:
- `RADIUS_DEG`: Metro area radius (default: 0.64° ≈ 40 miles)
- `DEFAULT_ZOOM`: Initial map zoom level
- `HEX_RADIUS`: Visual hex size on map
- H3 resolution (default: 6)

## Deployment

### Local Development:
```bash
shiny run shiny_app_fixed.py --reload
```

### Production Deployment:
- Use a production WSGI server
- Configure proper logging
- Set up data backup/refresh schedules
- Consider caching for large datasets

## Advanced Usage

### Custom Data Sources:
Modify `load_tracts()` in the data processor to load from:
- Shapefile: `geopandas.read_file('tracts.shp')`
- GeoJSON: `geopandas.read_file('tracts.geojson')`
- PostGIS: `geopandas.read_postgis(sql, connection)`

### Performance Tuning:
- Adjust H3 resolution (higher = more detail, slower)
- Modify viewport buffer factor
- Tune fast mode sampling rate
- Use different map tiles for better performance

### Custom Styling:
- Modify color schemes in `create_folium_map()`
- Adjust hex popup content
- Change base map tiles
- Add custom legends or controls

## Support

If you encounter issues:
1. Check the console output for detailed error messages
2. Verify all dependencies are installed correctly
3. Test with sample data first
4. Check file permissions and paths
5. Ensure adequate memory for large datasets

The fixed scripts include comprehensive error handling and will provide detailed feedback about any issues encountered.
