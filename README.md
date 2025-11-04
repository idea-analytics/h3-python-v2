# Texas Metro Population Hex Map

This project splits the original computationally intensive Shiny app into two parts:
1. **Data Processor** (`data_processor.py`) - Runs locally to generate hex data
2. **Visualization App** (`app.py`) - Lightweight Shiny app for Posit Connect

## Quick Start

### Step 1: Generate Data Locally

Run the data processor on your local machine:

```bash
python data_processor.py
```

This will:
- Load census tract and population data
- Filter to Texas metro areas (8 major metros)
- Generate H3 hexagonal grid at resolution 6
- Distribute population to hexes
- Calculate density scores
- Export `hex_data.parquet` and `hex_summary.json`

**Requirements for local processing:**
- pandas
- geopandas
- h3
- numpy
- shapely
- pyarrow (for Parquet export)
- Your existing `load_data.py` and `Population-2023-filtered.csv`

### Step 2: Deploy to Posit Connect

Upload these files to Posit Connect:
- `app.py` (the visualization app)
- `hex_data.parquet` (generated from Step 1)
- `hex_summary.json` (generated from Step 1)

**Requirements for Posit Connect:**
- shiny
- plotly
- pandas
- pyarrow

## Features

The visualization app provides:
- **Interactive Map**: Zoom, pan, and explore Texas metro areas
- **Color-Coded Hexes**: Population density gradient from blue (low) to red (high)
- **Hover Tooltips**: Shows hex ID, population, and density score
- **Summary Statistics**: Dataset overview with total population and score ranges
- **Color Legend**: Visual reference for density scale

## Metro Areas Covered

- Austin
- Dallas
- Houston
- San Antonio
- El Paso
- Corpus Christi
- Lubbock
- Midland-Odessa

Each metro area includes a ~40-mile radius around the city center.

## Technical Details

- **H3 Resolution 6**: Each hex covers approximately 36 km²
- **Population Distribution**: Census tract populations distributed to hexes based on area intersection
- **Density Score**: Population per square meter
- **Coordinate System**: WGS84 (EPSG:4326)

## File Structure

```
project/
├── data_processor.py         # Local data processing
├── app.py                   # Shiny visualization app
├── hex_data.parquet         # Generated hex data (upload to Connect)
├── hex_summary.json         # Generated summary stats (upload to Connect)
├── load_data.py             # Your existing data loader
├── Population-2023-filtered.csv  # Your census data
└── README.md               # This file
```

## Troubleshooting

1. **"hex_data.parquet not found"**: Make sure you've run `data_processor.py` and uploaded both Parquet and JSON files to Posit Connect

2. **Import errors in data_processor.py**: Ensure all required packages are installed locally (including `pyarrow`)

3. **Empty map**: Check that hex_data.parquet contains valid data and hex_summary.json has proper statistics

4. **Performance issues**: The Parquet format should load much faster than JSON. If still slow, consider reducing hex resolution

## Benefits of Parquet Format

- **File size**: 5-10x smaller than JSON
- **Loading speed**: Much faster read times
- **Compression**: Better data compression
- **Efficiency**: Optimized for large datasets

## Refreshing Data

To update the visualization with new data:
1. Run `data_processor.py` locally with updated source data
2. Upload the new `hex_data.parquet` and `hex_summary.json` to Posit Connect
3. Click "Refresh Map" in the app interface
