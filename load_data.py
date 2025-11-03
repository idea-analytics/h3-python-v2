import os
import urllib.request
import zipfile
import geopandas as gpd

def download_texas_tracts(force_download=False):
    """
    Download Texas census tract shapefiles from Census Bureau if not already present.
    
    Args:
        force_download: If True, download even if files exist
    
    Returns:
        Path to the shapefile
    """
    
    shapefile_dir = 'tl_2023_48_tract'
    shapefile_path = os.path.join(shapefile_dir, 'tl_2023_48_tract.shp')
    
    # Check if shapefile already exists
    if os.path.exists(shapefile_path) and not force_download:
        print(f"Shapefile already exists at {shapefile_path}")
        return shapefile_path
    
    # Create directory if it doesn't exist
    os.makedirs(shapefile_dir, exist_ok=True)
    
    # Census Bureau URL for Texas (FIPS code 48) census tracts
    url = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_48_tract.zip"
    zip_path = "tl_2023_48_tract.zip"
    
    print(f"Downloading Texas census tracts from {url}...")
    
    try:
        # Download the zip file
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
        
        # Extract the zip file
        print(f"Extracting to {shapefile_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(shapefile_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print("Extraction complete!")
        
        return shapefile_path
        
    except Exception as e:
        print(f"Error downloading shapefile: {e}")
        raise

def load_texas_tracts():
    """
    Load Texas census tracts, downloading if necessary.
    
    Returns:
        GeoDataFrame of Texas census tracts
    """
    shapefile_path = download_texas_tracts()
    print(f"Loading shapefile from {shapefile_path}...")
    tracts_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(tracts_gdf)} census tracts")
    return tracts_gdf

if __name__ == "__main__":
    # Test the download
    tracts = load_texas_tracts()
    print(f"\nShapefile info:")
    print(f"Number of tracts: {len(tracts)}")
    print(f"CRS: {tracts.crs}")
    print(f"Columns: {list(tracts.columns)}")
