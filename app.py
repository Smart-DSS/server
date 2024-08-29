from flask import Response, Flask, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import psycopg2
import os
import json
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import time
import io
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt



# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

cwd = os.getcwd()

# Setup Flask-SocketIO with Eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Get env variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
DATABASE_URL = os.environ.get('DATABASE_URL')

# Parse the DATABASE_URL to extract components
result = urlparse(DATABASE_URL)
username = result.username
password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port
query = parse_qs(result.query)

# Connect to PostgreSQL using parsed components
try:
    conn = psycopg2.connect(
        dbname=database,
        user=username,
        password=password,
        host=hostname,
        port=port,
        options=f"-c search_path={query.get('schema', ['public'])[0]}"
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    print("Connected to PostgreSQL database!")
except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL database: {e}")

# Fetch the latest camera_count for each unique camera_id
def fetch_latest_data():
    cursor.execute("""
        SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
        FROM real_time_data
        ORDER BY camera_id, timestamp DESC
    """)
    rows = cursor.fetchall()

    # Convert datetime objects to strings
    data = []
    for row in rows:
        camera_id, count_camera, timestamp = row
        data.append({
            "camera_id": camera_id,
            "count_camera": count_camera,
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
        })
    
    return data

@app.route("/test")
def test():
    return Response('Server is working!', status=201, mimetype='application/json')

@app.route("/db-test")
def db_test():
    try:
        cursor.execute("""
            SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
            FROM real_time_data
            ORDER BY camera_id, timestamp DESC
        """)
        rows = cursor.fetchall()
        
        # Convert datetime objects to strings
        data = []
        for row in rows:
            camera_id, count_camera, timestamp = row
            data.append({
                "camera_id": camera_id,
                "count_camera": count_camera,
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
            })

        print(f"Fetched {len(rows)} rows from real_time_data")
        return jsonify({"message": "Database connection successful", "data": data})
    except psycopg2.Error as e:
        print(f"Database query failed: {e}")
        return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# New route to generate and return a plot
# @app.route("/plot")
# def plot():
#     # Example plot
#     plt.figure(figsize=(10, 6))
#     plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
#     plt.title("Sample Plot")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
    
#     # Save the plot to a BytesIO object
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()  # Close the figure to free memory

#     return send_file(img, mimetype='image/png')




# -----


import numpy as np
import pandas as pd
import geopandas as gpd

from sqlalchemy import create_engine

from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from shapely.geometry import box
from shapely import wkt
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import matplotlib

from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display, clear_output
import time
import folium
from folium.plugins import HeatMap
from folium.features import DivIcon
import contextily as ctx


#from shapely.geometry import Point,wkt
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine
import numpy as np
from esda.getisord import G_Local
from libpysal.weights import KNN
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from sqlalchemy import create_engine
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.neighbors import KernelDensity
from shapely.geometry import polygon

import numpy as np
import pandas as pd
import geopandas as gpd

from sqlalchemy import create_engine

from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from shapely.geometry import box
from shapely import wkt
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import matplotlib

from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display, clear_output

import time


import folium
from folium.plugins import HeatMap


#from shapely.geometry import Point,wkt
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine
import numpy as np
from esda.getisord import G_Local
from libpysal.weights import KNN
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from sqlalchemy import create_engine
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon
from datetime import datetime, timedelta

def fetch_people_locations(conn, start_time, end_time, epsg=32643):
    """
    Fetches people's locations from the database and returns a GeoDataFrame.
    Args:
        engine: The database connection engine.
    Returns:
        gdf: A GeoDataFrame containing the fetched data with geometries.
    """
    # SQL query to fetch data from the specified table
    #query = "SELECT id, person_id, timestamp, ST_AsText(geom) as geometry FROM people_sim_wgs;" #from my local server
    query = f"""
    SELECT person_id, camera_id, timestamp, ST_AsText(geom) AS geometry
    FROM real_time_data
    WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
    ;"""

    # Execute the SQL query and load the result into a DataFrame
    # df = pd.read_sql(query, engine)  #before
    df = pd.read_sql_query(query, conn)  #after
    
    # Convert the 'timestamp' column to datetime format
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M")
    
    # Convert the WKT format geometries to Shapely geometries
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Convert the CRS to UTM zone 43N (EPSG:32643) or the sepcified crs for more accurate distance measurements
    gdf = gdf.to_crs(epsg=32643)

    gdf = gdf.dropna(subset=['geometry'])
    
    return gdf

# Load shapefiles
def load_shapefiles(street_shapefile, entry_exit_shapefile):
    """
    Loads the street and entry/exit shapefiles using GeoPandas.
    Returns the loaded GeoDataFrames.
    """
    street = gpd.read_file(street_shapefile)
    street= street.to_crs(epsg=32643)
    entry_exit = gpd.read_file(entry_exit_shapefile)
    entry_exit= entry_exit.to_crs(epsg=32643)  # Update with appropriate UTM zone for your area
    return street, entry_exit

def detect_clogging(entry_exit_gdf, people_gdf, distance_threshold, count_threshold):
    """
    Detects clogging at the entry and exit points of the street.
    Returns a GeoDataFrame of points where clogging is detected.
    """
    results = []
   
    for entry_exit in entry_exit_gdf.itertuples():
        point = entry_exit.geometry
        clogging_df = people_gdf[people_gdf.geometry.distance(point) < distance_threshold]
       
        if len(clogging_df) > count_threshold:
            results.append(point)
   
    # Convert results to GeoDataFrame
    results_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(results), crs=entry_exit_gdf.crs)
   
    return results_gdf


def detect_clustering(people_gdf, time_window, min_cluster_size):
    """
    Detects clustering of people using Hotspot/Coldspot analysis.
    Returns hotspot and coldspot points.
    """
    people_gdf['timestamp'] = pd.to_datetime(people_gdf['timestamp'])
   
    # Create a separate DataFrame for the rolling window
    df = people_gdf.set_index('timestamp')
    df.sort_index(inplace=True)

    print("DF for clustering:", df.head())  # Debugging

    hotspot_results = []
    coldspot_results = []

    # Rolling window for time_window minutes
    rolling_groups = [group for _, group in df.groupby(pd.Grouper(freq=f'{time_window}min'))]
   
    def analyze_group(group):
        n_points = len(group)
        k = min(min_cluster_size, n_points - 1)  # Ensure k is within bounds
        print("n_points:", n_points, "k=", k, "group size:", len(group))  # Debugging

        if k < 1:  # Not enough points to perform KNN
            return [], []

        # Drop duplicates if any exist
        group = group.drop_duplicates(subset='geometry')

        try:
            # Ensure the geometry column is not empty and is in the correct format
            if group.empty or group['geometry'].isnull().any():
                print("Empty or invalid geometries detected.")
                return [], []
           
            # Prepare the data for KNN
            coords = np.array([[geom.x, geom.y] for geom in group['geometry']])
           
            if len(coords) <= k:
                print("Not enough distinct points for the given k.")
                return [], []

            w = KNN.from_array(coords, k=k)

            # Check if the weights matrix is fully connected
            if w.n_components > 1:
                print(f"Warning: The weights matrix is not fully connected. {w.n_components} disconnected components.")
                # You can decide what to do with the disconnected components here

            g = G_Local(group['geometry'].apply(lambda geom: geom.x), w)
            hotspots = group[g.Zs > 1.96]['geometry'].tolist()  # Assuming 95% confidence interval for hotspots
            coldspots = group[g.Zs < -1.96]['geometry'].tolist()  # Assuming 95% confidence interval for coldspots
           
            return hotspots, coldspots
        except Exception as e:
            print(f"Error in KNN calculation: {e}")
            return [], []

    for group in rolling_groups:
        hotspots, coldspots = analyze_group(group)
        hotspot_results.extend(hotspots)
        coldspot_results.extend(coldspots)

    print("Hotspot Results:", hotspot_results)  # Debugging
    print("Coldspot Results:", coldspot_results)  # Debugging

    # Ensure that we return results even if they are empty
    return hotspot_results, coldspot_results, None  # Adjust None to the third return type as needed


def calculate_average_speeds(gdf,speed_unit="km/h"):
    """
    Calculates the average and instantaneous speed for each person in the GeoDataFrame.
    Args:
        gdf: The GeoDataFrame with points and timestamps.
    Returns:
        gdf: The GeoDataFrame with added speed information.
    """
    # Ensure the data is sorted by person_id and timestamp
    gdf = gdf.sort_values(by=['person_id', 'timestamp']).reset_index(drop=True)
   
    # Calculate the time difference in seconds
    gdf['time_diff'] = gdf.groupby('person_id')['timestamp'].diff().dt.total_seconds()
   
    # Debug: Check time_diff calculation
    #print("Time Diff Head:\n", gdf[['person_id', 'timestamp', 'time_diff']].head())
   
    # Calculate the distance between consecutive points using apply and shift
    gdf['distance'] = gdf.groupby('person_id')['geometry'].apply(lambda x: x.distance(x.shift())).reset_index(level=0, drop=True)
   
    # Debug: Check distance calculation
    #print("Distance Head:\n", gdf[['person_id', 'geometry', 'distance']].head())
   
    # Calculate cumulative distance and time
    gdf['cumulative_distance'] = gdf.groupby('person_id')['distance'].cumsum()
    gdf['cumulative_time'] = gdf.groupby('person_id')['time_diff'].cumsum()
   
    # Calculate the average speed (cumulative distance / cumulative time)
    # Calculate speed in kilo meters per hour
    if speed_unit=="km/h":
        gdf['average_speed'] = (gdf['cumulative_distance']*3.6) / (gdf['cumulative_time'])
    else:
        gdf['average_speed'] = (gdf['cumulative_distance']) / (gdf['cumulative_time'])
       
    # Debug: Check cumulative calculations
    #print("Cumulative Distance and Time Head:\n", gdf[['person_id', 'cumulative_distance', 'cumulative_time', 'average_speed']].head())
   
    # Calculate instantaneous speed (distance / time_diff)
    if speed_unit=="km/h":
        gdf['instantaneous_speed'] = (gdf['distance']*3.6) / (gdf['time_diff'])
    else:
        gdf['instantaneous_speed'] = (gdf['distance']) / (gdf['time_diff'])
   
   
    # Debug: Check instantaneous speed calculation
    #print("Instantaneous Speed Head:\n", gdf[['person_id', 'distance', 'time_diff', 'instantaneous_speed']].head())
   
    # Drop rows where time_diff or distance is NaN or zero (if necessary)
    gdf = gdf.dropna(subset=['time_diff', 'distance', 'instantaneous_speed'])
    gdf = gdf[(gdf['time_diff'] > 0)] # & (gdf['distance'] > 0)]
   
    # Debug: Final DataFrame check
    #print("Average speeds calculated:\n", gdf.head())
   
    return gdf



# Very big code starts here




def detect_running_away_anywhere(gdf, speed_threshold, distance_threshold, cluster_eps, min_samples):
    """
    Detects if people are suddenly running away from any point on the street in all directions.
   
    Args:
        people_gdf (GeoDataFrame): GeoDataFrame containing people's locations, instantaneous speeds, and timestamps.
        speed_threshold (float): The minimum speed to consider as "running."
        distance_threshold (float): The minimum distance to consider as "running away."
        cluster_eps (float): The maximum distance between two points to be considered in the same cluster (for DBSCAN).
        min_samples (int): The minimum number of people in a cluster to consider it a potential POI.
   
    Returns:
        bool: True if a running away event is detected, otherwise False.
        list: List of individuals' geometries who are detected to be running away.
        Point: The centroid of the event, if detected. Returns None if no event is detected.
    """
    results = []
    event_centroid = None
   
    # Filter people who are moving faster than the speed threshold
    running_people = gdf[gdf['instantaneous_speed'] > speed_threshold]
    if running_people.empty:
        return False, results, event_centroid

    # Extract coordinates for clustering
    coords = np.array(list(running_people.geometry.apply(lambda geom: (geom.x, geom.y))))
   
    # Apply DBSCAN clustering to find clusters of people
    db = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
   
    # Iterate over each cluster found
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # -1 is noise in DBSCAN, ignore it
            continue

        # Extract the cluster
        cluster = running_people[labels == label]
        cluster_coords = np.array(list(cluster.geometry.apply(lambda geom: (geom.x, geom.y))))
       
        # Calculate centroid of the cluster
        centroid = Point(cluster_coords.mean(axis=0))

        # Check if people are moving away from the centroid
        divergence_detected = False
        directions = []
       
        for person in cluster.itertuples():
            distance_from_centroid = person.geometry.distance(centroid)
            if distance_from_centroid > distance_threshold:
                # Calculate the direction of movement relative to the centroid
                movement_direction = np.arctan2(
                    person.geometry.y - centroid.y,
                    person.geometry.x - centroid.x
                )
                directions.append(movement_direction)
       
        # Check if there is sufficient angular dispersion
        if len(directions) >= min_samples:  # Need at least three people moving in different directions to consider dispersion
            min_direction, max_direction = min(directions), max(directions)
            if max_direction - min_direction > np.pi / 2:  # Adjust this threshold as needed
                divergence_detected = True
       
        if divergence_detected:
            results.extend(cluster.geometry)
            event_centroid = centroid
   
    return len(results) > 0, results, event_centroid


# Example usage:
# detected, people_geometries, event_centroid = detect_running_away_anywhere(people_gdf, speed_threshold=5.0, distance_threshold=10.0, cluster_eps=5.0, min_samples=3)
# if detected:
#     print("Running away event detected at centroid:", event_centroid)
#     print("People involved:", people_geometries)

def visualize_running_away_event_with_directions(street_gdf, people_gdf, running_geometries, event_centroid,show_arrow=True):
    """
    Visualizes the detected running away event on a map with direction arrows.

    Args:
        street_gdf (GeoDataFrame): GeoDataFrame containing the street data.
        people_gdf (GeoDataFrame): GeoDataFrame containing people's locations.
        running_geometries (list): List of geometries of people detected to be running away.
        event_centroid (Point): The centroid of the detected running away event.
    """
    # Base plot with street
    fig, ax = plt.subplots(figsize=(12, 8))
    street_gdf.plot(ax=ax, color='lightgrey', edgecolor='black')

    # Plot all people
    people_gdf.plot(ax=ax, color='blue', alpha=0.6, markersize=10, label="People")

    # Plot running people with direction arrows
    if running_geometries:
        running_gdf = gpd.GeoDataFrame(geometry=running_geometries)
        #running_gdf.plot(ax=ax, color='red', markersize=50, label="Running People")
       
        if show_arrow:
            for geom in running_geometries:
                # Calculate direction vector (from event centroid to running person)
                dx = geom.x - event_centroid.x
                dy = geom.y - event_centroid.y
                ax.arrow(event_centroid.x, event_centroid.y, dx, dy, head_width=2, head_length=2, fc='green', ec='green')
       
        running_gdf.plot(ax=ax, color='red', markersize=10, label="Running People")
   
    # Plot the centroid of the event
    if event_centroid:
        gpd.GeoDataFrame(geometry=[event_centroid]).plot(ax=ax, color='yellow', markersize=100, label="Event Centroid", marker='x')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title('Visualization of Running Away Event with Directions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show plot
    plt.show()

# Example usage:
# street_gdf = gpd.read_file(street_shapefile_path)
# visualize_running_away_event_with_directions(street_gdf, people_gdf, running_geometries, event_centroid)

def calculate_directions(gdf):
    """
    Calculates the direction (bearing) between consecutive points for each person.
    Returns a GeoDataFrame with an additional 'direction' column.
    """
    # Calculate direction (bearing) between consecutive points
    gdf['direction'] = np.degrees(np.arctan2(
        gdf['geometry'].y.diff(),
        gdf['geometry'].x.diff()
    ))
   
    # Fill NaN values with the previous direction (for continuity)
    gdf['direction'] = gdf['direction'].ffill()
   
    return gdf

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

def generate_grid_for_street(street_gdf, grid_size=10):
    """
    Generates a grid covering the extent of the street shapefile.
   
    Args:
        street_gdf (GeoDataFrame): GeoDataFrame containing the street geometry.
        grid_size (float): Size of the grid squares (in the same units as the street shapefile's CRS).
       
    Returns:
        grid_gdf (GeoDataFrame): GeoDataFrame containing the grid of polygons.
    """
    # Calculate the bounding box of the street shapefile
    minx, miny, maxx, maxy = street_gdf.total_bounds
   
    # Generate the grid squares
    x_range = np.arange(minx, maxx, grid_size)
    y_range = np.arange(miny, maxy, grid_size)
   
    # Create the grid polygons
    polygons = []
    for x in x_range:
        for y in y_range:
            polygons.append(Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size)
            ]))
   
    # Convert the list of polygons into a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=street_gdf.crs)
   
    # Optional: Clip the grid to the street shapefile extent (if needed)
    grid_gdf = gpd.clip(grid_gdf, street_gdf)
   
    return grid_gdf



def calculate_density(gdf, grid):
    # Spatial join between the points and the grid
    joined = gpd.sjoin(grid, gdf, how='left', predicate='contains')
   
    # Count the number of points in each grid cell
    density = joined.groupby(joined.index).size()
   
    # Assign the density counts back to the grid
    grid['density'] = density
   
    # Fill NaN values with 0 (for grid cells with no points)
    grid['density'] = grid['density'].fillna(0)
   
    return grid



# def plot_density_grid(gdf, street_shapefile, cell_size=10,):
def plot_density_grid(gdf, street_shapefile, cell_size=10,):

    grid = generate_grid_for_street(street_shapefile, grid_size=3)
    density_grid = calculate_density(gdf, grid)  # Calculate density
   
    # Plot the density map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
   
    # Overlay the shapefile
    street_shapefile.boundary.plot(ax=ax, color='blue', linewidth=1.5)
   
    grid.plot(column='density', ax=ax, legend=True,
              cmap='OrRd', edgecolor='k', linewidth=0.2)
    ax.set_title('Crowd Density Map')
    # plt.show()



def kde_density_map(gdf, street_gdf, bandwidth=50, grid_size=100):
    """
    Generates a KDE density map that covers the extent of the street polygon.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the points of interest.
        street_gdf (GeoDataFrame): GeoDataFrame containing the street geometry.
        bandwidth (float): Bandwidth for KDE.
        grid_size (int): Number of grid points along each axis.

    Returns:
        x_mesh (array): Meshgrid X-coordinates.
        y_mesh (array): Meshgrid Y-coordinates.
        z_masked (array): Masked KDE density values.
    """
    # Extract coordinates from GeoDataFrame
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
   
    # Perform KDE
    kde = KernelDensity(bandwidth=bandwidth, metric='euclidean', kernel='gaussian')
    kde.fit(coords)
   
    # Create grid covering the bounding box of the street geometry
    min_x, min_y, max_x, max_y = street_gdf.total_bounds
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
   
    # Evaluate KDE on grid
    grid_coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    z = np.exp(kde.score_samples(grid_coords)).reshape(x_mesh.shape)
   
    # Create a mask for the grid based on the street polygon
    street_polygon = street_gdf.union_all()
    mask = np.array([Point(x, y).within(street_polygon) for x, y in grid_coords]).reshape(x_mesh.shape)
   
    # Apply the mask to the KDE result
    z_masked = np.ma.masked_where(~mask, z)
   
    return x_mesh, y_mesh, z_masked

# Example usage
# street_gdf = gpd.read_file('path_to_your_street_shapefile.shp')
# points_gdf = gpd.read_file('path_to_your_points_shapefile.shp')

# x_mesh, y_mesh, z_masked = kde_density_map(points_gdf, street_gdf)

# Plotting the result
# plt.imshow(z_masked, extent=(x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()), origin='lower', cmap='hot')
# street_gdf.boundary.plot(ax=plt.gca(), edgecolor='blue')
# plt.colorbar(label='Density')
# plt.show()

# def plot_kde_density_map(gdf,street_gdf, bandwidth=50, grid_size=100):
def plot_kde_density_map(gdf,street_gdf, bandwidth=50, grid_size=100):
    x_mesh, y_mesh, z = kde_density_map(gdf,street_gdf, bandwidth=20, grid_size=100)
    # Plot KDE density map
    plt.contourf(x_mesh, y_mesh, z, cmap='Reds', alpha=0.5)
    plt.colorbar(label='Density')

    # Overlay the shapefile
    street_gdf.boundary.plot(ax=plt.gca(), color='gray', linewidth=0.8)

    # Additional customization (optional)
    plt.title("KDE Density Map with Shapefile Overlay")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Display the plot
    # plt.show()

def create_arrows(gdf, arrow_length=0.05):
    """
    Creates small arrow geometries representing the direction of motion as caps on points.
    Returns a GeoDataFrame with the arrow geometries.
    """
    arrow_geometries = []
    for idx, row in gdf.iterrows():
        if not np.isnan(row['direction']):
            # Calculate the arrow's end point based on direction and very short length
            dx = arrow_length * np.cos(np.radians(row['direction']))
            dy = arrow_length * np.sin(np.radians(row['direction']))
            start_point = row['geometry']
            end_point = Point(start_point.x + dx, start_point.y + dy)
            arrow_geometries.append(LineString([start_point, end_point]))
        else:
            arrow_geometries.append(None)
   
    # Create a new GeoDataFrame for arrows
    arrow_gdf = gpd.GeoDataFrame(gdf, geometry=arrow_geometries)
    return arrow_gdf



def plot_arrows_on_map(gdf, arrow_gdf, street_shapefile_gdf, cmap='viridis'):
    """
    Plots the arrows on the map with color based on speed.
    """
    # Normalize the speed for color mapping
    norm = mcolors.Normalize(vmin=gdf['average_speed_30s'].min(), vmax=gdf['average_speed_30s'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot the base map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    street_shapefile_gdf.boundary.plot(ax=ax, color='blue', linewidth=1.5)
   
    # Plot arrows
    for idx, row in arrow_gdf.iterrows():
        if row['geometry'] is not None:
            color = sm.to_rgba(row['average_speed_30s'])
            ax.plot(*row['geometry'].xy, color=color, linewidth=2, alpha=0.8)
   
    # Add color bar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Instantaneous Speed (km/h)')
   
    plt.title('Direction of Motion with Speed Indication')
    plt.show()



def count_people_in_area(gdf, start_time, end_time, street_gdf=None):
    """
    Counts the number of unique people within the specified time period.
    Optionally, filters the people within the specified street area.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the people data with 'timestamp' and 'person_id' columns.
        start_time (str or datetime): Start of the time period (e.g., '2024-08-22 10:00:00').
        end_time (str or datetime): End of the time period (e.g., '2024-08-22 11:00:00').
        street_gdf (GeoDataFrame, optional): GeoDataFrame containing the street geometry. If provided, will filter
                                             the data to only include points within this geometry.

    Returns:
        int: The number of unique people in the area during the specified time period.
    """
    # Ensure that the timestamp column is in datetime format
    gdf['timestamp'] = pd.to_datetime(gdf['timestamp'])
   
    # Filter the GeoDataFrame based on the time period
    time_filtered_gdf = gdf[(gdf['timestamp'] >= start_time) & (gdf['timestamp'] <= end_time)]
   
    if street_gdf is not None:
        # Create a unified street polygon from the street GeoDataFrame
        street_polygon = street_gdf.union_all()
       
        # Spatially filter the GeoDataFrame based on the street area
        time_filtered_gdf = time_filtered_gdf[time_filtered_gdf.geometry.within(street_polygon)]
   
    # Count the number of unique person IDs
    unique_people_count = time_filtered_gdf['person_id'].nunique()
   
    return unique_people_count


def update_people_count(gdf, map_obj):
    count = len(gdf)
    folium.map.Marker(
        # [latitude, longitude],
        [11.322854, 75.936500],
        icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
                     html=f'<div style="font-size: 24pt">People Count: {count}</div>'),
    ).add_to(map_obj)

def get_start_end_times(gdf):
    """
    Returns the earliest (start) and latest (end) timestamps in the GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing a 'timestamp' column.

    Returns:
        tuple: A tuple containing the earliest and latest timestamps as (start_time, end_time).
    """
    # Ensure that the timestamp column is in datetime format
    gdf['timestamp'] = pd.to_datetime(gdf['timestamp'])
   
    # Get the minimum (earliest) and maximum (latest) timestamps
    start_time = gdf['timestamp'].min()
    end_time = gdf['timestamp'].max()
   
    return start_time, end_time


def load_street_shapefile(street_shapefile_path, epsg=32643):
    """Load the street shapefile and calculate the centroid."""
    street_gdf = gpd.read_file(street_shapefile_path)
    # Convert the CRS to UTM zone 43N (EPSG:32643) for more accurate distance measurements
    street_gdf = street_gdf.to_crs(epsg)
    #centroid = street_gdf.geometry.centroid
    #return centroid.y.iloc[0], centroid.x.iloc[0]  # Latitude, Longitude
    return street_gdf





# very big code ends here



@app.route("/clogging")
def clogging():
    # start_time = '2024-08-23 16:30:00'
    # end_time = '2024-08-23 16:31:00'
    # Set the end time to now
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set the start time to 5 minutes before now
    start_time = (datetime.now() - timedelta(seconds=40)).strftime('%Y-%m-%d %H:%M:%S')



    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5) 
    
    # Convert the GeoDataFrame to GeoJSON format
    # clogging_geojson = clogging_results.to_json()
    # return jsonify({"clogging_results": json.loads(clogging_geojson)})
    

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot street shapefile
    street.to_crs(epsg=3857).plot(ax=ax, color='gray', linewidth=0.1, label='Street')

    # Plot people locations
    people_gdf.to_crs(epsg=3857).plot(ax=ax, color='blue', markersize=5, label='People Locations')

    # Plot entry/exit points
    entry_exit.to_crs(epsg=3857).plot(ax=ax, color='green', markersize=20, label='Entry/Exit Points', edgecolor='black')

    # Plot detected clogging points
    clogging_results.to_crs(epsg=3857).plot(ax=ax, color='red', markersize=30, label='Clogging Detected Points', edgecolor='black')

    # Add basemap
    #ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Add basemap (CartoDB Positron view)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.legend()
    ax.set_title('Clogging Detection at Entry/Exit Points with Basemap')
    # plt.show()


    img = io.BytesIO()
    plt.savefig(img, format='png',bbox_inches='tight', transparent=True)
    img.seek(0)
    plt.close()  # Close the figure to free memory

    return send_file(img, mimetype='image/png')



@app.route("/bottleneck")
def bottleneck():
    # start_time = '2024-08-23 16:30:00'
    # end_time = '2024-08-23 16:31:00'
    # Set the end time to now
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set the start time to 5 minutes before now
    start_time = (datetime.now() - timedelta(seconds=40)).strftime('%Y-%m-%d %H:%M:%S')

    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    # print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5)
    hotspot_results, coldspot_results, clustering_bottlenecks = detect_clustering(people_gdf, 1, 50) 
    
    # Convert the GeoDataFrame to GeoJSON format
    # clogging_geojson = clogging_results.to_json()
    # return jsonify({"clogging_results": json.loads(clustering_bottlenecks)})
    # return jsonify({"message": clustering_bottlenecks})
    

    # Extracting x and y coordinates from the Point objects for hotspot results
    hotspot_x_coords = [point.x for point in hotspot_results]
    hotspot_y_coords = [point.y for point in hotspot_results]

    # Extracting x and y coordinates from the Point objects for coldspot results
    coldspot_x_coords = [point.x for point in coldspot_results]
    coldspot_y_coords = [point.y for point in coldspot_results]

    # Creating the scatter plot
    plt.figure(figsize=(10, 8))

    # Plotting hotspot results
    plt.scatter(hotspot_x_coords, hotspot_y_coords, color='red', marker='o', s=10, label='Hotspots')

    # Plotting coldspot results
    plt.scatter(coldspot_x_coords, coldspot_y_coords, color='blue', marker='o', s=10, label='Coldspots')

    plt.title("Hotspot and Coldspot Results")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    plt.grid(True)
    # plt.show()


    img = io.BytesIO()
    plt.savefig(img, format='png',bbox_inches='tight', transparent=True)
    img.seek(0)
    plt.close()  # Close the figure to free memory

    return send_file(img, mimetype='image/png')



@app.route("/grid_density")
def plot_grid_density():
    # start_time = '2024-08-23 16:30:00'
    # end_time = '2024-08-23 16:36:00'
    # Set the end time to now
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set the start time to 5 minutes before now
    start_time = (datetime.now() - timedelta(seconds=40)).strftime('%Y-%m-%d %H:%M:%S')
    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    # print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5)
    hotspot_results, coldspot_results, clustering_bottlenecks = detect_clustering(people_gdf, 1, 50) 

    # street_shapefile_path = r"C:\Users\Admin\Desktop\codes\rch\road.shp"
    street_shapefile_path = os.path.join(os.getcwd(), "road.shp")

    street_gdf=load_street_shapefile(street_shapefile_path, epsg=32643)
    print(people_gdf)
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')


    people_count=count_people_in_area(people_gdf, start_time, end_time, street_gdf=None)
    print('People count ', people_count)

    plt.figure(figsize=(10, 8))
    plot_density_grid(people_gdf, street_gdf, cell_size=10)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@app.route("/kde_density")
def plot_kde_density():
    # start_time = '2024-08-23 16:30:00'
    # end_time = '2024-08-23 16:36:00'
    # Set the end time to now
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set the start time to 5 minutes before now
    start_time = (datetime.now() - timedelta(seconds=40)).strftime('%Y-%m-%d %H:%M:%S')
    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    # print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5)
    hotspot_results, coldspot_results, clustering_bottlenecks = detect_clustering(people_gdf, 1, 50) 

    # street_shapefile_path = r"C:\Users\Admin\Desktop\codes\rch\road.shp"
    street_shapefile_path = os.path.join(os.getcwd(), "road.shp")

    street_gdf=load_street_shapefile(street_shapefile_path, epsg=32643)
    print(people_gdf)
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')


    people_count=count_people_in_area(people_gdf, start_time, end_time, street_gdf=None)
    print('People count ', people_count)

    plt.figure(figsize=(10, 8))
    plot_kde_density_map(people_gdf, street_gdf, bandwidth=10, grid_size=10)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@app.route("/running_event")
def plot_running_event():
    # start_time = '2024-08-23 16:30:00'
    # end_time = '2024-08-23 16:36:00'
    # Set the end time to now
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set the start time to 5 minutes before now
    start_time = (datetime.now() - timedelta(seconds=40)).strftime('%Y-%m-%d %H:%M:%S')
    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    # print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5)
    hotspot_results, coldspot_results, clustering_bottlenecks = detect_clustering(people_gdf, 1, 50) 

    # street_shapefile_path = r"C:\Users\Admin\Desktop\codes\rch\road.shp"
    street_shapefile_path = os.path.join(os.getcwd(), "road.shp")

    street_gdf=load_street_shapefile(street_shapefile_path, epsg=32643)
    print(people_gdf)
    people_gdf=calculate_average_speeds(people_gdf,speed_unit='km/h')


    people_count=count_people_in_area(people_gdf, start_time, end_time, street_gdf=None)
    print('People count ', people_count)

    detected, running_geometries, event_centroid = detect_running_away_anywhere(
        people_gdf, speed_threshold=5.0, distance_threshold=10.0, cluster_eps=5.0, min_samples=5)

    if detected:
        plt.figure(figsize=(10, 8))
        visualize_running_away_event_with_directions(
            street_gdf, people_gdf, running_geometries, event_centroid, show_arrow=True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
    else:
        return jsonify({"message": "No running event detected"}), 204


# -----




if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)










# map working

# from flask import Response, Flask, jsonify, send_file
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import psycopg2
# import os
# import json
# from urllib.parse import urlparse, parse_qs
# from datetime import datetime
# import time
# import io
# import matplotlib
# matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
# import matplotlib.pyplot as plt

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# cwd = os.getcwd()

# # Setup Flask-SocketIO with Eventlet
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# # Get env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
# DATABASE_URL = os.environ.get('DATABASE_URL')

# # Parse the DATABASE_URL to extract components
# result = urlparse(DATABASE_URL)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
# port = result.port
# query = parse_qs(result.query)

# # Connect to PostgreSQL using parsed components
# try:
#     conn = psycopg2.connect(
#         dbname=database,
#         user=username,
#         password=password,
#         host=hostname,
#         port=port,
#         options=f"-c search_path={query.get('schema', ['public'])[0]}"
#     )
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = conn.cursor()
#     print("Connected to PostgreSQL database!")
# except psycopg2.Error as e:
#     print(f"Error connecting to PostgreSQL database: {e}")

# # Fetch the latest camera_count for each unique camera_id
# def fetch_latest_data():
#     cursor.execute("""
#         SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#         FROM real_time_data
#         ORDER BY camera_id, timestamp DESC
#     """)
#     rows = cursor.fetchall()

#     # Convert datetime objects to strings
#     data = []
#     for row in rows:
#         camera_id, count_camera, timestamp = row
#         data.append({
#             "camera_id": camera_id,
#             "count_camera": count_camera,
#             "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#         })
    
#     return data

# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')

# @app.route("/db-test")
# def db_test():
#     try:
#         cursor.execute("""
#             SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#             FROM real_time_data
#             ORDER BY camera_id, timestamp DESC
#         """)
#         rows = cursor.fetchall()
        
#         # Convert datetime objects to strings
#         data = []
#         for row in rows:
#             camera_id, count_camera, timestamp = row
#             data.append({
#                 "camera_id": camera_id,
#                 "count_camera": count_camera,
#                 "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#             })

#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": data})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# # New route to generate and return a plot
# @app.route("/plot")
# def plot():
#     # Example plot
#     plt.figure(figsize=(10, 6))
#     plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
#     plt.title("Sample Plot")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
    
#     # Save the plot to a BytesIO object
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()  # Close the figure to free memory

#     return send_file(img, mimetype='image/png')

# if __name__ == "__main__":
#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)














# from flask import Flask, send_file, jsonify
# import matplotlib.pyplot as plt
# import io
# from flask_cors import CORS
# import json
# import os

# cwd = os.getcwd()

# app = Flask(__name__)
# CORS(app)

# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# @app.route("/api/home",methods=['GET'])
# # def return_home():
#     # return jsonify({
#     #     "message":"Hello world!"
#     # })
#     # return 
# def get_json_data():
#     file_path = cwd+'/exceed_count_data.json'  # Adjust the path if necessary
#     json_data = read_json_file(file_path)
#     return jsonify(json_data)


# @app.route('/download-plot')
# def download_plot():
#     # Generate the plot
#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
    
#     # Save the plot to a BytesIO object
#     img = io.BytesIO()
#     fig.savefig(img, format='png')
#     img.seek(0)
    
#     # Return the image for download
#     return send_file(img, mimetype='image/png', as_attachment=True, download_name='plot.png')


# if __name__ == "__main__":
#     app.run(debug=True,port=8080)







# from flask import Response, Flask, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import psycopg2
# import os
# import json
# from urllib.parse import urlparse, parse_qs
# from datetime import datetime
# import time

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# cwd = os.getcwd()

# # Setup Flask-SocketIO with Eventlet
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# # Get env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
# DATABASE_URL = os.environ.get('DATABASE_URL')

# # Parse the DATABASE_URL to extract components
# result = urlparse(DATABASE_URL)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
# port = result.port
# query = parse_qs(result.query)

# # Connect to PostgreSQL using parsed components
# try:
#     conn = psycopg2.connect(
#         dbname=database,
#         user=username,
#         password=password,
#         host=hostname,
#         port=port,
#         options=f"-c search_path={query.get('schema', ['public'])[0]}"
#     )
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = conn.cursor()
#     print("Connected to PostgreSQL database!")
# except psycopg2.Error as e:
#     print(f"Error connecting to PostgreSQL database: {e}")



# # Fetch the latest camera_count for each unique camera_id
# def fetch_latest_data():
#     cursor.execute("""
#         SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#         FROM real_time_data
#         ORDER BY camera_id, timestamp DESC
#     """)
#     rows = cursor.fetchall()

#     # Convert datetime objects to strings
#     data = []
#     for row in rows:
#         camera_id, count_camera, timestamp = row
#         data.append({
#             "camera_id": camera_id,
#             "count_camera": count_camera,
#             "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#         })
    
#     return data

# # Listen to PostgreSQL notifications on real_time_data table
# # #old
# # def listen_to_notifications():
# #     cursor.execute("LISTEN new_crowd_data;")
# #     print("Listening for new_crowd_data notifications...")
# #     while True:
# #         conn.poll()
# #         while conn.notifies:
# #             notify = conn.notifies.pop(0)
# #             payload = json.loads(notify.payload)
# #             print(f"Received notification: {payload}")
# #             # Fetch the latest data for each unique camera_id
# #             latest_data = fetch_latest_data()
# #             # Send the latest data to all connected clients
# #             socketio.emit('new_crowd_data', latest_data)
# #             print(f"Emitted latest data: {latest_data}")

# # new
# # Updated listen_to_notifications function with added sleep and better error handling
# def listen_to_notifications():
#     cursor.execute("LISTEN new_crowd_data;")
#     print("Listening for new_crowd_data notifications...")
#     while True:
#         try:
#             conn.poll()
#             if conn.notifies:
#                 while conn.notifies:
#                     notify = conn.notifies.pop(0)
#                     try:
#                         payload = json.loads(notify.payload)
#                         print(f"Received notification: {payload}")
#                         # Fetch the latest data for each unique camera_id
#                         latest_data = fetch_latest_data()
#                         # Send the latest data to all connected clients
#                         socketio.emit('new_crowd_data', latest_data)
#                         print(f"Emitted latest data: {latest_data}")
#                     except Exception as e:
#                         print(f"Error processing notification: {e}")
#             else:
#                 # Avoid busy waiting
#                 time.sleep(1)  # Sleep for a second before polling again
#         except Exception as e:
#             print(f"Error during polling or processing notifications: {e}")
#             time.sleep(5)  # Sleep for 5 seconds to avoid rapid retry in case of an error


# @socketio.on('connect')
# def handle_connect():
#     print('WebSocket client connected')
#     # Send the initial data when a client connects
#     initial_data = fetch_latest_data()
#     socketio.emit('initial_data', initial_data)
#     # Emit a test message to confirm WebSocket is working
#     socketio.emit('test', {'message': 'WebSocket connection established'})

# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')

# @app.route("/db-test")
# def db_test():
#     try:
#         cursor.execute("""
#             SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#             FROM real_time_data
#             ORDER BY camera_id, timestamp DESC
#         """)
#         rows = cursor.fetchall()
        
#         # Convert datetime objects to strings
#         data = []
#         for row in rows:
#             camera_id, count_camera, timestamp = row
#             data.append({
#                 "camera_id": camera_id,
#                 "count_camera": count_camera,
#                 "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#             })

#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": data})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500
    
# @app.route("/clogging")
# def db_test():
#     try:
#         cursor.execute("""
#             SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#             FROM real_time_data
#             ORDER BY camera_id, timestamp DESC
#         """)
#         rows = cursor.fetchall()
        
#         # Convert datetime objects to strings
#         data = []
#         for row in rows:
#             camera_id, count_camera, timestamp = row
#             data.append({
#                 "camera_id": camera_id,
#                 "count_camera": count_camera,
#                 "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#             })

#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": data})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# if __name__ == "__main__":
#     print("Starting the notification listener in a background task...")
#     socketio.start_background_task(listen_to_notifications)
#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)


































# -----------------------------------------------------------------------------


# try 5 - working

# from flask import Response, Flask, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import psycopg2
# import os
# import json
# from urllib.parse import urlparse, parse_qs
# from datetime import datetime
# import time

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# cwd = os.getcwd()

# # Setup Flask-SocketIO with Eventlet
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# # Get env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
# DATABASE_URL = os.environ.get('DATABASE_URL')

# # Parse the DATABASE_URL to extract components
# result = urlparse(DATABASE_URL)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
# port = result.port
# query = parse_qs(result.query)

# # Connect to PostgreSQL using parsed components
# try:
#     conn = psycopg2.connect(
#         dbname=database,
#         user=username,
#         password=password,
#         host=hostname,
#         port=port,
#         options=f"-c search_path={query.get('schema', ['public'])[0]}"
#     )
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = conn.cursor()
#     print("Connected to PostgreSQL database!")
# except psycopg2.Error as e:
#     print(f"Error connecting to PostgreSQL database: {e}")


# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# @app.route("/api/home",methods=['GET'])
# # def return_home():
#     # return jsonify({
#     #     "message":"Hello world!"
#     # })
#     # return 
# def get_json_data():
#     file_path = cwd+'/exceed_count_data.json'  # Adjust the path if necessary
#     json_data = read_json_file(file_path)
#     return jsonify(json_data)

# # Fetch the latest camera_count for each unique camera_id
# def fetch_latest_data():
#     cursor.execute("""
#         SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#         FROM real_time_data
#         ORDER BY camera_id, timestamp DESC
#     """)
#     rows = cursor.fetchall()

#     # Convert datetime objects to strings
#     data = []
#     for row in rows:
#         camera_id, count_camera, timestamp = row
#         data.append({
#             "camera_id": camera_id,
#             "count_camera": count_camera,
#             "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#         })
    
#     return data

# # Listen to PostgreSQL notifications on real_time_data table
# # #old
# # def listen_to_notifications():
# #     cursor.execute("LISTEN new_crowd_data;")
# #     print("Listening for new_crowd_data notifications...")
# #     while True:
# #         conn.poll()
# #         while conn.notifies:
# #             notify = conn.notifies.pop(0)
# #             payload = json.loads(notify.payload)
# #             print(f"Received notification: {payload}")
# #             # Fetch the latest data for each unique camera_id
# #             latest_data = fetch_latest_data()
# #             # Send the latest data to all connected clients
# #             socketio.emit('new_crowd_data', latest_data)
# #             print(f"Emitted latest data: {latest_data}")

# # new
# # Updated listen_to_notifications function with added sleep and better error handling
# def listen_to_notifications():
#     cursor.execute("LISTEN new_crowd_data;")
#     print("Listening for new_crowd_data notifications...")
#     while True:
#         try:
#             conn.poll()
#             if conn.notifies:
#                 while conn.notifies:
#                     notify = conn.notifies.pop(0)
#                     try:
#                         payload = json.loads(notify.payload)
#                         print(f"Received notification: {payload}")
#                         # Fetch the latest data for each unique camera_id
#                         latest_data = fetch_latest_data()
#                         # Send the latest data to all connected clients
#                         socketio.emit('new_crowd_data', latest_data)
#                         print(f"Emitted latest data: {latest_data}")
#                     except Exception as e:
#                         print(f"Error processing notification: {e}")
#             else:
#                 # Avoid busy waiting
#                 time.sleep(1)  # Sleep for a second before polling again
#         except Exception as e:
#             print(f"Error during polling or processing notifications: {e}")
#             time.sleep(5)  # Sleep for 5 seconds to avoid rapid retry in case of an error


# @socketio.on('connect')
# def handle_connect():
#     print('WebSocket client connected')
#     # Send the initial data when a client connects
#     initial_data = fetch_latest_data()
#     socketio.emit('initial_data', initial_data)
#     # Emit a test message to confirm WebSocket is working
#     socketio.emit('test', {'message': 'WebSocket connection established'})

# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')

# @app.route("/db-test")
# def db_test():
#     try:
#         cursor.execute("""
#             SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#             FROM real_time_data
#             ORDER BY camera_id, timestamp DESC
#         """)
#         rows = cursor.fetchall()
        
#         # Convert datetime objects to strings
#         data = []
#         for row in rows:
#             camera_id, count_camera, timestamp = row
#             data.append({
#                 "camera_id": camera_id,
#                 "count_camera": count_camera,
#                 "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime to string
#             })

#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": data})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# if __name__ == "__main__":
#     print("Starting the notification listener in a background task...")
#     socketio.start_background_task(listen_to_notifications)
#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)





# try 4
# from flask import Response, Flask, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import psycopg2
# import os
# import json
# from urllib.parse import urlparse, parse_qs

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Setup Flask-SocketIO
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Get env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
# DATABASE_URL = os.environ.get('DATABASE_URL')

# # Parse the DATABASE_URL to extract components
# result = urlparse(DATABASE_URL)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
# port = result.port
# query = parse_qs(result.query)

# # Connect to PostgreSQL using parsed components
# try:
#     conn = psycopg2.connect(
#         dbname=database,
#         user=username,
#         password=password,
#         host=hostname,
#         port=port,
#         options=f"-c search_path={query.get('schema', ['public'])[0]}"
#     )
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = conn.cursor()
#     print("Connected to PostgreSQL database!")
# except psycopg2.Error as e:
#     print(f"Error connecting to PostgreSQL database: {e}")

# # Fetch the latest camera_count for each unique camera_id
# def fetch_latest_data():
#     cursor.execute("""
#         SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#         FROM real_time_data
#         ORDER BY camera_id, timestamp DESC
#     """)
#     rows = cursor.fetchall()
#     return rows

# # Listen to PostgreSQL notifications on real_time_data table
# def listen_to_notifications():
#     cursor.execute("LISTEN new_crowd_data;")
#     print("Listening for new_crowd_data notifications...")
#     while True:
#         conn.poll()
#         while conn.notifies:
#             notify = conn.notifies.pop(0)
#             payload = json.loads(notify.payload)
#             print(f"Received notification: {payload}")
#             # Fetch the latest data for each unique camera_id
#             latest_data = fetch_latest_data()
#             # Send the latest data to all connected clients
#             socketio.emit('new_crowd_data', latest_data)
#             print(f"Emitted latest data: {latest_data}")

# @socketio.on('connect')
# def handle_connect():
#     print('WebSocket client connected')
#     # Send the initial data when a client connects
#     initial_data = fetch_latest_data()
#     socketio.emit('initial_data', initial_data)
#     # Emit a test message to confirm WebSocket is working
#     socketio.emit('test', {'message': 'WebSocket connection established'})


# # ## working code
# # @socketio.on('connect')
# # def handle_connect():
# #     print('WebSocket client connected')
# #     # Send hardcoded initial data
# #     hardcoded_data = [
# #         {"camera_id": 1, "count_camera": 5, "timestamp": "2024-08-24 12:34:56"},
# #         {"camera_id": 2, "count_camera": 10, "timestamp": "2024-08-24 12:35:00"}
# #     ]
# #     print(f"Emitting hardcoded initial data: {hardcoded_data}")
# #     socketio.emit('initial_data', hardcoded_data)



# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')

# # @app.route("/home")
# # def home():
# #     return Response("This will be handled by Next.js", status=200, mimetype='text/html')


# @app.route("/db-test")
# def db_test():
#     try:
#         # cursor.execute("SELECT * FROM real_time_data;")
#         cursor.execute("""
#             SELECT DISTINCT ON (camera_id) camera_id, count_camera, timestamp
#             FROM real_time_data
#             ORDER BY camera_id, timestamp DESC
#         """)
#         rows = cursor.fetchall()
#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": rows})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# if __name__ == "__main__":
#     # Start listening to notifications
#     socketio.start_background_task(listen_to_notifications)
#     # Run the Flask app with WebSocket support
#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)













# from flask import Response, Flask, jsonify, request
# from flask_cors import CORS
# from dotenv import load_dotenv
# import json
# import os

# cwd = os.getcwd()
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # GET the env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')


# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', debug=True, port=8080)


# try3
# import os
# import psycopg2
# from flask import Flask
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import select
# import json
# import threading

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Database connection
# def get_db_connection():
#     conn = psycopg2.connect(os.environ['DATABASE_URL'])
#     cur = conn.cursor()
#     cur.execute('SET search_path TO public')  # Replace 'your_schema_name' with your schema
#     cur.close()
#     return conn

# # Listen for notifications from PostgreSQL
# def listen_to_db():
#     conn = get_db_connection()
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cur = conn.cursor()
    
#     # Listen to the 'crowd_data_channel'
#     cur.execute("LISTEN crowd_data_channel;")

#     print("Waiting for notifications on crowd_data_channel...")
#     while True:
#         if select.select([conn], [], [], 5) == ([], [], []):
#             continue
#         conn.poll()
#         while conn.notifies:
#             notify = conn.notifies.pop(0)
#             new_data = json.loads(notify.payload)
#             # Broadcast the new data to all connected clients
#             socketio.emit('crowd_data', json.dumps(new_data))

#     cur.close()
#     conn.close()

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# if __name__ == "__main__":
#     listener_thread = threading.Thread(target=listen_to_db)
#     listener_thread.daemon = True
#     listener_thread.start()

#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)









# try2 - working

# from flask import Response, Flask, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from dotenv import load_dotenv
# import psycopg2
# import os
# import json
# from urllib.parse import urlparse, parse_qs

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Setup Flask-SocketIO
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Get env variables
# app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
# DATABASE_URL = os.environ.get('DATABASE_URL')

# # Parse the DATABASE_URL to extract components
# result = urlparse(DATABASE_URL)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
# port = result.port
# query = parse_qs(result.query)

# # Connect to PostgreSQL using parsed components
# try:
#     conn = psycopg2.connect(
#         dbname=database,
#         user=username,
#         password=password,
#         host=hostname,
#         port=port,
#         options=f"-c search_path={query.get('schema', ['public'])[0]}"
#     )
#     conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = conn.cursor()
#     print("Connected to PostgreSQL database!")
# except psycopg2.Error as e:
#     print(f"Error connecting to PostgreSQL database: {e}")

# # Fetch latest data from the database
# def fetch_latest_data():
#     cursor.execute("SELECT * FROM real_time_data ORDER BY timestamp DESC LIMIT 100;")
#     rows = cursor.fetchall()
#     return rows

# # Listen to PostgreSQL notifications on real_time_data table
# def listen_to_notifications():
#     cursor.execute("LISTEN new_crowd_data;")
#     print("Listening for new_crowd_data notifications...")
#     while True:
#         conn.poll()
#         while conn.notifies:
#             notify = conn.notifies.pop(0)
#             payload = json.loads(notify.payload)
#             print(f"Received notification: {payload}")
#             socketio.emit('new_crowd_data', payload)
#             print(f"Emitted payload: {payload}")

# @socketio.on('connect')
# def handle_connect():
#     print('WebSocket client connected')
#     # Send the initial data when a client connects
#     initial_data = fetch_latest_data()
#     socketio.emit('initial_data', initial_data)
#     # Emit a test message to confirm WebSocket is working
#     socketio.emit('test', {'message': 'WebSocket connection established'})

# @app.route("/test")
# def test():
#     return Response('Server is working!', status=201, mimetype='application/json')

# @app.route("/db-test")
# def db_test():
#     try:
#         cursor.execute("SELECT * FROM real_time_data;")
#         rows = cursor.fetchall()
#         print(f"Fetched {len(rows)} rows from real_time_data")
#         return jsonify({"message": "Database connection successful", "data": rows})
#     except psycopg2.Error as e:
#         print(f"Database query failed: {e}")
#         return jsonify({"message": "Database connection failed", "error": str(e)}), 500

# if __name__ == "__main__":
#     # Start listening to notifications
#     socketio.start_background_task(listen_to_notifications)
#     # Run the Flask app with WebSocket support
#     socketio.run(app, host='0.0.0.0', port=8080, debug=True)
