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


@app.route("/clogging")
def clogging():
    start_time = '2024-08-23 16:30:00'
    end_time = '2024-08-23 16:31:00'
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
    fig, ax = plt.subplots(figsize=(10, 10))

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
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the figure to free memory

    return send_file(img, mimetype='image/png')



@app.route("/bottleneck")
def bottleneck():
    start_time = '2024-08-23 16:30:00'
    end_time = '2024-08-23 16:31:00'
    people_gdf = fetch_people_locations(conn, start_time, end_time,epsg=32643 )
    # street_shapefile = "/road.shp"
    # entry_exit_shapefile = "/Entry_Exit_Lines.shp"
    street_shapefile = os.path.join(os.getcwd(), "road.shp")
    entry_exit_shapefile = os.path.join(os.getcwd(), "Entry_Exit_Lines.shp")
    print(street_shapefile)
    street, entry_exit = load_shapefiles(street_shapefile, entry_exit_shapefile)  # Load shapefiles
    clogging_results = detect_clogging(entry_exit, people_gdf, 1, 5)
    hotspot_results, coldspot_results, clustering_bottlenecks = detect_clustering(people_gdf, 1, 50) 
    
    # Convert the GeoDataFrame to GeoJSON format
    # clogging_geojson = clogging_results.to_json()
    # return jsonify({"clogging_results": json.loads(clogging_geojson)})
    

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

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
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the figure to free memory

    return send_file(img, mimetype='image/png')

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
