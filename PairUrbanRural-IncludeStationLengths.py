
# coding: utf-8

# In[1]:

# import libraries
#get_ipython().magic(u'matplotlib inline')
import numpy as np
#import csv
#import matplotlib.pyplot as plt
import pandas as pd
import glob
import ulmo
import os
import scipy.spatial


# In[2]:

ghcn = pd.read_fwf('data/ghcnd-stations.txt', colspecs = [(0,11), (12,19), (21,29), (31,36),(38,40), (41,70), (72,74),(76,78),(80,85)], header = None) 
colnames = ['GHCN ID', 'lat', 'lon', 'elevation', 'state', 'name', 'gsn flag', 'HCN/CRN FLAG', 'WMO ID']
ghcn.columns = colnames

# append the brightness index 
BI = np.load('data/brightnessGHCN.npy')
ghcn['Brightness'] = BI
# from http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
# FORMAT OF "ghcnd-stations.txt"
#
# ------------------------------
# Variable   Columns   Type
# ------------------------------
# ID            1-11   Character
# LATITUDE     13-20   Real
# LONGITUDE    22-30   Real
# ELEVATION    32-37   Real
# STATE        39-40   Character
# NAME         42-71   Character
# GSN FLAG     73-75   Character
# HCN/CRN FLAG 77-79   Character
# WMO ID       81-85   Character
# ------------------------------

# These variables have the following definitions:

# ID         is the station identification code.  Note that the first two
#            characters denote the FIPS  country code, the third character 
#            is a network code that identifies the station numbering system 
#            used, and the remaining eight characters contain the actual 
#            station ID. 

#            See "ghcnd-countries.txt" for a complete list of country codes.
# 	   See "ghcnd-states.txt" for a list of state/province/territory codes.

#            The network code  has the following five values:

#            0 = unspecified (station identified by up to eight 
# 	       alphanumeric characters)
# 	   1 = Community Collaborative Rain, Hail,and Snow (CoCoRaHS)
# 	       based identification number.  To ensure consistency with
# 	       with GHCN Daily, all numbers in the original CoCoRaHS IDs
# 	       have been left-filled to make them all four digits long. 
# 	       In addition, the characters "-" and "_" have been removed 
# 	       to ensure that the IDs do not exceed 11 characters when 
# 	       preceded by "US1". For example, the CoCoRaHS ID 
# 	       "AZ-MR-156" becomes "US1AZMR0156" in GHCN-Daily
#            C = U.S. Cooperative Network identification number (last six 
#                characters of the GHCN-Daily ID)
# 	   E = Identification number used in the ECA&D non-blended
# 	       dataset
# 	   M = World Meteorological Organization ID (last five
# 	       characters of the GHCN-Daily ID)
# 	   N = Identification number used in data supplied by a 
# 	       National Meteorological or Hydrological Center
# 	   R = U.S. Interagency Remote Automatic Weather Station (RAWS)
# 	       identifier
# 	   S = U.S. Natural Resources Conservation Service SNOwpack
# 	       TELemtry (SNOTEL) station identifier
#            W = WBAN identification number (last five characters of the 
#                GHCN-Daily ID)

# LATITUDE   is latitude of the station (in decimal degrees).

# LONGITUDE  is the longitude of the station (in decimal degrees).

# ELEVATION  is the elevation of the station (in meters, missing = -999.9).


# STATE      is the U.S. postal code for the state (for U.S. stations only).

# NAME       is the name of the station.

# GSN FLAG   is a flag that indicates whether the station is part of the GCOS
#            Surface Network (GSN). The flag is assigned by cross-referencing 
#            the number in the WMOID field with the official list of GSN 
#            stations. There are two possible values:

#            Blank = non-GSN station or WMO Station number not available
#            GSN   = GSN station 

# HCN/      is a flag that indicates whether the station is part of the U.S.
# CRN FLAG  Historical Climatology Network (HCN).  There are three possible 
#           values:

#            Blank = Not a member of the U.S. Historical Climatology 
# 	           or U.S. Climate Reference Networks
#            HCN   = U.S. Historical Climatology Network station
# 	   CRN   = U.S. Climate Reference Network or U.S. Regional Climate 
# 	           Network Station

# WMO ID     is the World Meteorological Organization (WMO) number for the
#            station.  If the station has no WMO number (or one has not yet 
# 	   been matched to this station), then the field is blank.

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
giss = pd.read_fwf('data/v3.temperature.inv.txt',skiprows = 39, header = None,
                  colspecs=[(0,3),(3,8),(8,11), (12,44),(44,49), (52,58), (58,63), (63,67), (67,68), (69,73), (73,75), (75, 77), (78,79), (79,81), (81,82),(82,84), (84,100), (100,102), (103,106)])
colnames = ['icc country code', 'WMO ID', '3 digit modifier', 'name','lat', 'lon', 'elevation', 'TELe', 'P', 'Pop', 'Tp', 'V', 'Lo', 'Co', 'Airport', 'ds', 'Vege', 'bi', 'BI']
giss.columns = colnames

# LEGEND  
# ======
# icc  =3 digit country code; the first digit represents WMO region/continent
# WMO_#=5 digit WMO station number
# ...  =3 digit modifier; 000 means the station is probably the WMO
#       station; 001, etc. mean the station is near that WMO station
# Name =30 character station name
# Lat  =latitude in degrees, negative = South of Equator
# Lon  =longitude in degrees, negative = West of Greenwich (England)
# Elev =station elevation in meters, missing is -999
# TEle =station elevation interpolated from TerrainBase gridded data set
# P    =R if rural (not associated with a town of >10,000 population)
#       S if associated with a small town (10,000-50,000 population)
#       U if associated with an urban area (>50,000 population)
# Pop  =population of the small town or urban area in 1000s
#       If rural, no analysis:  -9.
# Tp   =general topography around the station:  FL flat; HI hilly,
#       MT mountain top; MV mountainous valley or at least not on the top
#       of a mountain.
# V    =general vegetation near the station based on Operational
#       Navigation Charts;  MA marsh; FO forested; IC ice; DE desert;
#       CL clear or open;  xx information not provided
# Lo   =CO if station is within 30 km from the coast
#       LA if station is next to a large (> 25 km**2) lake
#       no if neither of the above
#       Note: Stations which are both CO and LA will be marked CO
# Co   =distance in km to the coast if Lo=CO, else -9
# A    =A if the station is at an airport; else x
# ds   =distance in km from the airport to its associated
#       small town or urban center (not relevant for rural airports
#       or non airport stations in which case ds=-9)
# Vege =gridded vegetation for the 0.5x0.5 degree grid point closest
#       to the station from a gridded vegetation data base. 16 characters.
# bi   =brightness index    A=dark B=dim C=bright   (comment added by R.Ruedy)
# BI   =brightness index    0=dark -> 256 =bright   (based on satellite night light data)


# see: http://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly

# In[3]:

# subset the GHCN station list with the list of available stations
currentstations = ulmo.ncdc.ghcn_daily.get_stations(start_year=1985, end_year = 2016, elements = ['TMIN', 'TMAX', 'AWND'], as_dataframe=True, update=False)
currentGHCNstations = np.intersect1d(currentstations.id, ghcn.index.values) #ghcn['GHCN ID'].values)
ghcnSubset = ghcn.set_index('GHCN ID').loc[currentstations.id.values]

# at this point, ghcn must have the station id set as the index 


# In[4]:

ghcn = ghcn.set_index('GHCN ID').loc[currentstations.id.values]


# In[5]:

# compute distances between all stations
#tree = scipy.spatial.cKDTree(giss[['lon', 'lat']].values, leafsize=100)
# query the closest point 
#closestInd = tree.query(giss[['lon', 'lat']].values[11,:], k =2, distance_upper_bound=6)[1][1]


# In[5]:

atlas = pd.read_csv('data/sampleAtlas.csv') # derived  from http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
atlas = pd.read_csv('data/world_cities.csv')
tree = scipy.spatial.cKDTree(ghcn[['lon', 'lat']].values, leafsize=100)
#import sys
#sys.path.append('/../cityheat/Bmore/2015/')
#sys.path.append('/Users/annascott2/mountpoint/cityheat/Bmore/2015/')
#import spatialfunctions
atlas = atlas[atlas['pop']> 500000]


# In[ ]:




# In[473]:

brightness_threshold = 30 # this is the urban/rural threshold delimiter
pairs = []
for i in range(0, atlas.shape[0]): 
    lat = atlas.iloc[i]['lat']
    lon = atlas.iloc[i]['lng']
    city = atlas.iloc[i]['city']
    closeststations = tree.query([lon,lat], k =15, distance_upper_bound=1)

    # Make sure the closest stations are within a finite distance
    closestInds = closeststations[1][~np.isinf(closeststations[0])]
    # make sure that there are stations 
    if closestInds.shape[0] > 0 : 
        urban = ghcn.Brightness[closestInds].argmax()
        rural = ghcn.Brightness[closestInds].argmin()
        #         if urban != rural: 
        urban_data = ulmo.ncdc.ghcn_daily.get_data(urban, as_dataframe=True, update=False)
        rural_data = ulmo.ncdc.ghcn_daily.get_data(rural, as_dataframe=True, update=False)

        number_urban_stations = (ghcn.Brightness[closestInds] >= brightness_threshold).sum() # number of urban stations is equivalent to brightness over 30
        number_rural_stations = (ghcn.Brightness[closestInds] < brightness_threshold).sum() # number of rural stations is equivalent to brightness over 30
        delta_brightness = 30
        break_value = -1

        iii = 1
        while ('TMAX' not in urban_data.keys() or 'TMIN' not in urban_data.keys()) or (urban_data['TMAX'].index[0].to_timestamp().to_datetime() > startdate) or (urban_data['TMAX'].index[-1].to_timestamp().to_datetime() < enddate) : 
            # check if there's enough suitable urban stations to check the next one, ignoring the current one 
            if iii > number_urban_stations-1 : 
                print 'no suitable urban station for %s'%city
                urban = break_value
                break 
            # find the next brightest
            print 'finding the next urban station for %s'%city
            urban = ghcn.Brightness[closestInds][(-ghcn.Brightness[closestInds].values).argsort()].index[iii]
            urban_data = ulmo.ncdc.ghcn_daily.get_data(urban, as_dataframe=True, update=False)
            print urban, urban_data['TMAX'].index[0], urban_data['TMAX'].index[-1]
            iii = iii+1    

        # if no urban station found, don't bother to pair with rural 
        if urban != break_value : 
            # check rural data station different from urban has TMIN, TMAX, a long enough record, and is rural enough compared to urban (delta_brightness)
            iii = 1
            enlarge_circle = 0 # number of times we can try increasing search radius
            while (urban ==rural) or ('TMAX' not in rural_data.keys() and 'TMIN' not in rural_data.keys()) or (rural_data['TMAX'].index[0].to_timestamp().to_datetime() > startdate) or (rural_data['TMAX'].index[-1].to_timestamp().to_datetime() < enddate) or (ghcn.Brightness[urban]-ghcn.Brightness[rural]>delta_brightness): 
                print 'finding the next rural station for %s'%city

                # if we can't find any suitable stations, try enlarging the search radius, but only try this once
                if iii > number_rural_stations-1 : 
                    if enlarge_circle > 0 : 
                        rural = break_value
                        break #break when we've already used up all the search
                    else: 
                        # find 25 closest stations within 1.5 degree circle
                        closeststations1 = tree.query([lon,lat], k =35, distance_upper_bound=1.5 ) 
                        # eliminate the ones we've already search 
                        new_stations = np.setdiff1d(closeststations1[1], closeststations[1])
                        # Make sure that we've actually found new stations, otherwise break
                        if new_stations.shape[0] >0 : 
                            # Make sure the closest stations are within a finite distance
                            closestInds1 = closeststations1[1][~np.isinf(closeststations1[0])]
                            # Eliminate repeat stations
                            closestInds = np.intersect1d(new_stations, closestInds1)
                            # reset counter, number of rural stations
                            iii = 0
                            number_rural_stations = (ghcn.Brightness[closestInds] < brightness_threshold).sum()
                        else: 
                            rural = break_value
                            break
                        # reset list of rural stations
                    print 'increasing search radius'
                    enlarge_circle = enlarge_circle+1
                    
                # find the next dimmest
                rural = ghcn.Brightness[closestInds][(ghcn.Brightness[closestInds].values).argsort()].index[iii]
                rural_data = ulmo.ncdc.ghcn_daily.get_data(rural, as_dataframe=True, update=False)
                print rural, rural_data['TMAX'].index[0], rural_data['TMAX'].index[-1]
                iii = iii+1
        else: 
            rural = break_value
    # else condition for not being any stations
    else: 
        urban = break_value 
        rural = break_value
            
    # save out if we've found a good pairing
    if (urban != break_value) and (rural != break_value) : 
        print 'Found a pair for %s'%city
        frames.append([city, urban, #['GHCN ID'],
                       ghcn.loc[urban].lat, 
                       ghcn.loc[urban].lon, 
                       ghcn.loc[urban].Brightness, 
                       rural, #['GHCN ID'],
                       ghcn.loc[rural].lat, 
                       ghcn.loc[rural].lon, 

                       ghcn.loc[rural].Brightness])
    else : 
        frames.append([city, np.nan, #['GHCN ID'],
                       np.nan,
                       np.nan,
                       np.nan, 
                       np.nan, #['GHCN ID'],
                       np.nan,
                       np.nan,
                       np.nan])

    # periodiically save out results
    if i%50 ==0: 
        pairs = pd.DataFrame(frames, columns = ['City', 'Urban station', 
                                          'Urban Lat', 'Urban Lon','Urban brightness', 
                                          'Rural station', 
                                          'Rural Lat', 'Rural Lon','Rural brightness'])
        pairs.to_csv('GHCNpairedstations_checkedrecordlengths.csv')
        
pairs = pd.DataFrame(frames, columns = ['City', 'Urban station', 
                                  'Urban Lat', 'Urban Lon','Urban brightness', 
                                  'Rural station', 
                                  'Rural Lat', 'Rural Lon','Rural brightness'])
pairs.to_csv('GHCNpairedstations_checkedrecordlengths.csv')


# In[ ]:




# In[18]:

# # calculate the brightness index for all stations
# from osgeo import ogr, osr
# import os
# from shapely.geometry import Point
# import shapely.geometry
# import shapely.wkt
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# import gdal

# # rasterfile = 'data/F182013.v4/F182013.v4c_web.stable_lights.avg_vis.tif'
# # layer = gdal.Open(rasterfile)
# # gt =layer.GetGeoTransform()
# # bands = layer.RasterCount
# # src = layer.GetRasterBand(1)
# X = ghcnSubset['lon']
# Y = ghcnSubset['lat']


# In[19]:

# #rasterfile = 'data/F182013.v4/F182013.v4c_web.stable_lights.avg_vis.tif'
# rasterfile = 'data/F182013.v4/F182013.v4c_web.avg_vis.tif'
# layer = gdal.Open(rasterfile)
# gt =layer.GetGeoTransform()
# bands = layer.RasterCount
# src = layer.GetRasterBand(1)

# BI = np.zeros(X.shape)
# i = 0 
# for x,y in zip(X,Y): 
#     rasterx = int((x - gt[0]) / gt[1])
#     rastery = int((y - gt[3]) / gt[5])
#     BI[i] = src.ReadAsArray(rasterx,rastery, win_xsize=1, win_ysize=1)
#     i = i+1

# #np.save('brightnessGHCNsubset.npy', BI)
# #BI = np.load('data/brightnessGHCN.npy')


# In[ ]:



