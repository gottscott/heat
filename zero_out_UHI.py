import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ulmo
import os
import scipy.stats
results_filepath = 'plots/version1/'
pairs = pd.read_csv('USghcnpairs_stationlengths.csv')
df = pairs[917:]
df = df[~np.isnan(df['Urban brightness'])]
# compute UHI composite setting UHI @ HW day(0) to 0
composite_tmin = np.zeros([df.shape[0], 11])
composite_UHImin = np.zeros([df.shape[0], 11])
composite_UHImin2 = np.zeros([df.shape[0], 11])
composite_UHImin3 = np.zeros([df.shape[0], 11])

for i in range(0, df.shape[0]): 
    city = df.iloc[i]['City']
    city = unicode(city, errors = 'ignore')
    urbanID = df.iloc[i]['Urban station']
    ruralID = df.iloc[i]['Rural station']
    print city
    # Downloadd from NCDC the station data, using the station ID listed in station list
    urbandata = ulmo.ncdc.ghcn_daily.get_data(urbanID,
                                         as_dataframe=True)
    ruraldata = ulmo.ncdc.ghcn_daily.get_data(ruralID,
                                         as_dataframe=True)
    # Calculate minimum daily thresholds starting from 1965
    startdate = '1985-01-01'#max(min(ruraldata['TMIN'].index), min(urbandata['TMIN'].index))
    tmin = pd.to_numeric(ruraldata['TMIN'][startdate:].value/10.) 
    numhw = 30 # number of heatwaves
    # min hw
    tmin = tmin[startdate:]
    hottestmin = tmin.iloc[(-tmin.values).argsort()[:numhw]] #Get the hottest days based off tmin 
    minheatwaves = hottestmin

    # Make sure that events aren't duplicates 
    # get the time difference between events (sorted in temporal order, obviously)
    time_diff = (minheatwaves.sort_index().index.to_timestamp().values[1:] - minheatwaves.sort_index().index.to_timestamp().values[:-1]).astype('timedelta64[D]')
    # find where the events are not within 2 days of each other
    minheatwaves = minheatwaves.sort_index()[time_diff > np.timedelta64(2, 'D')]
    # Now the heatwaves are sorted in time order, but we want numhw (10) of the most severe events. Save the hottest 10 events
    minheatwaves = minheatwaves.sort_values().iloc[0:10]

    UHI = pd.to_numeric(urbandata['TMIN']['1985-01-01':].value/10.) - tmin
    temp = tmin
    heatwaves = minheatwaves
    compositeTemp = np.zeros([heatwaves.shape[0], 11])
    compositeUHI = np.zeros([heatwaves.shape[0], 11])
    compositeUHI2 = np.zeros([heatwaves.shape[0], 11])
    compositeUHI3 = np.zeros([heatwaves.shape[0], 11])
    ii = 0
    try: 
        for dates in heatwaves.index[:]: 
	    compositeUHI[ii,:] = UHI[dates.to_timestamp()-pd.DateOffset(days=5):dates.to_timestamp()+pd.DateOffset(days=5)].values# -UHI[dates.to_timestamp()] 
            compositeTemp[ii,:]= temp[dates.to_timestamp()-pd.DateOffset(days=5):dates.to_timestamp()+pd.DateOffset(days=5)].values
            ii = ii+1

        composite_tmin[i,:] = np.nanmean(compositeTemp, axis=0)
    # save out composite UHI
        composite_UHImin[i,:] = np.nanmean(compositeUHI, axis=0)
    except ValueError: 
	compositeUHI[ii,:] = np.nan*np.ones([1,11]) 
        compositeTemp[ii,:]= np.nan*np.ones([1,11]) 
    if np.mod(i,10) ==0 : 
	compositeTempDF = pd.DataFrame(composite_tmin, columns=np.arange(-5,6,1)).set_index(df['City'])
	compositeTempDF.to_csv(results_filepath + 'composite_temp.csv')

	compositeUHIDF = pd.DataFrame(composite_UHImin, columns=np.arange(-5,6,1)).set_index(df['City'])
	compositeUHIDF.to_csv(results_filepath + 'composite_UHI.csv')

compositeTempDF = pd.DataFrame(composite_tmin, columns=np.arange(-5,6,1)).set_index(df['City'])
compositeTempDF.to_csv(results_filepath + 'composite_temp.csv')

compositeUHIDF = pd.DataFrame(composite_UHImin, columns=np.arange(-5,6,1)).set_index(df['City'])
compositeUHIDF.to_csv(results_filepath + 'composite_UHI.csv')


# plot heatwave composites for all stations
x = np.arange(-5,6)
# plot heatwave composites for all stations
plt.figure(figsize = [15,15])
#plot temperature,raw
plt.subplot(2,2,1)
for i in range(0, compositeTempDF.shape[0]) : 
    #print compositeUHIDF2.iloc[i].values[1:]
    plt.plot(x, compositeTempDF.iloc[i].values[1:])

#compositeTempDF.mean(axis=0).plot(yerr = compositeTempDF.std(axis=0))
plt.plot( x, compositeTempDF.mean(), color = 'k', linewidth = 3)
plt.xlabel('Event Day')
plt.ylabel('Temp ($^\circ$C)')
plt.title('Heatwave Temperature')

#plot UHI, raw
plt.subplot(2,2,2)

for i in range(0, compositeUHIDF.shape[0]) : 
    #print compositeUHIDF2.iloc[i].values[1:]
    plt.plot(x, compositeUHIDF.iloc[i].values[1:])
plt.plot( x, compositeUHIDF.mean(), color = 'k', linewidth = 3)
plt.axhline(0, linestyle = ':', color = 'k')

plt.xlabel('Event Day')
plt.ylabel('UHI ($\Delta^\circ$C)')
plt.title('Heatwave UHI Composite')

# plot temp, zeroed
plt.subplot(2,2,3)
for i in range(0, compositeTempDF.shape[0]) : 
    plt.plot(x, compositeTempDF.iloc[i].values[1:]- compositeTempDF.iloc[i].values[1])

#plt.plot( x, compositeTempDF.mean(), color = 'k', linewidth = 3)
plt.xlabel('Event Day')
plt.ylabel('$\Delta$ Temp ($\Delta ^\circ$C)')
plt.title('Heatwave Temperature')
plt.axhline(0, linestyle = ':', color = 'k')

#Plot UHI, zeroed out
plt.subplot(2,2,4)
for i in range(0, compositeUHIDF.shape[0]) : 
    #print compositeUHIDF2.iloc[i].values[1:]
    plt.plot(x, compositeUHIDF.iloc[i].values[1:] - compositeUHIDF.iloc[i].values[1])
plt.axhline(0, linestyle = ':', color = 'k')
plt.plot(x, compositeUHIDF.mean(), color = 'k', linewidth = 3)
plt.xlabel('Event Day')

plt.ylabel('$\Delta$ UHI ($\Delta ^\circ$C)')
plt.title('Heatwave UHI Composite, zeroed to 1')

plt.savefig(results_filepath + 'allcityHWuhicomposite.png')
