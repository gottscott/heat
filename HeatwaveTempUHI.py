
# coding: utf-8

# In[1]:

# This notebook calculates the relationship between temperature and UHI during heatwaves
# import libraries
#%matplotlib inline
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ulmo
import os
import scipy.stats
import matplotlib
import cartopy.crs as ccrs


# In[2]:

#results_filename = 'US_results_HW.csv'
#results_filepath = 'plots/HWscatter/'
results_filename = 'plots/version1/USHWresults.csv'
results_filepath = 'plots/version1/HWscattercleaned/'
inputData = 'USghcnpairs_stationlengths.csv'


# In[3]:

# functions

# plotting 
# define functions used for plotting 
def hw_scatter(x,y,title, xlabel, ylabel) : 
# plots x,y (need to be np array) and calculates and prints their best fit line
    ind = ~np.isnan(y) & ~np.isnan(x) # subset values that aren't NaNs
    m,b = np.polyfit(x[ind],y[ind],1)
    plt.scatter(x,y)
    plt.plot(x, m*x+b, color = 'black')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # annotate the linear reqression, y = mx+b
    plt.annotate('y = %.2f x + %.2f'%(m,b), xy=(.5, .9), xycoords='axes fraction',  horizontalalignment='left', verticalalignment='bottom')

# statistics
def pearsonr_autocorrelated(x, y):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.
    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input
    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)
    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    r = np.corrcoef(y[1:],y[0:-1])[0,1] #y.autocorr(1)
    n_prime = n*(1-r)/(1+r)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n_prime-2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r, prob

def ss(a, axis=0):
    """
    Squares each element of the input array, and returns the sum(s) of that.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        The axis along which to calculate. If None, use whole array.
        Default is 0, i.e. along the first axis.
    Returns
    -------
    ss : ndarray
        The sum along the given axis for (a**2).
    See also
    --------
    square_of_sums : The square(s) of the sum(s) (the opposite of `ss`).
    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([1., 2., 5.])
    >>> stats.ss(a)
    30.0
    And calculating along an axis:
    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> stats.ss(b, axis=1)
    array([ 30., 65.])
    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

def betai(a, b, x):
    """
    Returns the incomplete beta function.
    I_x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt)
    where a,b>0 and B(a,b) = G(a)*G(b)/(G(a+b)) where G(a) is the gamma
    function of a.
    The standard broadcasting rules apply to a, b, and x.
    Parameters
    ----------
    a : array_like or float > 0
    b : array_like or float > 0
    x : array_like or float
        x will be clipped to be no greater than 1.0 .
    Returns
    -------
    betai : ndarray
        Incomplete beta function.
    """
    x = np.asarray(x)
    x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return scipy.special.betainc(a, b, x)


# In[4]:

# Calculate the relationship between summertime  UHI and t_min, save it out for every city
pairs = pd.read_csv(inputData)
paired_df = pairs[(pairs['Urban brightness'] - pairs['Rural brightness'] > 30) & (pairs['Urban station'].str.contains('US'))]

numhw = 150
finalhw = 50

# pre-allocate
slopes           = np.zeros(paired_df.shape[0])
residuals        = np.zeros(paired_df.shape[0])
correlations     = np.zeros(paired_df.shape[0])
urban_start_date = np.zeros(paired_df.shape[0])
urban_end_date   = np.zeros(paired_df.shape[0])
rural_start_date = np.zeros(paired_df.shape[0])
rural_end_date   = np.zeros(paired_df.shape[0])
n_events         = np.zeros(paired_df.shape[0])
mean_rural_temp  = np.zeros(paired_df.shape[0])
mean_UHI         = np.zeros(paired_df.shape[0])
p_value          = np.zeros(paired_df.shape[0])


# In[5]:

for i in range(0,3):# paired_df.shape[0]): 
    city = paired_df.iloc[i]['City']
    city = unicode(city, errors = 'ignore')
    urbanID = paired_df.iloc[i]['Urban station']
    ruralID = paired_df.iloc[i]['Rural station']
    print city
    # Downloadd from NCDC the station data, using the station ID listed in station list
    urbandata = ulmo.ncdc.ghcn_daily.get_data(urbanID,
                                         as_dataframe=True, update = False)
    ruraldata = ulmo.ncdc.ghcn_daily.get_data(ruralID,
                                         as_dataframe=True, update = False)
    if ('TMIN' in urbandata.keys()) & ('TMIN' in ruraldata.keys()) & (np.intersect1d(urbandata['TMIN']['1985-01-01':].index, ruraldata['TMIN']['1985-01-01':].index).shape[0] > 900): 

        # Find the date at which they both start
        startdate = max(min(ruraldata['TMIN'].index), min(urbandata['TMIN'].index))

        # Calculate minimum daily thresholds starting from 1985
        rural_tmin = pd.to_numeric(ruraldata['TMIN']['1985-01-01':].value/10.) #rural tmin
        urban_tmin = pd.to_numeric(urbandata['TMIN']['1985-01-01':].value/10.) 
        #Get the hottest days based off tmin 
        hottestmin = rural_tmin.iloc[(-rural_tmin.values).argsort()[:numhw]] 
        minheatwaves = hottestmin

        # Make sure that events aren't duplicates 
        # get the time difference between events (sorted in temporal order, obviously)
        time_diff = (minheatwaves.sort_index().index.to_timestamp().values[1:] - minheatwaves.sort_index().index.to_timestamp().values[:-1]).astype('timedelta64[D]')
        # find where the events are not within 2 days of each other
        minheatwaves = minheatwaves.sort_index().iloc[np.where(time_diff > np.timedelta64(2, 'D'))]
        # Now the heatwaves are sorted in time order, but we want finalhw (50) of the most severe events. Save the hottest events
        minheatwaves = minheatwaves.sort_values(ascending=False).iloc[0:finalhw]

        rural_start_date[i] = rural_tmin[~np.isnan(rural_tmin)].index[0].year
        rural_end_date[i]   = rural_tmin[~np.isnan(rural_tmin)].index[-1].year
        urban_start_date[i] = urban_tmin[~np.isnan(urban_tmin)].index[0].year
        urban_end_date[i]   = urban_tmin[~np.isnan(urban_tmin)].index[-1].year

        #calculate UHI
        UHI = urban_tmin - rural_tmin#pd.to_numeric(urbandata['TMIN'].value/10.)[tmin.index] - tmin
        #UHImax = pd.to_numeric(urbandata['TMAX'].value/10.)[tmax.index] - tmax
        mean_UHI[i] = UHI.mean()
        
        x = rural_tmin[minheatwaves.index]#[np.logical_or(rural_tmin.index.month==6, rural_tmin.index.month==7, rural_tmin.index.month==8)] 
        y = UHI[minheatwaves.index]#[np.logical_or(UHI.index.month==6, UHI.index.month==7, UHI.index.month==8)] 
        ind = ~np.isnan(y) & ~np.isnan(x) # subset values that aren't NaNs
        
        if ind.sum() > 5: 
            mean_rural_temp[i] = x[ind].mean()

            plt.figure()
            handle = hw_scatter(x[ind],y[ind],'%s HW Temp vs UHI'%city, 'Rural Temp', 'UHI')
            plt.savefig(results_filepath + 'hwUHI%s%s.png'%(city.replace(" ", "")[0:5], 'min'))
            plt.close()

            try: 
                V = np.polyfit(x[ind],y[ind],1, full = True)
                C = pearsonr_autocorrelated(x[ind], y[ind])
                slopes[i] = V[0][0]
                residuals[i] = V[1][0]
                correlations[i] = C[0]#np.corrcoef(x[ind],y[ind])[0,1]
                p_value[i]      = C[1]
                n_events[i] = ind.values.sum()

            except TypeError : 
                slopes[i] = np.nan
                residuals[i] = np.nan
                correlations[i] = np.nan
    else : 
        slopes[i] = np.nan
        residuals[i] = np.nan
        correlations[i] = np.nan
        
        
    if np.mod(i,10) == 0 : 
        
        results_df = pd.DataFrame()
        results_df['City']  = paired_df['City']
        results_df['Slope'] = slopes
        results_df['Residual'] = residuals
        results_df['Correlation'] = correlations
        results_df['P-value'] = p_value
        results_df['Urban start date'] = urban_start_date
        results_df['Urban end date'] = urban_end_date
        results_df['Rural start date'] = rural_start_date
        results_df['Rural end date'] = rural_end_date
        results_df['Data points'] = n_events
        results_df['Mean UHI'] = mean_UHI
        results_df['Mean JJA Rural Temp'] = mean_rural_temp

        results_df.to_csv(results_filename)
        
results_df = pd.DataFrame()
results_df['City']  = paired_df['City']
results_df['Slope'] = slopes
results_df['Residual'] = residuals
results_df['Correlation'] = correlations
results_df['P-value'] = p_value
results_df['Urban start date'] = urban_start_date
results_df['Urban end date'] = urban_end_date
results_df['Rural start date'] = rural_start_date
results_df['Rural end date'] = rural_end_date
results_df['Data points'] = n_events
results_df['Mean UHI'] = mean_UHI
results_df['Mean JJA Rural Temp'] = mean_rural_temp
results_df.to_csv(results_filename)


# In[6]:

i


# In[ ]:




# In[7]:

# plot slope histograme
data = slopes
data = data[~np.isnan(data)]
plt.figure(figsize=[8,12])
plt.subplot(3,1,1)
plt.hist(data, 20)
plt.title('Slope of regression curve (UHI versus Temp for JJA)')
#plt.xlabel('Slope ($ \Delta ^{\circ} /^{\circ} $)')
plt.ylabel('Count')
plt.annotate('$ \mu $ = %2.2f'%data.mean(), xy=(.5, .9), xycoords='axes fraction',  horizontalalignment='left', verticalalignment='bottom')
plt.annotate('$ \sigma $ = %2.2f'%data.std(), xy=(.5, .8), xycoords='axes fraction',  horizontalalignment='left', verticalalignment='bottom')
plt.savefig(results_filepath + 'slopehistogram.png')

# plot correlation histogram
data = correlations
data = data[~np.isnan(data)]
plt.figure(figsize=[8,12])
plt.subplot(3,1,2)
plt.hist(data, 20)
plt.title('Correlation of UHI with Temp for JJA')
#plt.xlabel('Slope ($ \Delta ^{\circ} /^{\circ} $)')
plt.ylabel('Count')
plt.annotate('$ \mu $ = %2.2f'%data.mean(), xy=(.5, .9), xycoords='axes fraction',  horizontalalignment='left', verticalalignment='bottom')
plt.annotate('$ \sigma $ = %2.2f'%data.std(), xy=(.5, .8), xycoords='axes fraction',  horizontalalignment='left', verticalalignment='bottom')
plt.savefig(results_filepath + '/correlationhistogram.png')

# make a map of the slopes
fig = plt.figure(figsize=[15,15])
# Define colors 
cmap = matplotlib.cm.coolwarm
c = slopes
bounds = np.linspace(-1,1,11)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
marker_size = 75
# Define the cartopy basemaps
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ind = np.isnan(c)
plotHandle = ax.scatter(paired_df['Urban Lon'], paired_df['Urban Lat'],#x,y,
                        c = c, s = marker_size, transform=ccrs.Geodetic(),
                 cmap = cmap,
                 norm = norm)
# mask insignificantly correlated cities with a white circle
insig_inds = np.where(results_df['P-value']>0.05)

ax.scatter(paired_df['Urban Lon'].iloc[insig_inds], paired_df['Urban Lat'].iloc[insig_inds], 
        c = 'none', s = marker_size, transform=ccrs.Geodetic(), edgecolors='white',)

cbar1 = plt.colorbar(plotHandle, label = 'Slope', orientation='horizontal')
plt.title('Slope of UHI vs. Temp')
plt.savefig(results_filepath + 'slopemap.png')

