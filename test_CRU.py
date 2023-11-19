#%%

import os
from types import NoneType
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats import pearsonr
from scipy import stats
from scipy import linalg
import scipy.signal as signal
from scipy.stats import chi2
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL 
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from sklearn.decomposition import PCA as pca
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import grangercausalitytests
import glob

warnings.filterwarnings('ignore')

def moving_mean(array, win, win_type='boxcar', center=False):
    return np.array(pd.DataFrame(array).rolling(win, win_type=win_type, center=center).mean())

def ortho_rotation(lam, method='varimax',gamma=None,
                   eps=1e-6, itermax=100):
    """
    Return orthogal rotation matrix
    TODO: - other types beyond 
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0
    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0
    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new
    return R

def extract_ENSO_months(df, events, months):

    """
    Extract ENSO months from variable DataFrame based on DataFrame constructed with function -> define_ENSO_months 

    Parameters
    ----------

    df : DataFrame: Features or pixels of variable as columns in a df with date index,
    events : DataFrame: Df with at least columns of start_date and end_date of each event of the same ENSO phase.
    months : list: Months selected to extract.

    Returns:
    --------

    DataFrame: of variable based on ENSO months.
    """

    enso_months = pd.DataFrame()

    for i in events.index:
        mask_event = (df.index >= events['start_date'].loc[i].to_timestamp()) & \
                (df.index <= events['end_date'].loc[i].to_timestamp() + timedelta(30)) # se le añaden 30 dias para que coja el ultimo mes
        enso_months_tmp = df[mask_event].loc[df[mask_event].index.month_name().isin(months)]
        enso_months = pd.concat([enso_months, enso_months_tmp], axis=0)

    return enso_months

def monthly_composites(df, events, months, lag = 0, type = "total"):

    """
    Composites of variable in df based on each phase of ENSO with a specific lag.

    Parameters:
    -----------

    df: DataFrame: DataFrame of Variable to calculate composites.
    events : DataFrame: Df with at least columns of start_date and end_date of each event of the same ENSO phase.
    months : list: Months selected to extract.
    lag: int: Lag of variable with respect to ENSO.
    type: str: total or season

    Returns:
    --------
    
    DataFrame: of features composites total or seasonal.
    """
    df_event = extract_ENSO_months(df.shift(-1*lag), events, months)
    df_event = df_event.groupby(df_event.index.month).mean()

    if type == "total":
        df_event_total = df_event.mean(axis=0).values
    elif type == "season":
        df_event.index = pd.to_datetime(df_event.index, format="%m")
        df_event_total = df_event.resample('QS-DEC').mean()
        df_event_total= df_event_total.groupby(df_event_total.index.month).mean()
    elif type == "month":
        df_event_total = df_event
    else:
        print("Check type")
    
    return df_event_total

def define_ENSO_phase(df, thrsld, min_duration):

    """
    Define ENSO events from an index based on a threshold and a minimun consecutive months.

    Parameters:
    -----------

    df: DataFrame or Series: with ENSO index (ONI in this case).
    thrsld: float: Threshold value of SST (3.4 region in this case) to define ENSO phase.
    min_duration: int: Number of consecutive months to define ENSO phase.

    Returns:
    --------

    3 DataFrames of each ENSO phase -> El Niño, La Niña y Neutral with the following columns:
    event_type, start_date, envd_date, duration_months, max_value, min_value.
    """

    # Define ONI thresholds for El Niño and La Niña
    oni_el_nino_threshold = thrsld
    oni_la_nina_threshold = -1*thrsld

    ONI_gracee = pd.DataFrame(df)
    # Create a new column in the DataFrame to indicate ENSO phase
    ONI_gracee['enso_phase'] = pd.cut(ONI_gracee[df.name], 
                                    bins=[-999, oni_la_nina_threshold, oni_el_nino_threshold, 999],
                                    labels=['la_nina', 'neutral', 'el_nino'])

    # Create a new column to identify the start of each event
    ONI_gracee['event_start'] = (ONI_gracee['enso_phase'] != ONI_gracee['enso_phase'].shift(1)) & \
                            ((ONI_gracee['enso_phase'] == 'el_nino') | (ONI_gracee['enso_phase'] == 'la_nina') \
                            | (ONI_gracee['enso_phase'] == 'neutral'))

    # Group the data by event and extract information about each event
    events = ONI_gracee.groupby((ONI_gracee['event_start'] == True).cumsum()).apply(
        lambda x: pd.Series({
            'event_type': x['enso_phase'].iloc[0],
            'start_date': x.index[0],
            'end_date': x.index[-1],
            'duration_months': len(x),
            'max_value': x[df.name].max(),
            'min_value': x[df.name].min()
        })
    )

    # Filter the events to include only El Niño and La Niña events with at least 5 consecutive months, the rest in neutral

    events.loc[events.duration_months < min_duration, "event_type"] = "neutral"

    el_nino_events = events[events['event_type'] == 'el_nino']
    la_nina_events = events[events['event_type'] == 'la_nina']
    neutral_events = events[events['event_type'] == 'neutral']

    return el_nino_events, la_nina_events, neutral_events

def lag_composites(array, index_dates, ENSO_index, lag, type):

    """
    Composite of ENSO phases.

    Parameters:
    -----------

    array: numpy.ndarray: Matrix with variable data with shape (n_times, features)
    index_dates: DatetimeIndex: Monthly index dates with dtype datetime64.
    ENSO_index: DataFrame or Series: With ENSO index (ONI in this case).
    lag: int: Monthly lag of array with respect to ENSO.
    type: str: total or season

    Check duration months and threshold for determine ENSO phase.

    Returns:
    --------
    numpy.ndarray: Composites array with each ENSO phase.
    if type == total : shape = (features, 3 -> each phase [El Niño, La Niña y Neutral])
    if type == season : shape (features, 12 -> [DFJ, MAM, JJA, SON] for each phase [El Niño, La Niña y Neutral])
    """

    months = ['January', "February", "March", "April", "May", "June", "July", "August", "September", \
          "October", "November", "December"]
    
    duration_months = 5
    threshold = 0.49

    df = pd.DataFrame(array, index=index_dates)

    el_nino_events, la_nina_events, neutral_events = define_ENSO_phase(ENSO_index, threshold, duration_months)

    df_nino = monthly_composites(df, el_nino_events, months, lag, type)
    df_nina = monthly_composites(df, la_nina_events, months, lag, type)
    df_neutral = monthly_composites(df, neutral_events, months, lag, type)

    if type == "total":
        composites = np.array([df_nino, df_nina, df_neutral])
    elif type == "season":
        composites = np.array(pd.concat([df_nino, df_nina, df_neutral], axis=0))
    elif type == "month":
        composites = np.array([df_nino, df_nina, df_neutral])
    else:
        print("check type")

    return composites.T

def plot_enso_years(df, y, ax=None, phase_values=[-1.,0.,1.], colors=['skyblue','white','salmon'], fill_params={}):
    if ax is None:
        ax = plt.gca()
    for phase in phase_values:
        mask = df.phase == phase
        ax.fill_between(df.index, np.nanmin(df[y].values), np.nanmax(df[y].values),\
             where=mask, facecolor=colors[int(phase)+1], **fill_params)
    return(ax)

def ENSO_category(df, oni):

    months = ['January', "February", "March", "April", "May", "June", "July", "August", "September", \
          "October", "November", "December"]
    duration_months = 5
    threshold = 0.499

    el_nino_events, la_nina_events, neutral_events = define_ENSO_phase(oni, threshold, duration_months)

    df_nino = extract_ENSO_months(df, el_nino_events, months)
    df_nina = extract_ENSO_months(df, la_nina_events, months)
    df_neutral = extract_ENSO_months(df, neutral_events, months)

    df_nino["phase"] = pd.Series([1 for x in range(len(df_nino.index))], index=df_nino.index)
    df_nina["phase"] = pd.Series([-1 for x in range(len(df_nina.index))], index=df_nina.index)
    df_neutral["phase"] = pd.Series([0 for x in range(len(df_neutral.index))], index=df_neutral.index)

    df_enso = pd.concat([df_nino, df_nina, df_neutral])
    df_enso.sort_index(inplace=True)
    return df_enso

def HPA_zones(data, mask):
    """
    Returns data divided by HPA zones
    """
    n = data.shape[0]
    data_tmp = mask*data.reshape(n, 24,18)
    North = (data_tmp[:, :10, :]).reshape(data_tmp.shape[0], 10*18)

    Central = np.concatenate((data_tmp[:, 10:17, :8].reshape(n, int(data_tmp[:, 10:17, :8].size / n)), \
                          data_tmp[:, 10:19, 8:].reshape(n, int(data_tmp[:, 10:19, 8:].size / n))), axis=1)

    South = np.concatenate((data_tmp[:, 19:, 8:].reshape(n, int(data_tmp[:, 19:, 8:].size / n)), \
                          data_tmp[:, 17:, :8].reshape(n, int(data_tmp[:, 17:, :8].size / n))), axis=1)
    
    return North, Central, South

def read_oni(data, dates, standard=False):

    data.columns=range(0,12)
    t = data_oni.stack()
    year, month = t.index.get_level_values(0).values, t.index.get_level_values(1).values
    t.index = pd.PeriodIndex(year=year, month=month+1, freq='M', name='Fecha')

    if standard == True:
        return estandarizar(pd.Series(t.loc[dates[0]:dates[1]],name='ONI'))
    if standard == False:
        return pd.Series(t.loc[dates[0]:dates[1]],name='ONI')


mask = np.genfromtxt('data/mask.csv', delimiter = ',')
data_oni = pd.read_csv('data/ONI.csv', index_col=0)

list_cru = sorted(glob.glob('cru*'))

#precip_total = np.empty((0, 24, 18))
precip_total = np.empty((0, 24, 30))
times = []
for file in list_cru:

    cru_prep = Dataset(file)

    precip = np.array(cru_prep.variables['pre'])
    time = np.array(cru_prep.variables['time'])

    lon_ ,lat_ = cru_prep.variables['lon'][:], cru_prep.variables['lat'][:]
    lon_ = 360 + lon_

    latgrN = np.where(lat_ == 44.25)[0][0] #44.25, #43.75
    latgrS = np.where(lat_ == 32.25)[0][0] #32.25  #31.75
    longrE = np.where(lon_ == 266.75)[0][0] #263.75 263.375
    longrW = np.where(lon_ == 251.75)[0][0] #254.75  254.375

    lat = lat_[latgrS:latgrN]
    lat = lat[::-1]
    lon = lon_[longrW:longrE]

    precip_ = precip[:,latgrS:latgrN,longrW:longrE]  #lwe*scale
    precip = precip_[:,::-1,:]

    precip_total = np.concatenate([precip_total, precip])
    times = np.concatenate([times, time])

dates = np.array([datetime(1900,1,1) + timedelta(days = int(i)) for i in times])

precip_N, precip_C, precip_S = HPA_zones(precip_total, mask)

orden_subplots = [precip_N, precip_C, precip_S]

labels_zones = ['North HPA', 'Central HPA', 'South HPA']
x_ticks_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(3, figsize=(12, 10), dpi=200, sharex=True)

for ix in range(3):
    subplt = orden_subplots[ix]
    if subplt.ndim > 2:
        subplt = orden_subplots[ix].reshape(orden_subplots[ix].shape[0], orden_subplots[ix].shape[1]*orden_subplots[ix].shape[2])
    else:
        pass
    composites = lag_composites(subplt, dates, read_oni(data_oni, ['1981-01','2022-12']), 1, "month")
    ax.flat[ix].plot(np.nanmean(composites, axis=0)[:, 0], c='navy', label='Niño')
    ax.flat[ix].plot(np.nanmean(composites, axis=0)[:, 1], c='firebrick', label='Niña')
    ax.flat[ix].plot(np.nanmean(composites, axis=0)[:, 2], c='gray', label='Neutral')
    ax.flat[ix].plot(np.nanmean(composites, axis=(0,2)), c='k', ls='--')
    ax.flat[ix].set_title(labels_zones[ix])
    ax.flat[ix].spines[['right', 'top']].set_visible(False)
    ax.flat[ix].grid(alpha=0.2)
    ax.flat[ix].set_ylim(0, 110)


#%%

# total_composites = lag_composites(precip_detrend.reshape(499, 24*18), pd.date_range('1981-03-01','2022-10-01', freq='M'), \
#                                   read_oni(data_oni, ['1981-03','2022-09']), 1, "month")

fig, ax = plt.subplots(figsize=(12, 8),subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)


ax.contourf(lon,lat, np.mean(precip_total, axis=0), cmap='GnBu',vmax=100,vmin=0)
ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
                name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
#ax.contourf(lon,lat, eof*mask, cmap='bwr_r',vmax=vals,vmin=-1*vals)