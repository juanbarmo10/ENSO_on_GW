#%%

import os
import warnings
from datetime import datetime, timedelta

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.signal as signal
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.dates import DateFormatter
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
import scipy.stats as stats

warnings.filterwarnings('ignore')


def read_grace(data_grace, variable='lwe_thickness'):


    lat_grace_ = np.array(data_grace.variables['lat'][:])
    lon_grace_ = np.array(data_grace.variables['lon'][:])
    LWE = np.array(data_grace.variables[variable][:])
    time_grace_ = np.array(data_grace.variables['time'][:]) #-22
    time_grace = time_grace_.astype('float64')


    latgrN = np.where(lat_grace_ == 44.25)[0][0] #44.25, #43.75
    latgrS = np.where(lat_grace_ == 32.25)[0][0] #32.25  #31.75
    longrE = np.where(lon_grace_ == 263.75)[0][0] #263.75 263.375
    longrW = np.where(lon_grace_ == 254.75)[0][0] #254.75  254.375

    lat_grace = lat_grace_[latgrS:latgrN]
    lat_grace = lat_grace[::-1]
    lon_grace = lon_grace_[longrW:longrE]

    TWS_ = LWE[:,latgrS:latgrN,longrW:longrE]  #lwe*scale
    TWS_incomplete = TWS_[:-1,::-1,:]


    TWS = TWS_incomplete.copy()

    if variable == 'lwe_thickness':
        # With index positions of missing months in GRACE NaN values are inserted to extend GRACE to the same size of GLDAS

        missing_months = [2,3,14,105,110,116,121,131,136,137,142,147,152,158,162,\
                        163,168,173,174,178,183,184,185,186,187,188,189,190,191,192,\
                            193,196,197]

        for i in range(len(missing_months)):
            missing = missing_months[i]
            TWS = np.insert(TWS, missing, np.nan, axis=0)
            
    print(TWS.shape)

    return TWS, lat_grace, lon_grace

def read_gldas(data_gldas, ix_ini, init_date):

    lat_gldas_ = np.array(data_gldas.variables['lat'][:])
    lon_gldas_ = np.array(data_gldas.variables['lon'][:])
    time_gldas = np.array(data_gldas.variables['time'][:])

    try:
        SWE_ = np.array(data_gldas.variables['SWE_inst'][:])
        SM1 = np.array(data_gldas.variables['SoilMoi0_10cm_inst'][:])
        SM2 = np.array(data_gldas.variables['SoilMoi10_40cm_inst'][:])
        SM3 = np.array(data_gldas.variables['SoilMoi40_100cm_inst'][:])
        SM4 = np.array(data_gldas.variables['SoilMoi100_200cm_inst'][:])

        SM_ = SM1 + SM2 + SM3 + SM4

        dates_gldas = np.array([init_date + timedelta(days = time_gldas[i]) for i in range(len(time_gldas))])

    except:
        SWE_ = np.array(data_gldas.variables['SWE'][:])
        dates_gldas = np.array([init_date + timedelta(hours = time_gldas[i]) for i in range(len(time_gldas))])

        try:
            SM_ = np.array(data_gldas.variables['SoilM_0_200cm'][:])
        except:
            SM_ = np.array(data_gldas.variables['SoilM_total'][:])

    latglN = np.where(lat_gldas_ == 44.25)[0][0] # 44.25 43.75
    latglS = np.where(lat_gldas_ == 32.25)[0][0] # 32.25 31.75
    longlE = np.where(lon_gldas_ == 263.75)[0][0] #263.75
    longlW = np.where(lon_gldas_ == 254.75)[0][0] #264.75

    lat_gldas = lat_gldas_[latglN:latglS]#[::-1]
    lon_gldas = lon_gldas_[longlW:longlE]

    SWE_ = SWE_[:,latglN:latglS,longlW:longlE]# / 10
    SM_ = SM_[:,latglN:latglS,longlW:longlE]# / 10

    SM = SM_[ix_ini:,::-1,:]
    SWE = SWE_[ix_ini:,::-1,:]

    SM_model = SM_[:-23,::-1,:]
    SWE_model = SWE_[:-23,::-1,:]

    print(lat_gldas)
    print(lon_gldas)

    print('Shapes normales')
    print(SM.shape)
    print(SWE.shape)

    print('Shapes model')
    print(SM_model.shape)
    print(SWE_model.shape)


    return SM, SWE, SM_model, SWE_model, dates_gldas

def read_model(data_model):

    lat_model_ = np.array(data_model.variables['lat'][:])
    lon_model_ = np.array(data_model.variables['lon'][:])
    lwe = np.array(data_model.variables['rec_ensemble_member'][:])
    time_grace_ = np.array(data_model.variables['time'][:]) 
    mask = np.genfromtxt('data/mask.csv', delimiter = ',')

    dates_model = np.array([datetime(1901,1,1) + timedelta(days = time_grace_[i]) for i in range(len(time_grace_))])

    lon_model_ = 360 + lon_model_

    latgrN = np.where(lat_model_ == 44.25)[0][0] #44.25, #43.75
    latgrS = np.where(lat_model_ == 32.25)[0][0] #32.25  #31.75
    longrE = np.where(lon_model_ == 263.75)[0][0] #263.75 263.375
    longrW = np.where(lon_model_ == 254.75)[0][0] #254.75  254.375

    lat_model_ = lat_model_[latgrS:latgrN]
    lat_model = lat_model_[::-1]
    lon_model = lon_model_[longrW:longrE]

    TWS_ = lwe[:,latgrS:latgrN,longrW:longrE]  #lwe*scale
    TWS_model = TWS_[:,::-1,:]/10

    print(TWS_model.shape)

    return TWS_model

def read_oni(data, dates, standard=False):

    data.columns=range(0,12)
    t = data.stack()
    year, month = t.index.get_level_values(0).values, t.index.get_level_values(1).values
    t.index = pd.PeriodIndex(year=year, month=month+1, freq='M', name='Fecha')

    if standard == True:
        return estandarizar(pd.Series(t.loc[dates[0]:dates[1]],name='ONI'))
    if standard == False:
        return pd.Series(t.loc[dates[0]:dates[1]],name='ONI')

def HPA_zones(data, mask):
    """
    Returns data divided by HPA zones
    """
    n = data.shape[0]
    data_tmp = mask*data.reshape(n, 24,18)
    #North = data_tmp[:, :10, :]
    North = (data_tmp[:, :10, :]).reshape(data_tmp.shape[0], 10*18)
    Central = np.concatenate((data_tmp[:, 10:17, :8].reshape(n, int(data_tmp[:, 10:17, :8].size / n)), \
                          data_tmp[:, 10:19, 8:].reshape(n, int(data_tmp[:, 10:19, 8:].size / n))), axis=1)

    South = np.concatenate((data_tmp[:, 19:, 8:].reshape(n, int(data_tmp[:, 19:, 8:].size / n)), \
                          data_tmp[:, 17:, :8].reshape(n, int(data_tmp[:, 17:, :8].size / n))), axis=1)
    
    return North, Central, South

def detrend(serie, axis):
    serie_out = signal.detrend(serie, axis, type='linear')
    return serie_out

def deseasonal(array_in):
    
    trend = np.zeros_like(array_in)
    seasonal = np.zeros_like(array_in)
    residual = np.zeros_like(array_in)
    
    for la in range(array_in.shape[1]):
        for lo in range(array_in.shape[2]):
            stl = STL(array_in[:,la,lo], period=12,seasonal=13)
            res = stl.fit()
            seasonal[:,la,lo] = res.seasonal
            trend[:,la,lo] = res.trend
            residual[:, la, lo] = res.resid
            
    return trend, seasonal, residual

def moving_mean(array, win, win_type='boxcar', center=False):
    return np.array(pd.DataFrame(array).rolling(win, win_type=win_type, center=center).mean())

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

def monthly_composites(df, events, months, type = "total"):

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
    #df_event = extract_ENSO_months(df.shift(-1*lag), events, months)
    df_event = extract_ENSO_months(df, events, months)
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

    if isinstance(array, np.ndarray):
        df = pd.DataFrame(array, index=index_dates)
    elif isinstance(array, pd.DataFrame):
        df = array
    else:
        print('Check data type')

    el_nino_events, la_nina_events, neutral_events = define_ENSO_phase(ENSO_index.shift(lag), threshold, duration_months)

    df_nino = monthly_composites(df, el_nino_events, months, type)
    df_nina = monthly_composites(df, la_nina_events, months, type)
    df_neutral = monthly_composites(df, neutral_events, months, type)

    if type == "total":
        composites = np.array([df_nino, df_nina, df_neutral])
    elif type == "season":
        composites = np.array(pd.concat([df_nino, df_nina, df_neutral], axis=0))
    elif type == "month":
        composites = np.array([df_nino, df_nina, df_neutral])
    else:
        print("check type")

    return composites.T

def plot_map(out, matrix, lat, lon, num, title,labeli, vals, grid=(1,4), fig_size=(8,4), cmap='bwr_r', points=None):
    
    csfont = {'fontname':'Arial'}
    reader = shpreader.Reader('data/HPA/HP.shp')
    lon_ = lon - 360
    fig ,ax= plt.subplots(grid[0], grid[1], figsize=fig_size,sharex=True, sharey=True,\
                          dpi=200, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    if num > 1:
        for i in range(num):
            
            eof = matrix[i, :].reshape(lat.shape[0], lon.shape[0])

            ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
                    facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
            ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
                name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
            ax.flat[i].contourf(lon,lat, eof*mask, cmap=cmap,vmax=vals,vmin=-1*vals)

            gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 4, dtype=int),
                        draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 3, dtype=int))
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.ylabel_style = {'fontname':'Arial', 'fontsize': 10}
            gl.xlabel_style = {'fontname':'Arial', 'fontsize': 10}        

            ax.flat[i].set_title(title[i],loc='left',fontsize=12, **csfont)

            if i not in np.arange(0, grid[0]*grid[1], grid[1], dtype=int):
                gl.ylabels_left = False

            if i not in np.arange(grid[0]*grid[1]-grid[1], grid[0]*grid[1], dtype=int):
                gl.xlabels_bottom = False
                
            if points is not None:
                points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
                ax.flat[i].scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())

    else:
        eof = matrix.reshape(lat.shape[0], lon.shape[0])
        ax.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
                    facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
        ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
            name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
        ax.contourf(lon,lat, eof*mask, cmap=cmap,vmax=vals,vmin=-1*vals)

        gl = ax.gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3, dtype=int),
                    draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 3, dtype=int))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'fontname':'Arial', 'fontsize': 10}
        gl.xlabel_style = {'fontname':'Arial', 'fontsize': 10}        

        ax.set_title(title,loc='left',fontsize=12, **csfont)
            
        if points is not None:
            points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
            ax.scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())
            
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.1, -0.05, 0.8, 0.01])
    cbar_ax.grid(False)
    cbar_ax.axis('off')
    [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
    cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
            norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
        fraction=5, aspect=80,extend='both', )

    cbar.ax.get_xaxis().labelpad = 5
    cbar.ax.set_xlabel(labeli, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches='tight')

def deseasonal_df(df):

    df_trend = pd.DataFrame(index=df.index, columns=df.columns)
    df_seasonal = pd.DataFrame(index=df.index, columns=df.columns)
    df_resid = pd.DataFrame(index=df.index, columns=df.columns)

    for column in df.columns:
        stl = STL(df[column], period=12,seasonal=13)
        res = stl.fit()
        df_seasonal[column] = res.seasonal
        df_trend[column] = res.trend
        df_resid[column] = res.resid
            
    return df_trend, df_seasonal, df_resid

def red_noise_spectrum(serie, Freq, chunksize, confidence, scaling='density'):
    alpha = signal.correlate(serie,serie,'full')
    alpha = alpha/np.max(alpha)
    alpha = alpha[int(np.size(alpha)/2)+1]

    dof = ((2*len(serie) - chunksize/2) // chunksize) + 1
   # print(len(serie), dof)
    #dof = 2*n_chunks
    fstat = stats.f.ppf(confidence,dof,1000)

    if scaling == 'spectrum':
        Te = -1./np.log(alpha)
        rnoise = 2.8*2*Te/(1+(Te**2)*((Freq)*2*np.pi)**2)
        rnoise = rnoise / np.sum(rnoise)

        #np.sum rnoise is = variance of serie
        
    elif scaling == 'density':
        rspec = []
        for i in np.arange(1,len(Freq)+2,1):
            rspec.append((1.-alpha*alpha)/(1.-2.*alpha*np.cos(np.pi*(i-1.)/chunksize)+alpha*alpha))
            rnoise = np.array(rspec)
        rnoise = rnoise[:-1]
        #print(np.var(rnoise), np.sum(rnoise))
    else:
        print('Scaling error')
    return fstat*rnoise

def FFT(serie):
    FTT = np.fft.fft(serie, norm='forward')
    Freq = np.fft.fftfreq(len(serie), 1)
    Fpos = Freq[ Freq >= 0 ]
    Power = np.abs(FTT) ** 2
    
    df = Freq[1] - Freq[0]
    P = np.sum(Power) * df

    periodos = 1/Fpos
    
    F, S = signal.welch(serie, window='boxcar', nperseg=len(serie), scaling='spectrum')
    df2 = F[1] - F[0]
    P2 = np.sum(S) * df2
    
    return (periodos, Power[:len(Fpos)], F, S)

"**************************************************************************************************"

# GRACE

data_grace = Dataset('data/GRCTellus.JPL.200204_202211.GLO.RL06.1M.MSCNv03CRI.nc')
mask = np.genfromtxt('data/mask.csv', delimiter = ',')
dates_grace = pd.date_range('2002-04-15','2022-11-17', freq='M')
_, lat_grace, lon_grace = read_grace(data_grace)
time_grace = np.array([datetime(2002,1,1) + timedelta(days = times) for times in np.array(data_grace["time"])])
with open('data/tws_fill.npy', 'rb') as f:
    TWS = np.load(f)*10

ix_nan = np.argwhere(np.isnan(_[:,0,0]))[:, 0]
mask_nan = np.in1d(range(_.shape[0]), ix_nan)
mask_nan_m = mask_nan[:208]
mask_nan = mask_nan[2:-1]

# GLDAS

data_gldas = Dataset('data/GLDAS_NOAH_1979-2021.nc')
data_gldas2 = Dataset('data/GLDAS_NOAH_2000-2022.nc')
_, _, SM_model, SWE_model, dates_gldas = read_gldas(data_gldas, 279, datetime(1948,1,1))
SM, SWE, _, _, dates_gldas2 = read_gldas(data_gldas2, 28, datetime(2000,1,1))

# NLDAS !!! 
data_nldas_noah = Dataset('data/NLDAS_NOAH_1979-2023.nc')
SM_noah, SWE_noah, _, _, dates_nldas = read_gldas(data_nldas_noah, 279, datetime(1979,1,1))
SM_noah = SM_noah[:-12]
SWE_noah = SWE_noah[:-12]

data_nldas_vic = Dataset('data/NLDAS_VIC_1979-2023.nc')
SM_vic, SWE_vic, _, _, dates_nldas = read_gldas(data_nldas_vic, 279, datetime(1979,1,1))
SM_vic = SM_vic[:-12]
SWE_vic = SWE_vic[:-12]

data_nldas_mosaic = Dataset('data/NLDAS_MOSAIC_1979-2023.nc')
SM_mosaic, SWE_mosaic, _, _, dates_nldas = read_gldas(data_nldas_mosaic, 279, datetime(1979,1,1))
SM_mosaic = SM_mosaic[:-12]
SWE_mosaic = SWE_mosaic[:-12]

SM_nldas = np.average([SM_mosaic, SM_vic, SM_noah], axis=0)
SWE_nldas = np.average([SWE_mosaic, SWE_vic, SWE_noah], axis=0)

# MODELO

data_model = Dataset('data/GRACE_REC_v03_JPL_ERA5_monthly_ens082.nc')
TWS_model_best = read_model(data_model)
time_model = np.array(data_model.variables['time'][:]) 
dates_model = np.array([datetime(1901,1,1) + timedelta(days = time_model[i]) for i in range(len(time_model))])
dates_model = pd.date_range('1979-01-15','2019-08-17', freq='M')
dates_model_grace = dates_model[279:]
with open('data/fit_model.npy', 'rb') as f:
    TWS_model = np.load(f)*10

# CHIRPS

data_chirps = Dataset('data/chirps_monthly_gracegrid.nc')
precip = read_grace(data_chirps, 'precip')[0] 
time_chirps = np.array(data_chirps.variables['time'][:]) 
dates_chirps = np.array([datetime(1980,1,1) + timedelta(days = int(i)) for i in time_chirps])
dates_chirps = dates_chirps[:-2]
precip = precip[:-2]

# ONI

data_oni = pd.read_csv('data/ONI.csv', index_col=0)
ONI_model = read_oni(data_oni, ['1979-01','2019-07'], standard=False)
ONI_grace = read_oni(data_oni, ['2002-04','2022-10'], standard=False)
ONI_precip = read_oni(data_oni, ['1981-03','2022-09'], standard=False)

# ANOMALIES

TWSa = TWS - np.nanmean(TWS, axis=0)
SMa, SMa_model = SM_nldas - np.nanmean(SM_nldas, axis=0), SM_model - np.nanmean(SM_model, axis=0)
SWEa, SWEa_model = SWE_nldas - np.nanmean(SWE_nldas, axis=0), SWE_model - np.nanmean(SWE_model, axis=0)

GWa = TWSa - SMa - SWEa    # Ecuacion 1

GWa_m = moving_mean(GWa.reshape(247, 24*18), 3, center=True)[2:-1]
TWSa_m = moving_mean(TWSa.reshape(247, 24*18), 3, center=True)[2:-1]
precip_m = moving_mean(precip.reshape(502, 24*18), 3, center=True)[2:-1]
SMa_m = moving_mean(SMa.reshape(247, 24*18), 3, center=True)[2:-1]
SWEa_m = moving_mean(SWEa.reshape(247, 24*18), 3, center=True)[2:-1]

# GWa_m = GWa[2:-1]
# precip_m = precip[2:-1]
# TWSa_m = TWSa[2:-1]
# SMa_m = SMa[2:-1]
# SWEa_m = SWEa[2:-1]

dates_grace = dates_grace[2:-1]
dates_chirps = pd.date_range("1981-01", "2022-11", freq="M")[2:-1]


GWa_detrend = detrend(GWa_m, axis=0)
TWSa_detrend = detrend(TWSa_m, axis=0)
precip_a = precip_m - np.nanmean(precip_m, axis=0)
precip_detrend = detrend(precip_a, axis=0)
SMa_detrend = detrend(SMa_m, axis=0)

# CUT VARIABLES IN HPA ZONES

# GWa_N, GWa_C, GWa_S = HPA_zones(GWa_m, mask)
# TWSa_N, TWSa_C, TWSa_S = HPA_zones(TWSa_m, mask)
# precip_N, precip_C, precip_S = HPA_zones(precip_m, mask)
# SMa_N, SMa_C, SMa_S = HPA_zones(SMa_m, mask)

GWa_N, GWa_C, GWa_S = HPA_zones(GWa_detrend, mask)
TWSa_N, TWSa_C, TWSa_S = HPA_zones(TWSa_detrend, mask)
precip_N, precip_C, precip_S = HPA_zones(precip_detrend, mask)
SMa_N, SMa_C, SMa_S = HPA_zones(SMa_detrend, mask)

# REMOVE SEASONAL COMPONENT

GWa_residual = deseasonal(GWa_detrend.reshape(244, 24, 18))[0] + deseasonal(GWa_detrend.reshape(244, 24, 18))[2] # sumar tendencia y residual
TWSa_residual = deseasonal(TWSa_detrend.reshape(244, 24, 18))[0] + deseasonal(TWSa_detrend.reshape(244, 24, 18))[2]
precip_residual = deseasonal(precip_detrend.reshape(499, 24, 18))[0] + deseasonal(precip_detrend.reshape(499, 24, 18))[2]
SMa_residual = deseasonal(SMa_detrend.reshape(244, 24, 18))[0] + deseasonal(SMa_detrend.reshape(244, 24, 18))[2]

GWa_residual = np.reshape(GWa_residual, (244, 432))
TWSa_residual = np.reshape(TWSa_residual, (244, 432))
precip_residual = np.reshape(precip_residual, (499, 432))
SMa_residual = np.reshape(SMa_residual, (244, 432))

# MODEL
# HERE WE MUST EXTRACT SEASONAL COMPONENT FROM GLDAS DATA IN ORDER TO OBTAIN GWA FROM MODEL

SWEa_model_residual = deseasonal(SWEa_model)[0] + deseasonal(SWEa_model)[2] # sumar tendencia y residual
SWEa_model_residual = np.reshape(SWEa_model_residual, (487, 24 * 18))
SMa_model_residual = deseasonal(SMa_model)[0] + deseasonal(SMa_model)[2] # sumar tendencia y residual
SMa_model_residual = np.reshape(SMa_model_residual, (487, 24 * 18))

GWa_model = TWS_model - SMa_model_residual - SWEa_model_residual

TWS_model_m = moving_mean(TWS_model, 3, center=True)[2:-1]
GWa_model_m = moving_mean(GWa_model, 3, center=True)[2:-1]
SMa_model_m = moving_mean(SMa_model_residual, 3, center=True)[2:-1]
SWEa_model_m = moving_mean(SWEa_model_residual, 3, center=True)[2:-1]

dates_model = dates_model[2:-1]

GWa_detrend_model = detrend(GWa_model_m, axis=0)

# TWS_model_m = TWS_model[2:-1]
# GWa_model_m = GWa_model[2:-1]
# SMa_model_m = SMa_model_residual[2:-1]
# SWEa_model_m = SWEa_model_residual[2:-1]
# dates_model = dates_model[2:-1]

# CUT VARIABLES IN HPA ZONES IN THE MODEL

GWa_model_N, GWa_model_C, GWa_model_S = HPA_zones(GWa_model_m.reshape(484, 24, 18), mask)
TWS_model_N, TWS_model_C, TWS_model_S = HPA_zones(TWS_model_m.reshape(484, 24, 18), mask)
SMa_model_N, SMa_model_C, SMa_model_S = HPA_zones(SMa_model_m.reshape(484, 24, 18), mask)

#%%

def red_noise_spectrum(serie, Freq, chunksize, confidence, scaling='density'):

    alpha = signal.correlate(serie,serie,'full')
    alpha = alpha/np.max(alpha)
    alpha = alpha[int(np.size(alpha)/2)+1]

    dof = ((2*len(serie) - chunksize/2) // chunksize)# + 1
    # print(len(serie), dof)
    #dof = 2*n_chunks
    fstat = stats.f.ppf(confidence,dof,1000)

    if scaling == 'density':
        Te = -1./np.log(alpha)
        rnoise = 2.8*2*Te/(1+(Te**2)*((Freq)*2*np.pi)**2)
        rnoise = rnoise / np.sum(rnoise)

        #np.sum rnoise is = variance of serie
        
    elif scaling == 'spectrum':
        rspec = []
        for i in np.arange(1,len(Freq)+2,1):
            rspec.append((1.-alpha*alpha)/(1.-2.*alpha*np.cos(np.pi*(i-1.)/chunksize)+alpha*alpha))
            rnoise = np.array(rspec)
        rnoise = rnoise[:-1]
        #print(np.var(rnoise), np.sum(rnoise))
    else:
        print('Scaling error')
    return fstat*rnoise


# PCscore_selected_m = moving_mean(PCscore_selected, 3)
# PCscore_model_selected_m = moving_mean(PCscore_model_selected, 3)

# selected = PCscore[:,[0,2,3]]
# selected_model = PCscore_model[:,[0,2,3]]

precip_N = pd.DataFrame(precip_N, index=dates_chirps)
precip_C = pd.DataFrame(precip_C, index=dates_chirps)
precip_S = pd.DataFrame(precip_S, index=dates_chirps)

TWSa_N = pd.DataFrame(TWSa_N, index=dates_grace)
TWSa_C = pd.DataFrame(TWSa_C, index=dates_grace)
TWSa_S = pd.DataFrame(TWSa_S, index=dates_grace)

GWa_N = pd.DataFrame(GWa_N, index=dates_grace)
GWa_C = pd.DataFrame(GWa_C, index=dates_grace)
GWa_S = pd.DataFrame(GWa_S, index=dates_grace)

precip_N = precip_N - precip_N.groupby(precip_N.index.month).transform("mean")
precip_C = precip_C - precip_C.groupby(precip_C.index.month).transform("mean")
precip_S = precip_S - precip_S.groupby(precip_S.index.month).transform("mean")

TWSa_N = TWSa_N - TWSa_N.groupby(TWSa_N.index.month).transform("mean")
TWSa_C = TWSa_C - TWSa_C.groupby(TWSa_C.index.month).transform("mean")
TWSa_S = TWSa_S - TWSa_S.groupby(TWSa_S.index.month).transform("mean")

GWa_N = GWa_N - GWa_N.groupby(GWa_N.index.month).transform("mean")
GWa_C = GWa_C - GWa_C.groupby(GWa_C.index.month).transform("mean")
GWa_S = GWa_S - GWa_S.groupby(GWa_S.index.month).transform("mean")

#%%
GWa_N, GWa_C, GWa_S = HPA_zones(GWa_residual, mask)
TWSa_N, TWSa_C, TWSa_S = HPA_zones(TWSa_residual, mask)
precip_N, precip_C, precip_S = HPA_zones(precip_residual, mask)
SMa_N, SMa_C, SMa_S = HPA_zones(SMa_residual, mask)

gg1 = np.array([precip_N, TWSa_N, GWa_N])
gg2 = np.array([precip_C, TWSa_C, GWa_C])
gg3 = np.array([precip_S, TWSa_S, GWa_S])

# gg1 = np.array([precip_N, precip_C, precip_S])
# gg2 = np.array([TWSa_N, TWSa_C, TWSa_S])
# gg3 = np.array([GWa_N, GWa_C, GWa_S])

gg = np.concatenate([gg1, gg2, gg3]).reshape(3,3).T.flatten()

def plot_FFT_2c(out, matrix, labels, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(3,3, sharex=True, figsize=(8,6), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}

    #for i in range(gg.shape[0]):
    
    # ax[numero_PCs, 0].plot(1 / freqs_ONI, powers_ONI, label=labeli[0], c='k')
    # ax[numero_PCs, 0].plot(1 / freqs_ONI, red_noise_spectrum(oni, freqs_ONI, chunk_size, confidence, scale), c='firebrick', ls='--')

    # ax[numero_PCs, 1].plot(1 / freqs_ONI_model, powers_ONI_model, label=labeli[0], c='k')
    # ax[numero_PCs, 1].plot(1 / freqs_ONI_model, red_noise_spectrum(oni_model, freqs_ONI_model, chunk_size, confidence, scale), c='firebrick', ls='--')

    for i, axe in enumerate(ax.flatten()):
        
        serie = np.nanmean(matrix[i], axis=1)
        serie = serie - np.nanmean(serie)
        n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')

        freqs, powers = signal.csd(serie, serie, window=win,noverlap = chunk_size/4, nperseg = chunk_size, \
                scaling = scale, detrend = 'linear')
        
        axe.stem(freqs * 12, powers/np.sum(powers), label=labels[i], linefmt='black',markerfmt='k')
        #axe.plot(1 / freqs, powers, label=labels[i], c='k')
        axe.plot(freqs * 12, red_noise_spectrum(serie, freqs, chunk_size, confidence, scale), c='firebrick', ls='--')

        print(f'{labels[i]} Max: {np.nanmax(powers)}, {np.round(np.max(powers/np.sum(powers)), 2)} %') 
        print(f'{labels[i]} freq max: {12*freqs[np.argmax(powers)]} cycles per year')
        print(f'{labels[i]} 2 freq max: {12*freqs[np.argsort(powers)[-2]]} cycles per year')
        
        #print(freqs * 12) 

        axe.axvline(1/(2),linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(1/(7), linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=2, frameon=False).remove()
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        # axe.set_ylim(0, 50)
        axe.set_xlim(0, 1)
        #axe.set_xscale('log')
        axe.set_title(labels[i])

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    #axe.set_xlabel('Period [Months]',labelpad=20, **csfont)
    axe.set_xlabel('Frequency $(year⁻¹)$',labelpad=20, **csfont)
    axe.set_ylabel('Power (Fraction of variance)', labelpad=25, **csfont)

    # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    # axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    # axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=False, edgecolor='none', bbox_inches="tight")



# SPECTRAL GRAPH NEW

for chunk_size in [96, 108, 120, 132, 144, 156, 244]:
    #chunk_size = 105 #120 dio bien
    confidence = .95
    scale = 'density'
    win = 'boxcar'

    #labels = ['Precip N', 'TWSa N', 'GWa N', 'Precip C', 'TWSa C', 'GWa C', 'Precip S', 'TWSa S', 'GWa S']
    labels = ['Precip N', 'Precip C', 'Precip S', 'TWSa N', 'TWSa C', 'TWSa S', 'GWa N', 'GWa C', 'GWa S']
    plot_FFT_2c(f'figures/05_SA_series_zonas.pdf', gg, labels)
    print(f'chunk_size: {chunk_size}')


#%%
# COHERENCE


confidence = .99

# PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
# PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T


def plot_coherence_2c(out, matrix, oni, oni_p, labels, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(3,3, sharex=False, figsize=(6,5), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(9):
        

        # serie = PC_score[:,i]
        # serie_model = PCscore_model[:, i]

        serie = np.nanmean(matrix[i], axis=1)
        serie = serie - np.nanmean(serie)

        # serie = matrix[0].mean(axis=0)

        chunk_size = 124
        n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')
        dof = 2*n_chunks
        fval = stats.f.ppf(confidence, 2, dof - 2)
        r2_cutoff = fval / (n_chunks - 1. + fval)

        # chunk_size_m = 244
        # n_chunks_m = np.round((len(serie_model) * 2 / chunk_size_m) - 1).astype('int')
        # dof_m = 2*n_chunks_m
        # fval_m = stats.f.ppf(0.99, 2, dof_m - 2)
        # r2_cutoff_m = fval_m / (n_chunks_m - 1. + fval_m)
        if i > 2:
            freqs, Cxy = signal.coherence(serie.astype('complex'), oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size,detrend = False)
        else:
            freqs, Cxy = signal.coherence(serie.astype('complex'), oni_p.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size,detrend = False)
        
        
        # freqs_model, Cxy_model = signal.coherence(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size_m/2, nperseg = chunk_size_m, \
        #                detrend = False)
        
        ix_enso = np.intersect1d(np.where(freqs <= 1/24)[0], np.where(freqs >= 1/(8*12))[0])
        #ix_enso_m = np.intersect1d(np.where(freqs_model <= 1/24)[0], np.where(freqs_model >= 1/(8*12))[0])

        j = np.argsort(Cxy[ix_enso])[-1]
       # j_m = np.argsort(Cxy_model[ix_enso_m])[-1]

       # print(Cxy[ix_enso], j)
        #print(Cxy_model[ix_enso_m], j_m)

        ax.flatten()[i].plot(freqs[:int(len(freqs)/2 + 1)][1:-1] * 12, Cxy[:int(len(freqs)/2 + 1)][1:-1] , label=labels[i], c='k')
        ax.flatten()[i].hlines(r2_cutoff, 0, 180, linewidth = 2, ls='--', color='r')
        ax.flatten()[i].plot(freqs[ix_enso][j]  * 12, Cxy[ix_enso][j] ,'or',linewidth = 2,markersize = 3)

        # ax[i, 1].plot(1 / freqs_model, Cxy_model, label=labels[i], c='k')
        # ax[i, 1].hlines(r2_cutoff_m, 0, 180, linewidth = 2, ls='--', color='r')
        # ax[i, 1].plot(1 / freqs_model[ix_enso_m][j_m], Cxy_model[ix_enso_m][j_m],'or',linewidth = 2,markersize = 3)

    for axe in ax.flatten():
        
        axe.axvline(1/12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(1/12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=9, frameon=False) .remove()
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        #axe.set_ylim(0, 0.01)
        axe.set_xlim(0, 2)
        axe.set_title(labels[i])

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Frecuency $[year⁻¹]$',labelpad=20, **csfont)
    axe.set_ylabel(r'$R^{2}$', labelpad=25, **csfont)

    # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    # axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    # axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")

plot_coherence_2c(f'figures/05_SA_coherence_zonas.pdf', gg, ONI_grace[1:-2], ONI_precip, labels, 3, FFT)

#%%
# PHASE 

# PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
# PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T


def plot_phase_2c(out, matrix, oni, oni_p, labels, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(3,3, sharex=True, figsize=(6,5), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(9):
        

        # serie = PC_score[:,i]
        # serie_model = PCscore_model[:, i]

        serie = np.nanmean(matrix[i], axis=1)
        serie = serie - np.nanmean(serie)

        chunk_size = 124
        n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')
        dof = 2*n_chunks
        fval = stats.f.ppf(0.99, 2, dof - 2)
        r2_cutoff = fval / (n_chunks - 1. + fval)

        # chunk_size_m = 244
        # n_chunks_m = np.round((len(serie_model) * 2 / chunk_size_m) - 1).astype('int')
        # dof_m = 2*n_chunks_m

        # freqs, Cxy = signal.coherence(serie, oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
        #                detrend = False)
        
        # freqs, Pxy = signal.csd(serie, oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
        #          scaling = 'density', detrend = False)
        
        if i > 2:
            freqs, Cxy = signal.coherence(serie.astype('complex'), oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                       detrend = False)
        
            freqs, Pxy = signal.csd(serie.astype('complex'), oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                     scaling = 'density', detrend = False)
        else:
            freqs, Cxy = signal.coherence(serie.astype('complex'), oni_p.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                       detrend = False)
        
            freqs, Pxy = signal.csd(serie.astype('complex'), oni_p.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                     scaling = 'density', detrend = False)
            
        
        P = -np.angle(Pxy, deg = True)
        
        # freqs_model, Cxy_model = signal.coherence(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size_m/2, nperseg = chunk_size_m, \
        #                detrend = False)
        
        # freqs_model, Pxy_model = signal.csd(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
        #          scaling = 'density', detrend = False)

        # P_model = -np.angle(Pxy_model, deg = True)

        ix_enso = np.intersect1d(np.where(freqs <= 1/24)[0], np.where(freqs >= 1/(8*12))[0])
        # ix_enso_m = np.intersect1d(np.where(freqs_model <= 1/24)[0], np.where(freqs_model >= 1/(8*12))[0])

        j = np.argsort(Cxy[ix_enso])[-1]
        # j_m = np.argsort(Cxy_model[ix_enso_m])[-1]


        ax.flatten()[i].plot(freqs[:int(len(freqs)/2 + 1)][:-1] * 12, P[:int(len(freqs)/2 + 1)][:-1], label=labels[i], c='k')
        ax.flatten()[i].plot(freqs[ix_enso][j] * 12, P[ix_enso][j],'or',linewidth = 2,markersize = 3)

        # ax[i, 1].plot(1 / freqs_model, P_model, label=labels[i], c='k')
        # ax[i, 1].plot(1 / freqs_model[ix_enso_m][j_m], P_model[ix_enso_m][j_m],'or',linewidth = 2,markersize = 3)

        print('-------------------------------------')
        print('            PERIOD ........... PHASE')
        print('   GRACE:     ' + str(np.round(1./freqs[ix_enso][j],0)) + ' months.        ' + str(np.round(P[ix_enso][j])) + ' deg.')
        # print('   HYG MODEL:     ' + str(np.round(1./freqs_model[ix_enso_m][j_m],0)) + ' months.        ' + str(np.round(P_model[ix_enso_m][j])) + ' deg.')

    for axe in ax.flatten():
        
        axe.axvline(1 / 12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(1 / 12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=9, frameon=False) .remove()
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        axe.set_ylim(-185, 185)
        axe.set_xlim(0, 2)
        #axe.set_xlim(12, 8*12)
        axe.axhline(y=0,color='gray')
        axe.set_title(labels[i])

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('frecuency $[year⁻¹]$',labelpad=20, **csfont)
    axe.set_ylabel('Phase difference [degree]', labelpad=30, **csfont)

    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    # axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    # axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")

plot_phase_2c(f'figures/05_SA_phase_zones.pdf', gg, ONI_grace[1:-2], ONI_precip, labels, 3, FFT)

# %%
