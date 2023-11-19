
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

def annual_cycle(array, dates, daily=False):

    if array.ndim == 3:
        array = array.reshape(array.shape[0], array.shape[1] * array.shape[2])

    annual = pd.DataFrame((array), index=dates)

    if daily:
        average = annual.groupby(annual.index.day).mean()
        average.index = pd.date_range('1/15/2020', '1/15/2021', freq='D').strftime('%b')

    else:
        average = annual.groupby(annual.index.month).mean()
        average.index = pd.date_range('1/15/2020', '1/15/2021', freq='M').strftime('%b')

    acycle = np.array(average)

    return acycle, average.index

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

def estandarizar(df):
    dfo = (df - df.mean())/df.std()
    return dfo

def spatial_mean(arr_in):

    arr_in = np.array(arr_in) if isinstance(arr_in, list) else arr_in
    try:
        arr_out = np.nanmean(arr_in, axis=(1,2))
    except:
        arr_out = np.nanmean(arr_in, axis=-1)
    return arr_out

def corr_matrix(variables, nombres, dates):
    
    lista = np.array([spatial_mean(var) for var in variables]).T
    df = pd.DataFrame(lista, index=dates, columns=nombres)
    return df.corr()

def corrmap(X,Y):
    corrmap = np.zeros(X.shape[1])
    pvalues = np.zeros(X.shape[1])
    pvalue = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        corrmap[j] = np.array([np.real(pearsonr(X[:,j], Y)[0])])
        pvalues[j] = np.array([np.real(pearsonr(X[:,j], Y)[1])])
            
        if pvalues[j] >= 0.01:
            pvalues[j] = np.nan
        else:
            pvalues[j] = 1
    
    return corrmap, pvalue

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

def plot_map(out, matrix, lat, lon, num, title,labeli, vals, grid=(1,4), fig_size=(8,4), points=None):
    
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
            ax.flat[i].contourf(lon,lat, eof*mask, cmap='bwr_r',vmax=vals,vmin=-1*vals)

            gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
                        draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 2, dtype=int))
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.ylabel_style = {'fontname':'Arial', 'fontsize': 10}
            gl.xlabel_style = {'fontname':'Arial', 'fontsize': 10}        

            ax.flat[i].set_title(title[i],loc='left',fontsize=12, **csfont)
                
            if points is not None:
                points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
                ax.flat[i].scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())

    else:
        eof = matrix.reshape(lat.shape[0], lon.shape[0])
        ax.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
                    facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
        ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
            name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
        ax.contourf(lon,lat, eof*mask, cmap='bwr_r',vmax=vals,vmin=-1*vals)

        gl = ax.gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
                    draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 2, dtype=int))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'fontname':'Arial', 'fontsize': 10}
        gl.xlabel_style = {'fontname':'Arial', 'fontsize': 10}        

        ax.set_title(title,loc='left',fontsize=12, **csfont)
            
        if points is not None:
            points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
            ax.scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())
            
    fig.subplots_adjust(bottom=0.3)
    cbar_ax = fig.add_axes([0.1, 0.0001, 0.8, 0.01])
    cbar_ax.grid(False)
    cbar_ax.axis('off')
    [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
    cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

    fig.colorbar(plt.cm.ScalarMappable(cmap='bwr_r', \
            norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
        fraction=3, aspect=80, pad=0.08, label=labeli,extend='both', )

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches='tight')


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

# CUT VARIABLES IN HPA ZONES

GWa_N, GWa_C, GWa_S = HPA_zones(GWa_m, mask)
TWSa_N, TWSa_C, TWSa_S = HPA_zones(TWSa_m, mask)
precip_N, precip_C, precip_S = HPA_zones(precip_m, mask)
SMa_N, SMa_C, SMa_S = HPA_zones(SMa_m, mask)


GWa_detrend = detrend(GWa_m, axis=0)
TWSa_detrend = detrend(TWSa_m, axis=0)
precip_a = precip_m - np.nanmean(precip_m, axis=0)
precip_detrend = detrend(precip_a, axis=0)
SMa_detrend = detrend(SMa_m, axis=0)

# REMOVE SEASONAL COMPONENT

GWa_residual = deseasonal(GWa_detrend.reshape(244, 24, 18))[0] + deseasonal(GWa_detrend.reshape(244, 24, 18))[2] # sumar tendencia y residual
GWa_residual = np.reshape(GWa_residual, (244, 432))

precip_residual = deseasonal(precip_detrend.reshape(499, 24, 18))[0] + deseasonal(precip_detrend.reshape(499, 24, 18))[2] # sumar tendencia y residual
precip_residual = np.reshape(precip_residual, (499, 432))

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

# SERIES GRACE Y MODEL !!

nombres_df_GWa = ["GWa_mean","GWa_min", "GWa_max", "GWa_N_mean", "GWa_N_min", "GWa_N_max", "GWa_C_mean", "GWa_C_min",
              "GWa_C_max", "GWa_S_mean", "GWa_S_min", "GWa_S_max"]

nombres_df_TWSa = ["TWSa_mean","TWSa_min", "TWSa_max", "TWSa_N_mean", "TWSa_N_min", "TWSa_N_max", "TWSa_C_mean", "TWSa_C_min",
              "TWSa_C_max", "TWSa_S_mean", "TWSa_S_min", "TWSa_S_max"]

nombres_df_precip = ["precip_mean","precip_min", "precip_max", "precip_N_mean", "precip_N_min", "precip_N_max", "precip_C_mean", "precip_C_min",
              "precip_C_max", "precip_S_mean", "precip_S_min", "precip_S_max"]

GWa_stat = np.array([np.nanmean(GWa_m.reshape(244, 24, 18)*mask, axis=(1,2)), np.nanmin(GWa_m.reshape(244, 24, 18)*mask, axis=(1,2)),
                                 np.nanmax(GWa_m.reshape(244, 24, 18)*mask, axis=(1,2)), np.nanmean(GWa_N, axis=1),np.nanmin(GWa_N, axis=1),
                                 np.nanmax(GWa_N, axis=1),np.nanmean(GWa_C, axis=1),np.nanmin(GWa_C, axis=1),
                                 np.nanmax(GWa_C, axis=1), np.nanmean(GWa_S, axis=1), np.nanmin(GWa_S, axis=1), 
                                 np.nanmax(GWa_S, axis=1)])

TWSa_stat = np.array([np.nanmean(TWSa_m.reshape(244, 24, 18)*mask, axis=(1,2)), np.nanmin(TWSa_m.reshape(244, 24, 18)*mask, axis=(1,2)),
                                 np.nanmax(TWSa_m.reshape(244, 24, 18)*mask, axis=(1,2)), np.nanmean(TWSa_N, axis=1),np.nanmin(TWSa_N, axis=1),
                                 np.nanmax(TWSa_N, axis=1),np.nanmean(TWSa_C, axis=1),np.nanmin(TWSa_C, axis=1),
                                 np.nanmax(TWSa_C, axis=1), np.nanmean(TWSa_S, axis=1), np.nanmin(TWSa_S, axis=1), 
                                 np.nanmax(TWSa_S, axis=1)])

precip_stat = np.array([np.nanmean(precip_m.reshape(499, 24, 18)*mask, axis=(1,2)), np.nanmin(precip_m.reshape(499, 24, 18)*mask, axis=(1,2)),
                                 np.nanmax(precip_m.reshape(499, 24, 18)*mask, axis=(1,2)), np.nanmean(precip_N, axis=1),np.nanmin(precip_N, axis=1),
                                 np.nanmax(precip_N, axis=1),np.nanmean(precip_C, axis=1),np.nanmin(precip_C, axis=1),
                                 np.nanmax(precip_C, axis=1), np.nanmean(precip_S, axis=1), np.nanmin(precip_S, axis=1), 
                                 np.nanmax(precip_S, axis=1)])

df_GWa_variables = pd.DataFrame(GWa_stat.T, index=dates_grace, columns=nombres_df_GWa)
df_TWSa_variables = pd.DataFrame(TWSa_stat.T, index=dates_grace, columns=nombres_df_TWSa)
df_precip_variables = pd.DataFrame(precip_stat.T, index=dates_chirps, columns=nombres_df_precip)

df_GWa_variables[mask_nan] = np.nan
GWa_detrend[mask_nan] = np.nan
df_TWSa_variables[mask_nan] = np.nan
TWSa_detrend[mask_nan] = np.nan

df_All_variables = pd.concat((df_precip_variables, df_TWSa_variables, df_GWa_variables), axis=1)
df_All_variables = ENSO_category(df_All_variables, read_oni(data_oni, ['1981-01','2022-10'], standard=False))
df_All_variables = df_All_variables.loc["2002":]

y = [["precip_N_mean", "precip_C_mean", "precip_S_mean"], ["TWSa_N_mean", "TWSa_C_mean", "TWSa_S_mean"], 
     ["GWa_N_mean", "GWa_C_mean", "GWa_S_mean"]]

styles = ["-", "--"]
titles = ["Precipitation", "TWSa", "GWa"]
colores = ["firebrick","darkgoldenrod", "indigo"]

label = ["Northern HPA", "Central HPA", "Southern HPA"]

fill_params = {"alpha" : 0.2}
fig, ax = plt.subplots(3, figsize=[10,8], sharex=False, dpi=200)

df_All_variables.plot(y=y[0], ax=ax[0], kind='bar', stacked=True, title=titles[0], color = colores, lw=2)

for phase in [-1.,0.,1.]:
    mask_enso = df_All_variables.phase == phase
    ax[0].fill_between(np.arange(len(df_All_variables.index)), 0, 400,\
            where=mask_enso, facecolor=['skyblue','white','salmon'][int(phase)+1], **fill_params)

ax[0].xaxis.set_tick_params(labelbottom=False)
ax[0].set_ylim(0,300)
ax[0].xaxis.set_minor_locator(mdates.YearLocator(2))
ax[0].set_xticks(ax[0].get_xticks()[::12])

for ixx, ys in enumerate(y[1:]):
    ax[ixx + 1].xaxis.set_minor_locator(mdates.YearLocator(2))
    df_All_variables.plot(y=ys, ax=ax[ixx + 1], style=styles, title=titles[ixx + 1], color = colores, lw=1.2, label=label)
    plot_enso_years(df_All_variables, y[ixx + 1], ax=ax[ixx + 1], fill_params=fill_params)
    ax[ixx + 1].set_ylim(np.nanmin(df_All_variables[y[ixx + 1]]), np.nanmax(df_All_variables[y[ixx + 1]]))

for axe in ax:
    axe.grid(which="both", alpha=0.3)
    axe.legend().remove()
    axe.spines[['top', "right"]].set_visible(False)
    
ax[1].set_xticklabels([])
ax[1].set_ylabel('mm')
ax[-1].set_xlabel('Time')
ax[-1].legend(bbox_to_anchor=(0.8, -0.25), ncol=3)

fig.savefig('figures/03_CA_series.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')

#%%
# FIGURA DEL MODELO !!!

nombres_df = ["GWa_mean","GWa_min", "GWa_max", "GWa_N_mean", "GWa_N_min", "GWa_N_max", "GWa_C_mean", "GWa_C_min",
              "GWa_C_max", "GWa_S_mean", "GWa_S_min", "GWa_S_max"]

nombres_df_TWSa = ["TWSa_mean","TWSa_min", "TWSa_max", "TWSa_N_mean", "TWSa_N_min", "TWSa_N_max", "TWSa_C_mean", "TWSa_C_min",
              "TWSa_C_max", "TWSa_S_mean", "TWSa_S_min", "TWSa_S_max"]

GWa_model_stat = np.array([np.nanmean(GWa_model_m.reshape(484, 24, 18)*mask, axis=(1,2)), np.nanmin(GWa_model_m.reshape(484, 24, 18)*mask, axis=(1,2)),
                                 np.nanmax(GWa_model_m.reshape(484, 24, 18)*mask, axis=(1,2)), np.nanmean(GWa_model_N, axis=1),np.nanmin(GWa_model_N, axis=1),
                                 np.nanmax(GWa_model_N, axis=1),np.nanmean(GWa_model_C, axis=1),np.nanmin(GWa_model_C, axis=1),
                                 np.nanmax(GWa_model_C, axis=1), np.nanmean(GWa_model_S, axis=1), np.nanmin(GWa_model_S, axis=1), 
                                 np.nanmax(GWa_model_S, axis=1)])

TWS_model_stat = np.array([np.nanmean(TWS_model_m.reshape(484, 24, 18)*mask, axis=(1,2)), np.nanmin(TWS_model_m.reshape(484, 24, 18)*mask, axis=(1,2)),
                                 np.nanmax(TWS_model_m.reshape(484, 24, 18)*mask, axis=(1,2)), np.nanmean(TWS_model_N, axis=1),np.nanmin(TWS_model_N, axis=1),
                                 np.nanmax(TWS_model_N, axis=1),np.nanmean(TWS_model_C, axis=1),np.nanmin(TWS_model_C, axis=1),
                                 np.nanmax(TWS_model_C, axis=1), np.nanmean(TWS_model_S, axis=1), np.nanmin(TWS_model_S, axis=1), 
                                 np.nanmax(TWS_model_S, axis=1)])

df_GWa_model_variables = pd.DataFrame(GWa_model_stat.T, index=dates_model, columns=nombres_df)
df_TWSa_model_variables = pd.DataFrame(TWS_model_stat.T, index=dates_model, columns=nombres_df_TWSa)

df_all_variables_model = pd.concat((df_precip_variables, df_TWSa_model_variables, df_GWa_model_variables), axis=1)
df_all_variables_model = ENSO_category(df_all_variables_model, read_oni(data_oni, ['1981-01','2022-10'], standard=False))
df_all_variables_model = df_all_variables_model.loc["1979":"2019"]

y = [["precip_N_mean", "precip_C_mean", "precip_S_mean"], ["TWSa_N_mean", "TWSa_C_mean", "TWSa_S_mean"], 
     ["GWa_N_mean", "GWa_C_mean", "GWa_S_mean"]]

styles = ["-", "--"]
titles = ["Precipitation", "TWSa", "GWa"]
colores = ["firebrick","darkgoldenrod", "indigo"]

label = ["Northern HPA", "Central HPA", "Southern HPA"]
fill_params = {"alpha" : 0.2}
fig, ax = plt.subplots(3, figsize=[10,8], sharex=False, dpi=200)

df_all_variables_model.plot(y=y[0], ax=ax[0], kind='bar', stacked=True, title=titles[0], color = colores, lw=2)

for phase in [-1.,0.,1.]:
    mask_enso = df_all_variables_model.phase == phase
    ax[0].fill_between(np.arange(len(df_all_variables_model.index)), 0, 400,\
            where=mask_enso, facecolor=['skyblue','white','salmon'][int(phase)+1], **fill_params)

ax[0].xaxis.set_tick_params(labelbottom=False)
ax[0].set_ylim(0, 300)
ax[0].xaxis.set_minor_locator(mdates.YearLocator(2))
ax[0].set_xticks(ax[0].get_xticks()[::12])

for ixx, ys in enumerate(y[1:]):
    ax[ixx + 1].xaxis.set_minor_locator(mdates.YearLocator(2))
    df_all_variables_model.plot(y=ys, ax=ax[ixx + 1], style=styles, title=titles[ixx + 1], color = colores, lw=1.2, label=label)
    plot_enso_years(df_all_variables_model, y[ixx + 1], ax=ax[ixx + 1], fill_params=fill_params)
    ax[ixx + 1].set_ylim(np.nanmin(df_all_variables_model[y[ixx + 1]]), np.nanmax(df_all_variables_model[y[ixx + 1]]))

for axe in ax:
    axe.grid(which="both", alpha=0.3)
    axe.legend().remove()
    axe.spines[['top', "right"]].set_visible(False)
    
ax[1].set_xticklabels([])
ax[1].set_ylabel('mm')
ax[-1].set_xlabel('Time')
ax[-1].legend(bbox_to_anchor=(0.8, -0.25), ncol=3)

fig.savefig('figures/03_CA_series_model.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')

#%%

nombres = ['Precip', r"TWSa", r"SMa", r"SWEa", 'GWa', 'ONI']
nombres_model = ['Precip', r"TWSa m", r"SMa", r"SWEa", 'GWa m', 'ONI']
variables = [precip_m[255:], TWSa_m, SMa_m, SWEa_m, GWa_m, ONI_grace[2:-1].values]
variables_model = [precip_m[:460], TWS_model_m[24:], SMa_model_m[24:], SWEa_model_m[24:], GWa_model_m[24:], ONI_model[26:-1].values]

lista = np.array([spatial_mean(var) for var in variables[:-1]]).T
df = pd.DataFrame(lista, index=dates_grace, columns=nombres[:-1])
df[nombres[-1]] = variables[-1]

lista = np.array([spatial_mean(var) for var in variables_model[:-1]]).T
df_model = pd.DataFrame(lista, index=dates_model[24:], columns=nombres_model[:-1])
df_model[nombres_model[-1]] = variables_model[-1]

#plot corr
fig, axes = plt.subplots(1,2,figsize=(12,6), dpi=200)

sns.heatmap(df.corr(),ax=axes[0] , annot=True, cmap='coolwarm', vmin=-1, vmax=1)
sns.heatmap(df_model.corr(),ax=axes[1] , annot=True, cmap='coolwarm', vmin=-1, vmax=1)

fig.savefig('figures/03_CA_corr_total.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')

#%%
nombres = ['Precip', r"TWSa", r"SMa", 'GWa', 'ONI']
label_z = ['North', 'Central', 'South']
variables = [[precip_N[255:], TWSa_N, SMa_N, GWa_N, ONI_grace[2:-1].values], \
              [precip_C[255:], TWSa_C, SMa_C, GWa_C, ONI_grace[2:-1].values], \
              [precip_S[255:], TWSa_S, SMa_S, GWa_S, ONI_grace[2:-1].values]]

variables_model = [[precip_N[:460], TWS_model_N[24:], SMa_model_N[24:], GWa_model_N[24:], ONI_model[26:-1].values], \
              [precip_C[:460], TWS_model_C[24:], SMa_model_C[24:], GWa_model_C[24:], ONI_model[26:-1].values], \
              [precip_S[:460], TWS_model_S[24:], SMa_model_S[24:], GWa_model_S[24:], ONI_model[26:-1].values]]

fig, axes = plt.subplots(2,3,figsize=(14,8), dpi=200)

for ix in range(len(variables)):
    lista = np.array([spatial_mean(var) for var in variables[ix][:-1]]).T
    df = pd.DataFrame(lista, index=dates_grace, columns=nombres[:-1])
    df[nombres[-1]] = variables[ix][-1]
    sns.heatmap(df.corr(),ax=axes.flat[ix] , annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    axes.flat[ix].set_title(label_z[ix])

for ix in range(len(variables_model)):
    lista = np.array([spatial_mean(var) for var in variables_model[ix][:-1]]).T
    df = pd.DataFrame(lista, index=dates_model[24:], columns=nombres[:-1])
    df[nombres[-1]] = variables_model[ix][-1]
    sns.heatmap(df.corr(),ax=axes.flat[ix+3] , annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    fig.savefig('figures/03_CA_corr_zones.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')

#%%



# r_precip_grace = corrmap(precip_m, read_oni(data_oni, ['1981-03','2022-09']))[0]
# r_GWa_grace = corrmap(GWa_m, ONI_grace[2:-1])[0]

# r_precip = np.stack((r_precip_grace, r_GWa_grace), axis=0)

# corr_lagged_precip = np.array([corrmap(precip_m, np.roll(read_oni(data_oni, ['1981-03','2022-09']), i))[0] for i in range(-3,3)])
# corr_lagged_GWa = np.array([corrmap(GWa_m, np.roll(ONI_grace[2:-1], i))[0] for i in range(-3,3)])

# corr_lagged_precip = np.array([corrmap(precip_detrend, np.roll(read_oni(data_oni, ['1981-03','2022-09']), i))[0] for i in range(-3,3)])
# corr_lagged_GWa = np.array([corrmap(GWa_detrend, np.roll(ONI_grace[2:-1], i))[0] for i in range(-3,3)])

corr_lagged_precip = np.array([corrmap(precip_residual, np.roll(read_oni(data_oni, ['1981-03','2022-09']), i))[0] for i in range(-3,3)])
corr_lagged_GWa = np.array([corrmap(GWa_residual, np.roll(ONI_grace[2:-1], i))[0] for i in range(-3,3)])

lagged_subplot = np.concatenate((corr_lagged_precip, corr_lagged_GWa))

labels = ['']*12
plot_map(f'figures/03_CA_map.jpg', lagged_subplot, lat_grace, lon_grace, 12, labels,r'Pearson correlation',\
        .5, grid=(2,6), fig_size=(12,6), points=None)



