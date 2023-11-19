
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
from scipy import linalg
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

def plot_variance_explained(out, val):
    
    # Variance explained by principal components.

    #np.savetxt('FIGURAS_AGU/4val.txt', np.real(val), fmt='%.4e')
    north = (100*val/np.sum(val))*np.sqrt(2/(len(val)-1))

    fig, ax = plt.subplots(figsize=(3,3), dpi=200)
    csfont = {'fontname':'Arial', 'fontsize':12}
    
    #plt.title('% of Variance explained by PCs', fontsize=16, fontweight='bold')
    ax.set_xlabel(r'$PC \: number$', **csfont)
    ax.set_ylabel(r'$Percentage \; of \; Variance$', **csfont)
    ax.bar(np.arange(1,val.shape[0]+1), np.cumsum(val/np.sum(val)*100), 0.6,\
            alpha=0.3, label=r'$\% \,Cumulative$')
    #ax.plot(np.arange(1,val.shape[0]+1),val/np.sum(val)*100,'ko', ls='-',label='% Variance')
    ax.errorbar(np.arange(1,val.shape[0]+1), 100*val/np.sum(val), yerr=north, color='k',\
                label=r"$\% \, PC$", fmt='-')
    ax.legend(frameon=False)
    ax.set_xlim((0,8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='gray', linestyle='--', alpha=0.3)

    fig.savefig(out, dpi=200, transparent=False, edgecolor='none')

def plot_EOFs(out, matrix, lat, lon, num, title,labeli, vals, grid=(1,4), fig_size=(8,4), points=None):
    
    csfont = {'fontname':'Arial'}
    reader = shpreader.Reader('data/HPA/HP.shp')
    lon_ = lon - 360
    fig ,ax= plt.subplots(grid[0], grid[1], figsize=fig_size,sharex=True, sharey=True,\
                          dpi=200, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    if num > 1:
        for i in range(num):
            
            eof = matrix[:,i].reshape(lat.shape[0], lon.shape[0])

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

def plot_scores(out, matrix, eigenval, number_com, colors, label, oni, pdo, dates):

    csfont = {'fontname':'Arial', 'fontsize' : 12}
    style = ['-' , '-', '--', '-.']

    fig, ax = plt.subplots(1,1,figsize=(8,3), dpi=200, sharex=True, sharey=True)

    for i in range(number_com):
        per_var = np.round(np.real(eigenval[i]/np.sum(eigenval)*100),2)
        ax.plot(dates, matrix[:,i], label=label + str(i + 1), c=colors[i], ls=style[i], zorder=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='gray', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, loc=2, ncol=2)
    ax0 = ax.twinx()
    ax0.plot(dates, oni.values, label=r'$ONI$', color='k', zorder=1)
    ax0.fill_between(dates,pdo.values[:],0, color='grey', alpha=0.4, label=r'$PDO$', zorder=0)
    ax0.spines['top'].set_visible(False)
    ax0.legend(frameon=False)    
  
    fig.tight_layout()
    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe0 = axe.twinx()
    axe0.tick_params(axis=u'both', which=u'both',length=0)
    axe0.spines['top'].set_visible(False)
    axe0.spines['right'].set_visible(False)
    axe0.spines['left'].set_visible(False)
    ax.set_ylabel(r'$Stand. \, anomalies$',labelpad=25, **csfont)
    axe.set_xlabel(r'$Time \, [Months]$', labelpad=25, **csfont)
    #  axe.set_ylabel('', labelpad=55, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe0.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe0.yaxis.get_ticklabels()]

    fig.savefig(out, dpi=100, transparent=False, edgecolor='none')

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

GWa_tmp = pd.DataFrame(GWa_detrend, index=dates_grace)
GWa_no_ac = GWa_tmp - GWa_tmp.groupby(GWa_tmp.index.month).transform("mean")

#%%

def PCA(array, matrix_used='cov', normalized=False):
    
    if matrix_used == 'cov':
        
     # Covariance matrix (dot product matrix by its transpose) divided by n-1
     #  cov = (array.T @ array) / array.shape[0]-
        matrix = np.cov(array.T)
      #  print(f'Covariance shape {matrix.shape}')
        vari = np.nanstd(matrix ,axis=0)
        val, vec = linalg.eig(matrix) 
        # PC are obtained from dot product from original matrix and eigenvectors
        PC = array @ vec

        if normalized == True:
        
            # Loadings are eigenvectors normalized by eigenvalues square root  
            # covariance between variables and components

            loadings = (vec[:,:] @ np.diag(np.sqrt(val[:])))#@np.linalg.inv(np.diag(vari[:259]))
    #      print('Loadings shape ', loadings.shape)

            # Sum of squared loading values give variance explained by PC (eigenvalue)
            print('Variance explained by loading 1' , np.sum(loadings[:,0]**2))
            print('Variance explained by loading 2' , np.sum(loadings[:,1]**2))

            # Stardarized loadings

        # Regresssion coefficients Score
        # B = Loadings . inv(diag(eigenvalues)  )
        # B = inv(cov) . loadings - why this doesn't work?

            coef = loadings[:,:] @ np.linalg.inv(np.diag(val[:]))
        #coef = loadings @ np.linalg.inv(np.diag(val))
        #  coef2 = np.linalg.inv(corr) @ loadings[:,:]

            # PC scores
            PCscore = array @ coef

            return loadings, PCscore, val
        
        else:
            
            print('Variance PC 1 ' , np.var(PC[:,0]))
            print('Variance PC 2 ' , np.var(PC[:,1]))
            
            return vec, PC, val
        
    elif matrix_used == 'corr':
        
        # Corr matrix
        #matrix = np.corrcoef(array.T)
        matrix = np.corrcoef(array.T)
       # print('Correlations shape ' + str(comatrixrr.shape))
        val, vec = linalg.eig(matrix) 
        # PC are obtained from dot product from original matrix and eigenvectors
        PC = ((array - np.nanmean(array, axis=0)) / np.std(array, axis=0)) @ vec
    
        if normalized == True:

            PCscore = PC @ np.diag(np.sqrt(val[:]))
            coef = PCscore[:,:] @ np.linalg.inv(np.diag(val[:]))

        #  coef = (coef - np.nanmean(coef, axis=0) / np.var(coef, axis=0))

            loadings = array.T @ coef / len(array)
        #  loadings = (PCscore.T @ array) / len(array)

            # Sum of squared loading values give variance explained by PC (eigenvalue)
            print('Variance explained by loading 1' , np.sum(loadings[:,0]**2))
            print('Variance explained by loading 2' , np.sum(loadings[:,1]**2))
        

            return loadings, coef, val
        
        else:
            return vec, PC, val

matrix_used = 'corr'

print('TWS GRACE residual')
#vec, PC, val = PCA(GWa_residual, matrix_used=matrix_used)
loadings, PCscore, val = PCA(GWa_residual, normalized=True, matrix_used=matrix_used)

plot_variance_explained('figures/06_EA_variance_explained.pdf', val)




# Get rotated matrix from 3 first loadings)
matrix_rot = ortho_rotation(loadings[:,:4])
#matrix_rot_model = ortho_rotation(loadings_model[:,:4])

# Rotated loadings and components
# Components not orthonogals
rotated_load = loadings[:,:4] @ matrix_rot
#rotated_load_model = loadings_model[:,:4] @ matrix_rot_model

# New component data matrix . pseudo.inverser of transponse of rotaed loadings
rotated_PC = GWa_residual @ np.linalg.pinv(rotated_load).T
#rotated_PC_model = GWa_detrend_model_m @ np.linalg.pinv(rotated_load_model).T

variance = np.real(np.sum(loadings**2, axis=0))
#variance_model = np.real(np.sum(loadings_model**2, axis=0))
print('variance normal: ', np.sum(np.round(variance,2)))
print([f"EOF {i+1} : {np.round(100* variance[i] / np.sum(variance), 1)}" for i in range(4)])
# print('variance model: ', np.sum(np.round(variance_model,2)))
# print([f"EOF {i+1} : {np.round(100 * variance_model[i] / np.sum(variance_model),1)}" for i in range(4)])

print(" rotated .. ")
variance_rotated = np.real(np.sum(rotated_load**2, axis=0))
# variance_model_rotated = np.real(np.sum(rotated_load_model**2, axis=0))
print('variance rotated: ', np.sum(np.round(variance_rotated,2)))
print([f"EOF {i+1} : {np.round(100* variance_rotated[i] / np.sum(variance_rotated),1)}" for i in range(4)])
# print('variance model: ', np.sum(np.round(variance_model_rotated,2)))
# print([f"EOF {i+1} : {np.round(100 * variance_model_rotated[i] / np.sum(variance_model_rotated),1)}" for i in range(4)])


# Recovered data is not equal
rotated_GWa_trunc = rotated_PC @ rotated_load.T
#rotated_model_trunc = rotated_PC_model @ rotated_load_model.T

PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
#PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T

rotated_PC_selected = np.array([rotated_PC[:,0], rotated_PC[:,2], rotated_PC[:,3]]).T
#rotated_PC_model_selected = np.array([rotated_PC_model[:,0], rotated_PC_model[:,2], rotated_PC_model[:,3]]).T

GWa_trunc = (PCscore[:, [0]]) @ loadings[:, [0]].T
#GWa_trunc_model = (PCscore_model[:, [0]]) @ loadings_model[:, [0]].T

# para unrotated cov
#componente 3 (2) es el mas poderoso
# 1 y 5 juntos (0 y 4) hacen
# 4 y 5 (3 y 4) tambien muestran algo raro ahi
#

GWa_rotated_trunc = rotated_PC[:,[3]] @ rotated_load[:,[3]].T
#GWa_rotated_trunc_model = rotated_PC_model[:,[3]] @ rotated_load_model[:,[3]].T

# rotated cov
# componente 1 [0] poderoso
# 2,3,4 juntos hacen algo a lag 1 pero poco, quitante el 3 tambien tienen eso, parece controlado por la tendencia negativa en el sur
# 3 y 4 solos parecen tener algo parecido a lo de arriba

# 3 rotado es el mismo que 4 sin rotar
# 4 rotado es 3 sin rotar

# in model 0 tiene granger desde lag 2. 1 y 3 tiene en todos. En 3 tambien se produce granger al revez
# 2 y 3 en modelo parecen ser el mismo propagandose


# para unrotated corr
# componente 3 [2] no tiene granger al lag 1, de resto si en modelo, en GRACE tiene melo
# componente 4 [3] tiene granger en modelo pero raro en GRACE
# 2 no hay nada
#1 tambien esta raro

# rotated corr
# 1 en grace no tiene en lag 2 de resto si, en modelo tiene a partir de lag 2
# 3 [2] en modelo no tiene nada en GRACE tiene hasta el lag 2
# 2 [1] no tiene nada en GRACE y todo en modelo
# 4 [3] tiene granger en modelo pero en GRACE no


# EOFS GRACE GRAPH

rot = False

if rot == True:
    matrix = rotated_load
    matrix2 = rotated_PC
    eigenval = variance_rotated
    name = 'rotated'
else:
    matrix = loadings
    matrix2 = PCscore
    eigenval = val
    name = 'norot'

dates = dates_grace
oni = ONI_grace[1:-2]
csfont = {'fontname':'Arial', 'fontsize' : 12}
lat = lat_grace
lon = lon_grace
vals = 80
labeli = "mm/std"
colors = ['firebrick', 'royalblue', 'darkturquoise', 'lime']
label = "PC"
title = ["PC1", "PC2", "PC3", "PC4"]
reader = shpreader.Reader('data/HPA/HP.shp')
lon_ = lon - 360

fig ,ax= plt.subplots(4, 1, figsize=[8,5.6],sharex=False, sharey=False,\
                        dpi=200, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

for i in range(4):
    
    eof = matrix[:,i].reshape(lat.shape[0], lon.shape[0])

    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].contourf(lon,lat, eof*mask, cmap='bwr_r',vmax=vals,vmin=-1*vals)
    ax.flat[i].set_title(f'EOF {i + 1}', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
                draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
    gl.xlabels_top = False
    gl.ylabels_right = False
    if i < 3:
        gl.xlabels_bottom = False
    gl.ylocator = mticker.FixedLocator([34, 41])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'fontname' : 'Arial', 'fontsize': 10}#csfont
    gl.xlabel_style = {'fontname' : 'Arial', 'fontsize': 10}#csfont     

    # if points is not None:

    #     points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
    #     ax.flat[i].scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())


cbar_ax = fig.add_axes([0.55, 0.03, 0.01, 0.93])
cbar_ax.grid(False)
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

fig.colorbar(plt.cm.ScalarMappable(cmap='bwr_r', \
        norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='vertical',\
    fraction=3, aspect=80, pad=0.03, label='',extend='both', )

y_positions = [0.8, 0.55, 0.29, 0.035]
for i in range(4):

    ax_b = fig.add_axes([0.65, y_positions[i], 0.55, 0.18])
    per_var = np.round(np.real(eigenval[i]/np.sum(eigenval)*100),2)

    ax_b.plot(dates, matrix2[:,i], label=label, color='k', lw=1)
    #       ax.flat[i].set_ylabel('Amplitude [cm]')
    ax_b.set_title(str(per_var) + '%', loc='left', **csfont)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(color='gray', linestyle='--', alpha=0.3)
    ax_b.legend(frameon=False, loc=2)
    ax_b.set_ylim(-3,3)
    ax0 = ax_b.twinx()
    ax0.plot(dates, oni.values, label='ONI', color='r', ls='--')
    # ax0.fill_between(dates,pdo.values[:],0, color='grey', alpha=0.5, label='PDO')
    #      ax0.set_ylabel('SST anomalies [°C]')
    ax0.spines['top'].set_visible(False)
     #       ax0.spines['right'].set_visible(False)
    #ax0.legend(frameon=False)

    if i < 3:
        ax_b.set_xticklabels([])
    else:
        ax_b.set_xlabel('Time', **csfont)

ax_b.text(0.35, 0.999, "a)", transform=fig.transFigure, **csfont)
ax_b.text(0.6, 0.99, "b)", transform=fig.transFigure, **csfont)

fig.subplots_adjust(top = 0.99, bottom=0.01, left=0.01)
#fig.tight_layout()
fig.savefig(f'figures/06_EA_EOF_GWa_{name}_{matrix_used}_detrend.pdf', dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")


# %%
# 

# EOFS MODEL GRAPH

# matrix = loadings_model
# matrix2 = PCscore_model
# eigenval = val_model

# matrix = rotated_load_model
# matrix2 = rotated_PC_model
# eigenval = variance_model_rotated

# dates = dates_model[2:]

# oni = ONI_model.loc[:'2019-07'][2:]
# pdo = PDO_model.loc[:'2019-07'][2:]

# csfont = {'fontname':'Arial', 'fontsize' : 12}
# lat = lat_grace
# lon = lon_grace
# vals = 8
# labeli = "cm/std"
# colors = colors
# label = "PC"
# title = ["PC1", "PC2", "PC3", "PC4"]

# #csfont = {'fontname':'Arial', 'weight':'bold'}
# reader = shpreader.Reader('data/HPA/HP.shp')
# lon_ = lon - 360


# fig ,ax= plt.subplots(4, 1, figsize=[8,5.6],sharex=False, sharey=False,\
#                         dpi=200, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

# for i in range(4):
    
#     eof = matrix[:,i].reshape(lat.shape[0], lon.shape[0])

#     ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
#             facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
#     ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
#         name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
#     ax.flat[i].contourf(lon,lat, eof*mask, cmap='bwr_r',vmax=vals,vmin=-1*vals)
#     ax.flat[i].set_title(f'EOF {i + 1}', **csfont)
#     gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
#                 draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     if i < 3:
#         gl.xlabels_bottom = False
#     gl.ylocator = mticker.FixedLocator([34, 41])
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     gl.ylabel_style = {'fontname' : 'Arial', 'fontsize': 10}#csfont
#     gl.xlabel_style = {'fontname' : 'Arial', 'fontsize': 10}#csfont     

#     # if points is not None:

#     #     points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
#     #     ax.flat[i].scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())


# cbar_ax = fig.add_axes([0.55, 0.03, 0.01, 0.93])
# cbar_ax.grid(False)
# cbar_ax.axis('off')
# [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
# [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
# cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

# fig.colorbar(plt.cm.ScalarMappable(cmap='bwr_r', \
#         norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='vertical',\
#     fraction=3, aspect=80, pad=0.03, label='',extend='both', )

# y_positions = [0.8, 0.55, 0.29, 0.035]
# for i in range(4):

#     ax_b = fig.add_axes([0.65, y_positions[i], 0.55, 0.18])
#     per_var = np.round(np.real(eigenval[i]/np.sum(eigenval)*100),2)

#     ax_b.plot(dates, matrix2[:,i], label=label, color='k', lw=1)
# #       ax.flat[i].set_ylabel('Amplitude [cm]')
#     ax_b.set_title(str(per_var) + '%', loc='left', **csfont)
#     ax_b.spines['top'].set_visible(False)
#     ax_b.spines['right'].set_visible(False)
#     ax_b.grid(color='gray', linestyle='--', alpha=0.3)
#     ax_b.legend(frameon=False, loc=2)
#     ax_b.set_ylim(-3,3)
#     ax0 = ax_b.twinx()
#     ax0.plot(dates, oni.values, label='ONI', color='r', ls='--')
#    # ax0.fill_between(dates,pdo.values[:],0, color='grey', alpha=0.5, label='PDO')
# #      ax0.set_ylabel('SST anomalies [°C]')
#     ax0.spines['top'].set_visible(False)
# #       ax0.spines['right'].set_visible(False)
#     #ax0.legend(frameon=False)

#     if i < 3:
#         ax_b.set_xticklabels([])
#     else:
#         ax_b.set_xlabel('Time', **csfont)

# ax_b.text(0.35, 0.999, "a)", transform=fig.transFigure, **csfont)
# ax_b.text(0.6, 0.99, "b)", transform=fig.transFigure, **csfont)

# fig.subplots_adjust(top = 0.99, bottom=0.01, left=0.01)


# fig.savefig(f'figures/EOF_model_r_{matrix_used}.pdf', dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")
