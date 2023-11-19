
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
import numpy as np
import pandas as pd
import scipy.signal as signal
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.dates import DateFormatter
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import STL

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

# ANOMALIES

TWSa = TWS - np.nanmean(TWS, axis=0)
SMa, SMa_model = SM_nldas - np.nanmean(SM_nldas, axis=0), SM_model - np.nanmean(SM_model, axis=0)
SWEa, SWEa_model = SWE_nldas - np.nanmean(SWE_nldas, axis=0), SWE_model - np.nanmean(SWE_model, axis=0)

GWa = TWSa - SMa - SWEa    # Ecuacion 1

# GWa_m = moving_mean(GWa.reshape(247, 24*18), 3, center=True)[2:-1]
# TWSa_m = moving_mean(TWSa.reshape(247, 24*18), 3, center=True)[2:-1]
# precip_m = moving_mean(precip.reshape(502, 24*18), 3, center=True)[2:-1]
# SMa_m = moving_mean(SMa.reshape(247, 24*18), 3, center=True)[2:-1]

GWa_m = GWa[2:-1]
precip_m = precip[2:-1]
TWSa_m = TWSa[2:-1]
SMa_m = SMa[2:-1]

dates_grace = dates_grace[2:-1]
dates_chirps = pd.date_range("1981-01", "2022-11", freq="M")[2:-1]

# CUT VARIABLES IN HPA ZONES

GWa_N, GWa_C, GWa_S = HPA_zones(GWa_m, mask)
TWSa_N, TWSa_C, TWSa_S = HPA_zones(TWSa_m, mask)
precip_N, precip_C, precip_S = HPA_zones(precip_m, mask)
SMa_N, SMa_C, SMa_S = HPA_zones(SMa_m, mask)

# CALCULATE AVERAGE ANNUAL CYCLE OF EACH VARIABLE IN TOTAL HPA AND EACH ZONE

GWa_annual, dates_annual = annual_cycle(GWa_m, dates_grace)
GWa_annual_N, _ = annual_cycle(GWa_N, dates_grace)
GWa_annual_C, _ = annual_cycle(GWa_C, dates_grace)
GWa_annual_S, _ = annual_cycle(GWa_S, dates_grace)

precip_annual, dates_annual = annual_cycle(precip_m, dates_chirps)
precip_annual_N, _ = annual_cycle(precip_N, dates_chirps)
precip_annual_C, _ = annual_cycle(precip_C, dates_chirps)
precip_annual_S, _ = annual_cycle(precip_S, dates_chirps)

TWSa_annual, dates_annual = annual_cycle(TWSa_m, dates_grace)
TWSa_annual_N, _ = annual_cycle(TWSa_N, dates_grace)
TWSa_annual_C, _ = annual_cycle(TWSa_C, dates_grace)
TWSa_annual_S, _ = annual_cycle(TWSa_S, dates_grace)

SMa_annual, dates_annual = annual_cycle(SMa_m, dates_grace)
SMa_annual_N, _ = annual_cycle(SMa_N, dates_grace)
SMa_annual_C, _ = annual_cycle(SMa_C, dates_grace)
SMa_annual_S, _ = annual_cycle(SMa_S, dates_grace)

# CALCULATE ANOMALIES FROM WELLS

nombres = sorted(os.listdir('data/pozos'))
delimiters = [' ', ' ', '\t', '\t','\t','\t','\t','\t','\t',' ','\t','\t','\t','\t',' ', '\t','\t','\t', '\t']
col_date = [3,5,2,3,3,2,3,3,3,3,3,3,3,3,3,3,3,3,3]
col_val = [4,10,3,6,6,3,6,6,6,4,6,6,6,6,4,6,6,6,6]
df = pd.DataFrame(index=pd.date_range('01-01-1940', '31-10-2021', freq='M'))

for i, j in enumerate(nombres):
    serie = pd.read_csv(f'data/pozos/{nombres[i]}', delimiter=delimiters[i], header = None, usecols = [col_date[i],col_val[i]], index_col=0,names = ['date' ,nombres[i].split('.')[0]], infer_datetime_format = True)
    serie.index = pd.to_datetime(serie.index, errors='coerce')
    serie = serie.resample('M').mean()
    df = df.join(serie)
    
ft_to_mm = 304.8
s_y = -0.15

df_anom = (df*ft_to_mm - (df*ft_to_mm).mean()) * s_y
df_anom = df_anom.interpolate('linear', limit_direction = 'both')
df_anom = df_anom['1979':'06-2021']
df_anom = df_anom.drop('NE04', axis=1)

locacion = pd.read_excel('data/loc_pozos.xls', index_col=0)
locacion = locacion.drop('NE04')
names = df_anom.columns
names_zoom = names[1:10]
names = np.setdiff1d(names, names_zoom)

average = df_anom.groupby(df_anom.index.month).mean()
average.index = pd.date_range('1/15/2020', '1/15/2021', freq='M').strftime('%B')
average.plot(figsize=(14,8),subplots=True, layout=(6,4), title= 'CICLO ANUAL')

df_anom = df.rolling(3, win_type='boxcar', center=True).mean()
df_trend, df_season, df_residual = deseasonal_df(df_anom)
df_des = df_trend + df_residual

av_north = (average[['NE01', 'NE02', 'NE03', 'NE05', 'NE06', 'NE08', 'NE09', 'NE10','NE12', 'NE14',\
                      'NE15', 'NE17', 'NE19', 'KS02']]).mean(axis=1)
av_central = (average[['TX08', 'TX08']]).mean(axis=1)

#%%
# FIGURE: ANNUAL CYCLE GWA

variables = [GWa_annual_N, GWa_annual_C, GWa_annual_S]
nombres_GRACE = ["GWa North", "GWa Central", "GWa South"]
months = ['J', 'F', 'M', 'A', 'M ', ' J', ' J ', 'A', 'S', 'O', 'N', 'D']

csfont = {'fontname':'Arial', 'fontsize' : 12}
reader = shpreader.Reader('data/HPA/HP.shp')
mascara = np.genfromtxt('data/mask.csv', delimiter = ',')

cmap = 'seismic_r'
matrix = GWa_annual
title = dates_annual
labeli = "GW anomalies [mm]"
vals = 100

fig ,ax= plt.subplots(3,6,figsize=(7.8,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    mes = matrix[i,:].reshape(24,18)
    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=-1*vals)

for k in ax[1,:]:
    gl = k.gridlines(crs=ccrs.PlateCarree(),xlocs=[-104,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylabels_left = False
    gl.ylocator = mticker.FixedLocator([31,35,39,43])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = csfont
    
for z in ax[:2,0]:   
    gl2 = z.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xlabels_bottom = False
    gl2.ylocator = mticker.FixedLocator([31,35,39,43])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.ylabel_style = csfont     

for y in ax.flat[-6:]:
    y.axis('off')

cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1,label=labeli, extend='both')

colors_s = ['r', 'darkblue', 'k']
ax_b = fig.add_axes([0.1, 0.0001, 0.41, 0.23])
for i in range(len(variables)):
    ax_b.plot(title, spatial_mean(variables[i]), label=nombres_GRACE[i], color=colors_s[i])   
ax_b.set_xlabel('Time')
ax_b.set_ylabel('GWa [mm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
ax_b.set_ylim(-50, 50)
ax_b.set_xticklabels(months)

ax_c = fig.add_axes([0.55, 0.0001, 0.4, 0.23])
ax_c.plot(av_north - np.mean(av_north), label='North', c='r')
ax_c.plot(av_central - np.mean(av_central), label='Central', c='darkblue', ls='-')
ax_c.set_xlabel('Time', **csfont)
ax_c.legend(prop={'family' : 'Arial'})
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(alpha=0.3)
ax_c.set_xticklabels(months)
ax_c.text(0.04, 0.86, "a)", transform=fig.transFigure, **csfont)
ax_c.text(0.04, 0.24, "b)", transform=fig.transFigure, **csfont)
ax_c.text(0.5, 0.24, "c)", transform=fig.transFigure, **csfont)

fig.savefig('figures/02_AC_GWa.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')


#%%
# FIGURE: ANNUAL CYCLE PRECIPITATION

csfont = {'fontname':'Arial', 'fontsize' : 12}
reader = shpreader.Reader('data/HPA/HP.shp')
mascara = np.genfromtxt('data/mask.csv', delimiter = ',')

precip_N_mean = spatial_mean(precip_annual_N)
precip_C_mean = spatial_mean(precip_annual_C)
precip_S_mean = spatial_mean(precip_annual_S)

print(np.sum(precip_N_mean), 'mm/year in north')
print(np.sum(precip_C_mean), 'mm/year in central')
print(np.sum(precip_S_mean), 'mm/year in south')

cmap = 'Blues'
matrix = precip_annual
title = dates_annual
labeli = "Rain [mm]"
vals = 100

df_precip_mean = pd.DataFrame(np.stack((precip_N_mean,precip_C_mean, precip_S_mean), axis=1),\
                              index=title, columns=['North', 'Central', 'South'])

fig ,ax= plt.subplots(3,6,figsize=(7,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    mes = matrix[i,:].reshape(24,18)
    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=0)

for k in ax[1,:]:
    gl = k.gridlines(crs=ccrs.PlateCarree(),xlocs=[-104,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylabels_left = False
    gl.ylocator = mticker.FixedLocator([31,35,39,43])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'fontsize':10, 'fontname':'Arial'}#csfont
    
for z in ax[:2,0]:   
    gl2 = z.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xlabels_bottom = False
    gl2.ylocator = mticker.FixedLocator([31,35,39,43])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.ylabel_style = {'fontsize':10, 'fontname':'Arial'}     

for y in ax.flat[-6:]:
    y.axis('off')

cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(0,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1, extend='both')


ax_b = fig.add_axes([0.1, 0.0001, 0.8, 0.23])
df_precip_mean.plot(kind="bar", rot=0, ax=ax_b,  color=['steelblue', 'k', 'dodgerblue'])
ax_b.set_xlabel('Time')
ax_b.set_ylabel('Rain [mm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
ax_b.set_ylim(0, 100)
ax_b.text(0.45, 0.28, labeli, transform=fig.transFigure, **csfont) # color bar label, la puse aca para facilidad
ax_b.text(0.02, 0.85, "a)", transform=fig.transFigure, **csfont)
ax_b.text(0.02, 0.28, "b)", transform=fig.transFigure, **csfont)

fig.savefig('figures/02_AC_precip.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')


#%%

TWSa_N_mean = spatial_mean(TWSa_annual_N)
TWSa_C_mean = spatial_mean(TWSa_annual_C)
TWSa_S_mean = spatial_mean(TWSa_annual_S)
df_TWSa_mean = pd.DataFrame(np.stack((TWSa_N_mean,TWSa_C_mean, TWSa_S_mean), axis=1),\
                              index=title, columns=['North', 'Central', 'South'])

cmap = 'PiYG'
matrix = TWSa_annual
title = dates_annual
labeli = "TWSa [mm]"
vals = 50

fig ,ax= plt.subplots(3,6,figsize=(7,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    mes = matrix[i,:].reshape(24,18)
    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=-1*vals)

for k in ax[1,:]:
    gl = k.gridlines(crs=ccrs.PlateCarree(),xlocs=[-104,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylabels_left = False
    gl.ylocator = mticker.FixedLocator([31,35,39,43])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'fontsize':10, 'fontname':'Arial'}#csfont
    
for z in ax[:2,0]:   
    gl2 = z.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xlabels_bottom = False
    gl2.ylocator = mticker.FixedLocator([31,35,39,43])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.ylabel_style = {'fontsize':10, 'fontname':'Arial'}     

for y in ax.flat[-6:]:
    y.axis('off')

cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1, extend='both')


ax_b = fig.add_axes([0.1, 0.0001, 0.8, 0.23])
df_TWSa_mean.plot(rot=0, ax=ax_b,  color=['olive', 'k', 'darkmagenta'])
ax_b.set_xlabel('Time')
ax_b.set_ylabel('TWSa [mm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
ax_b.set_ylim(-50, 50)
ax_b.text(0.45, 0.28, labeli, transform=fig.transFigure, **csfont) # color bar label, la puse aca para facilidad
ax_b.text(0.02, 0.85, "a)", transform=fig.transFigure, **csfont)
ax_b.text(0.02, 0.28, "b)", transform=fig.transFigure, **csfont)

fig.savefig('figures/02_AC_TWSa.jpg', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')

#%%

SMa_N_mean = spatial_mean(SMa_annual_N)
SMa_C_mean = spatial_mean(SMa_annual_C)
SMa_S_mean = spatial_mean(SMa_annual_S)
df_SMa_mean = pd.DataFrame(np.stack((SMa_N_mean,SMa_C_mean, SMa_S_mean), axis=1),\
                              index=title, columns=['North', 'Central', 'South'])

cmap = 'PRGn'
matrix = SMa_annual
title = dates_annual
labeli = "SMa [mm]"
vals = 50


fig ,ax= plt.subplots(3,6,figsize=(7,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    mes = matrix[i,:].reshape(24,18)
    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=-1*vals)

for k in ax[1,:]:
    gl = k.gridlines(crs=ccrs.PlateCarree(),xlocs=[-104,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylabels_left = False
    gl.ylocator = mticker.FixedLocator([31,35,39,43])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'fontsize':10, 'fontname':'Arial'}#csfont
    
for z in ax[:2,0]:   
    gl2 = z.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-99],
                    draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xlabels_bottom = False
    gl2.ylocator = mticker.FixedLocator([31,35,39,43])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.ylabel_style = {'fontsize':10, 'fontname':'Arial'}     

for y in ax.flat[-6:]:
    y.axis('off')

cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1, extend='both')


ax_b = fig.add_axes([0.1, 0.0001, 0.8, 0.23])
df_SMa_mean.plot(rot=0, ax=ax_b,  color=['g', 'k', 'purple'])
ax_b.set_xlabel('Time')
ax_b.set_ylabel('SMa [mm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
ax_b.set_ylim(-35, 35)
ax_b.text(0.45, 0.28, labeli, transform=fig.transFigure, **csfont) # color bar label, la puse aca para facilidad
ax_b.text(0.02, 0.85, "a)", transform=fig.transFigure, **csfont)
ax_b.text(0.02, 0.28, "b)", transform=fig.transFigure, **csfont)

fig.savefig('figures/02_AC_SMa.jpg', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')
