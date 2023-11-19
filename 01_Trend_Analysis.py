
#%%

import os
import warnings
from datetime import datetime, timedelta

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from netCDF4 import Dataset
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')


"**************************************************************************"

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

    TWS_ = LWE[:,latgrS:latgrN,longrW:longrE]*10 # pasando a mm
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
    TWS_model = TWS_[:,::-1,:]

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

def read_pdo(data, dates):

    data.index = pd.to_datetime(data.index, format='%Y%m', errors = "ignore")
    return estandarizar(pd.Series(data['Value'].loc[dates[0]:dates[1]],index=data.loc[dates[0]:dates[1]].index))

def estandarizar(df):
    dfo = (df - df.mean())/df.std()
    return dfo

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def tendencia(array_in):
    
    array_interp = np.zeros_like(array_in)*np.nan
    array_detrended = np.zeros_like(array_in)
    array_trend = np.zeros_like(array_in[0])
    
    for i in range(array_interp.shape[1]):
        for j in range(array_interp.shape[2]):
            serie = array_in[:,i,j]
            nans, x = nan_helper(serie)
            serie[nans]= np.interp(x(nans), x(~nans), serie[~nans])

            x0 = np.linspace(0,10, len(array_in[:,0,0]))
            model = np.polyfit(x0, serie, 1)
            predicted = np.polyval(model, x0)
            array_detrended[:,i,j] = serie - predicted
            array_trend[i,j] = (predicted[11]-predicted[0])/(x0[11]-x0[0])

            
    return array_trend, array_detrended

def MK_test_df(df):
    
    df_out = pd.DataFrame(index = df.columns, columns = ['Trend', 'Slope', 'Z'])
    
    for columna in df.columns:
        test = list(MK_test(df[columna].values))
        test[1] = test[1] / len(df.index.year.unique()) # pass slpe to yearly [mm/year]
        df_out.loc[columna,:] = test
        
    return df_out

def MK_test(x,eps=0.0001, alpha=0.01):
    
    from scipy.special import ndtr, ndtri
    from scipy.stats import (linregress, norm, pearsonr, skew,  # kendalltau
                             spearmanr)

    # Se establece la cantidad de datos
    n = len(x)
    # Se crea una matriz nxn para aguardar los signos
    sgn = np.zeros((n, n), dtype="int")
    # Ciclo que va restar a cada fila[i] la posición x[i]
    # Quedará una matriz con una diagonal de ceros.
    for i in range(n):
            tmp = x - x[i]
            # Donde el valor absoluto de la diferencia se menor
            # o igual a eps coloque un cero 
            tmp[np.where(np.fabs(tmp) <= eps)] = 0.
            # Almacene los signos del proceso en la fila [i]
            sgn[i] = np.sign(tmp)
    # almacene en un vector todos los números que esté en cima
    # de la diagonal, desplazada en una unidad (k) y sumelos
    S = sgn[np.triu_indices(n, k=1)].sum()
       
    ############## Para calcular los empates ################
    # Se sacan las posiciones donde hubo empate
    
    # Se llena carga la diagonal de la matriz con el numero 10000
    np.fill_diagonal(sgn, 10000)
    # Se busca la posiciones (i,j) donde el valor es 0, es decir que
    # hubo empates
    i, j = np.where(sgn == 0.)
    # Se buscan los valores X[i] que tiene empates, y se saca una vector único
    empates = np.unique(x[i])
    # Se determina la cantidad de empates
    g = len(empates)
    # Se crea un vector con filas = num de empates
    t = np.zeros(g, dtype="int")
    for k in range(g):
        idx =  np.where(np.fabs(x - empates[k]) < eps)[0]
        t[k] = len(idx)
    # Se determinan los parámetros de la varianza
    sin_empates = n * (n - 1) * (2 * n + 5)
    con_empates = (t * (t - 1) * (2 * t + 5)).sum()
    # Se calcula la varianza
    varS = float(sin_empates - con_empates) / 18.
    # Se Calcula Zmk para la serie
    if S > 0:
        Zmk = (S - 1) / np.sqrt(varS)
    elif np.fabs(S) == 0:
        Zmk = 0.
    elif S < 0:
        Zmk = (S + 1) / np.sqrt(varS)
    
    # A continuación se analizan las hipótesis  
    
    # Se calcula la estadístico de comparación
    p = 2*(1-norm.cdf(abs(Zmk))) # two tail test
    Z_ = ndtri(1. - alpha/2.)
    # La función ndtri, devuelve el el estadístico para un área debajo
    # de la curva de la Normal
    
    #Se calcula la tendencia de la serie por medio de la regresión lineal
    x_o = np.linspace(0,10,len(x))
    #b_o, a_o, r_o, p_o, stderr  = linregress(x_o,x)

    model = np.polyfit(x_o, x, 1)
    predicted = np.polyval(model, x_o)
    b_o = (predicted[11]-predicted[0])/(x_o[11]-x_o[0])
    # Se evalúa si hay tendencia de alguna clase
    if np.fabs(Zmk) >= Z_:
        
        Z_1 = ndtri(1. - alpha)
      #  return np.sign(b_o),Zmk,b_o#,p,Zmk

        if Zmk >= Z_1:
         #    tendencia.append('Existe tendencia positiva')
             return 1.,b_o,Zmk

        else:

              return -1.,b_o, Zmk
        

    
    # No hay tendencia en la serie
    else:       
        return 0,np.round(b_o,2),Zmk
 
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

def corr_matrix(variables, nombres, dates):
    
    lista = np.array([spatial_mean(var) for var in variables]).T
    df = pd.DataFrame(lista, index=dates, columns=nombres)
    return df.corr()

def spatial_mean(arr_in):

    arr_in = np.array(arr_in) if isinstance(arr_in, list) else arr_in
    try:
        arr_out = np.nanmean(arr_in, axis=(1,2))
    except:
        arr_out = np.nanmean(arr_in, axis=-1)
    return arr_out

def plot_MK_test(out, array_in, tendencias,latitudes, longitudes, lims=[-5,5], cmap='BrBG'):
    
    test=[]

    for lat in range(array_in.shape[1]):
        for lon in range(array_in.shape[2]):
            test.append(MK_test(array_in[:,lat,lon])[0]) #yue_wang_modification_test

    mk_test = np.array(test).reshape(array_in.shape[1],array_in.shape[2])

    # Plot MK test     
    csfont = {'weight':'bold'}   
    reader = shpreader.Reader('data/HPA/HP.shp')
    fig ,ax= plt.subplots(1,2,figsize=(6,4),sharex=True, sharey=True,\
                          dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})
    
  #  ax[0].set_title('a) Trend',loc='left',fontsize=14, **csfont)
    mapa = ax[0].contourf(longitudes,latitudes, tendencias, cmap=cmap,vmax=lims[1],vmin=lims[0])
    
   # ax[1].set_title('b) (MK test)', loc='left',fontsize=14, **csfont)
    mapa = ax[1].contourf(longitudes, latitudes, mk_test, levels=[-1.5,-0.5,0.5,1.5],cmap='gist_rainbow')
    
    for axe in ax.flat:
        
        axe.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
           facecolor='none', edgecolor=[0,0,0,1],linewidth=1.5)
        
        axe.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
           name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=1)) 
        
        gl = axe.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-104,-102,-100,-98,-96],
              draw_labels=True,linewidth=2,color='k',alpha=0.3, linestyle='--')             
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylocator = mticker.FixedLocator([31,33,35,37,39,41,43,45])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
      
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
             norm=plt.Normalize(lims[0], lims[1])),
             ax=ax[0],orientation='horizontal',fraction=0.03, pad=0.07, label=r'$ mm / year$', extend='both')
    
    
    dic = {"+":1,"no trend":0,"-":-1}
    dic_inv =  { val: key for key, val in dic.items()  }
    
    cbar = fig.colorbar(mapa,ax=ax[1],orientation='horizontal',fraction=0.03, pad=0.07)
    cbar.set_ticks([1,0,-1])
    cbar.set_ticklabels([dic_inv[t] for t in [1,0,-1]])
    
    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none')

def LOESS_decomp_3D(array_in):
    
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

"**************************************************************************************************"

# GRACE

data_grace = Dataset('data/GRCTellus.JPL.200204_202211.GLO.RL06.1M.MSCNv03CRI.nc')
mask = np.genfromtxt('data/mask.csv', delimiter = ',')
dates_grace = pd.date_range('2002-04-15','2022-11-17', freq='M')
time_grace = np.array([datetime(2002,1,1) + timedelta(days = times) for times in np.array(data_grace["time"])])
_, lat_grace, lon_grace = read_grace(data_grace)
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


data_chirps = Dataset('data/chirps_monthly_gracegrid.nc')
precip = read_grace(data_chirps, 'precip')[0] / 10
time_chirps = np.array(data_chirps.variables['time'][:]) 
dates_chirps = np.array([datetime(1980,1,1) + timedelta(days = int(i)) for i in time_chirps])
dates_chirps = dates_chirps[:-2]
precip = precip[:-2]
dates_chirps = pd.date_range("1981-01", "2022-11", freq="M")

# ANOMALIES

TWSa = TWS - np.nanmean(TWS, axis=0)
SMa, SMa_model = SM - np.nanmean(SM_nldas, axis=0), SM_model - np.nanmean(SM_model, axis=0)
SWEa, SWEa_model = SWE - np.nanmean(SWE_nldas, axis=0), SWE_model - np.nanmean(SWE_model, axis=0)

# MODEL
# HERE WE MUST EXTRACT SEASONAL COMPONENT FROM GLDAS DATA IN ORDER TO OBTAIN GWA FROM MODEL
# WE USED LOESS DECOMPOSITION FOR CONSISTENCY

SWEa_model_residual = LOESS_decomp_3D(SWEa_model)[0] + LOESS_decomp_3D(SWEa_model)[2] # sumar tendencia y residual
SWEa_model_residual = np.reshape(SWEa_model_residual, (487, 24 * 18))

SMa_model_residual = LOESS_decomp_3D(SMa_model)[0] + LOESS_decomp_3D(SMa_model)[2] # sumar tendencia y residual
SMa_model_residual = np.reshape(SMa_model_residual, (487, 24 * 18))


GWa = TWSa - SMa - SWEa    # Ecuacion 1
GWa_model = TWS_model - SMa_model_residual - SWEa_model_residual

# variables = [TWSa, SMa, SWEa]#, SMa_model, SWEa_model]
# nombres = [r"TWSa", r"SMa", r"SWEa"]#, r"SMa model", r"SWEa model"]

# #plot corr
# fig, axes = plt.subplots(1,2,figsize=(12,3))
# sns.heatmap(corr_matrix(variables[:3], nombres[:3], dates_grace),ax=axes[0] , annot=True, cmap='Blues')
# sns.heatmap(corr_matrix(variables[3:], nombres[3:], dates_model),ax=axes[1] , annot=True, cmap='Blues')


# MK TEST
plot_MK_test('figures/01_TA_GWa.jpg', GWa, tendencia(GWa)[0] / 10, lat_grace, lon_grace)
plot_MK_test('figures/01_TA_TWSa.jpg', TWSa, tendencia(TWSa)[0] / 10, lat_grace, lon_grace, cmap='PuOr')
plot_MK_test('figures/01_TA_precip.jpg', precip[-247:], tendencia(precip[-247:])[0] / 10, lat_grace, lon_grace, cmap='PiYG', lims=[-0.1, 0.1])
plot_MK_test('figures/01_TA_SMa.jpg', SMa, tendencia(SMa)[0] / 10, lat_grace, lon_grace, cmap='PRGn', lims=[-1.5, 1.5])
plot_MK_test('figures/01_TA_SWEa.jpg', SWEa, tendencia(SWEa)[0] / 10, lat_grace, lon_grace, cmap='RdGy', lims=[-0.1, 0.1])
plot_MK_test('figures/01_TA_TWSa_model.jpg', TWS_model.reshape(487, 24, 18), tendencia(TWS_model.reshape(487, 24, 18))[0] / 10, lat_grace, lon_grace, cmap='PuOr', lims=[-0.5, 0.5])
plot_MK_test('figures/01_TA_GWa_model.jpg', GWa_model.reshape(487, 24, 18), tendencia(GWa_model.reshape(487, 24, 18))[0] / 10, lat_grace, lon_grace, lims=[-1.5, 1.5])
plot_MK_test('figures/01_TA_precip_model.jpg', precip, tendencia(precip)[0]/ 10, lat_grace, lon_grace, cmap='PiYG', lims=[-0.1, 0.1])
plot_MK_test('figures/01_TA_SMa_model.jpg', SMa_model.reshape(487, 24, 18), tendencia(SMa_model.reshape(487, 24, 18))[0] / 10, lat_grace, lon_grace, cmap='PRGn', lims=[-1.5, 1.5])

#%%
nombres = sorted(os.listdir('data/pozos'))

delimiters = [' ', ' ', '\t', '\t','\t','\t','\t','\t','\t',' ','\t','\t','\t','\t',' ', '\t','\t','\t', '\t']
col_date = [3,5,2,3,3,2,3,3,3,3,3,3,3,3,3,3,3,3,3]
col_val = [4,10,3,6,6,3,6,6,6,4,6,6,6,6,4,6,6,6,6]

df = pd.DataFrame(index=pd.date_range('01-01-1940', '31-10-2021', freq='M'))


for i, j in enumerate(nombres):
    serie = pd.read_csv(f'data/pozos/{nombres[i]}', delimiter=delimiters[i], header = None, \
                usecols = [col_date[i],col_val[i]], index_col=0,names = ['date' ,nombres[i].split('.')[0]], infer_datetime_format = True)
    serie.index = pd.to_datetime(serie.index, errors='coerce')
    serie = serie.resample('M').mean()
    
    df = df.join(serie)
    
df

ft_to_mm = 304.8
s_y = -0.15

df_anom = (df*ft_to_mm - (df*ft_to_mm).mean()) * s_y
#df_anom = s_y * ft_to_cm * df.diff()

df_anom = df_anom.interpolate('linear', limit_direction = 'both')
df_anom = df_anom['1979':'06-2021']

df_anom = df_anom.drop('NE04', axis=1)

mann_kendall = MK_test_df(df_anom)
#mann_kendall = MK_test_df(df_gmean_season + df_gmean_residuales)

print(mann_kendall)

locacion = pd.read_excel('data/loc_pozos.xls', index_col=0)
locacion = locacion.drop('NE04')

s = mann_kendall['Slope'].values.astype('float16')

# Zoom
locacion_zoom = locacion['NE01':'NE12']
s_zoom = mann_kendall['Slope']['NE01':'NE12'].values.astype('float16')

names = df_anom.columns
names_zoom = names[1:10]
names = np.setdiff1d(names, names_zoom)

df_trend, df_season, df_residual = deseasonal_df(df_anom)
df_des = df_trend + df_residual



#%%

# trend figure}


csfont = {'fontname':'Arial', 'fontsize' : 12} 

array_in = GWa
longitudes = lon_grace
latitudes = lat_grace
trends = tendencia(GWa)[0] / 10


fig = plt.figure(figsize=[7,6], dpi=300)
spec = gridspec.GridSpec(ncols=5, nrows=2, figure=fig)
ax1 = fig.add_subplot(spec[:, :3], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(spec[0, 3:], projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(spec[1, 3:], projection=ccrs.PlateCarree())

test=[]

longitudes_wells = locacion["LONGITUDE"].values
latitudes_wells = locacion["LATITUDE"].values

longitudes_zoom = locacion_zoom["LONGITUDE"].values
latitudes_zoom = locacion_zoom["LATITUDE"].values

reader = shpreader.Reader('data/HPA/HP.shp')

ax1.scatter(longitudes_wells, latitudes_wells, np.abs(s)*10, c= s, norm=plt.Normalize(-7,7),cmap='RdYlGn', zorder=5)
ax1.set_extent([-105, -97, 32, 44])

style = {'fontname':'Arial', 'fontsize' : 6, 'rotation' : 20, 'weight': 'bold'} 

for i, txt in enumerate(names):
    ax1.annotate(txt, (np.setdiff1d(longitudes_wells, longitudes_zoom, assume_unique=True)[i] - .3, \
                        np.setdiff1d(latitudes_wells, latitudes_zoom, assume_unique=True)[i]+ .28) , **style, zorder=6)
    
ax_zoomin = ax1.inset_axes([-0.1, 0.45, 0.4, 0.3])
ax_zoomin.set_xlim([-102.2, -100.5])
ax_zoomin.set_ylim([40, 41.2])
ax_zoomin.scatter(longitudes_zoom, latitudes_zoom, np.abs(s_zoom)*10, c= s_zoom, norm=plt.Normalize(-7,7),cmap='RdYlGn', zorder=5)

ax_zoomin.xaxis.set_visible(False)
ax_zoomin.yaxis.set_visible(False)

ax_zoomin.set_facecolor('tan')
for i, txt in enumerate(names_zoom):
    ax_zoomin.annotate(txt, (longitudes_zoom[i]-.03, latitudes_zoom[i]+.05), **style, zorder=6)

ax1.indicate_inset_zoom(ax_zoomin, edgecolor="black", lw=2)

for lat in range(array_in.shape[1]):
        for lon in range(array_in.shape[2]):
            test.append(MK_test(array_in[:,lat,lon], alpha=0.05)[0]) #yue_wang_modification_test

mk_test = np.array(test).reshape(array_in.shape[1],array_in.shape[2])


#ax2.set_title('b)',loc='left', **csfont)
mapa = ax2.contourf(longitudes,latitudes, trends, cmap='BrBG',vmax=5,vmin=-5, zorder=0)

#ax3.set_title('c)', loc='left', **csfont)
mapa = ax3.contourf(longitudes, latitudes, mk_test, levels=[-1.5,-0.5,0.5,1.5],cmap='gist_rainbow', zorder=0)

for axe in [ax1, ax2, ax3]:
        
    axe.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
        facecolor='none', edgecolor=[0,0,0,1],linewidth=0.7, zorder=1)
    
    axe.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.4), zorder=3) 
    
    gl = axe.gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(-98, -104, 4,dtype=int),
            draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')             
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylocator = mticker.FixedLocator(np.linspace(31 - 1, 42-1, 4,dtype=int))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = csfont
    gl.xlabel_style = csfont

ax1.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
        facecolor='tan', edgecolor=[0,0,0,1],linewidth=0.01, zorder=2)

cbar_ax2 = fig.colorbar(plt.cm.ScalarMappable(cmap='BrBG',norm=plt.Normalize(-5, 5)), ax=ax2,
             orientation='horizontal',fraction=0.03, pad=0.11, extend='both')

cbar_ax2.set_label("mm / year", labelpad=0.001, font='Arial', fontsize=11)
#cbar_ax2.set_ticklabels([-5, 0, 5], **csfont)

cbar_ax1 = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-7, 7)),ax=ax1, \
                        fraction=0.065, extend='both', pad=0.05)
cbar_ax1.set_label("mm/year", labelpad=0.01, font='Arial', fontsize=12, rotation=90)

dic = {"+":1,"no trend":0,"-":-1}
dic_inv =  { val: key for key, val in dic.items()  }

cbar_ax3 = fig.colorbar(mapa,ax=ax3,orientation='horizontal',fraction=0.03, pad=0.12)
cbar_ax3.set_ticks([1,0,-1])
cbar_ax3.set_ticklabels([dic_inv[t] for t in [1,0,-1]], **csfont)

ax3.text(0.07, 0.86, "a)", transform=fig.transFigure, **csfont)
ax3.text(0.63, .86, "b)", transform=fig.transFigure, **csfont)
ax3.text(0.63, 0.46, "c)", transform=fig.transFigure, **csfont)

#fig.tight_layout()

fig.subplots_adjust(wspace=2.3)

fig.savefig('figures/01_TA_GWa.pdf', dpi=200, edgecolor='none', bbox_inches='tight')