#%%

from tkinter.messagebox import RETRYCANCEL
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy import linalg
import scipy.signal as signal
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')


"********************************************************************************************"

# FUNCIONES


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

def read_gldas(data_gldas):

    lat_gldas_ = np.array(data_gldas.variables['lat'][:])
    lon_gldas_ = np.array(data_gldas.variables['lon'][:])
    SWE_ = np.array(data_gldas.variables['SWE_inst'][:])
    SM1 = np.array(data_gldas.variables['SoilMoi0_10cm_inst'][:])
    SM2 = np.array(data_gldas.variables['SoilMoi10_40cm_inst'][:])
    SM3 = np.array(data_gldas.variables['SoilMoi40_100cm_inst'][:])
    SM4 = np.array(data_gldas.variables['SoilMoi100_200cm_inst'][:])
    time_gldas = np.array(data_gldas.variables['time'][:])

    SM_ = SM1 + SM2 + SM3 + SM4

    latglN = np.where(lat_gldas_ == 44.25)[0][0] # 44.25 43.75
    latglS = np.where(lat_gldas_ == 32.25)[0][0] # 32.25 31.75
    longlE = np.where(lon_gldas_ == 263.75)[0][0] #263.75
    longlW = np.where(lon_gldas_ == 254.75)[0][0] #264.75

    lat_gldas = lat_gldas_[latglN:latglS]#[::-1]
    lon_gldas = lon_gldas_[longlW:longlE]

    SWE_ = SWE_[:,latglN:latglS,longlW:longlE] / 10
    SM_ = SM_[:,latglN:latglS,longlW:longlE] / 10

    SM = SM_[279:,::-1,:]
    SWE = SWE_[279:,::-1,:]

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

    dates_gldas = np.array([datetime(1948,1,1) + timedelta(days = time_gldas[i]) for i in range(len(time_gldas))])

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

def anomalies(arr_in, in_mean=0):
    arr_out = np.zeros_like(arr_in)*np.nan
    for la in range(arr_in.shape[1]):
        for lo in range(arr_in.shape[2]):
            arr_out[:,la,lo] = arr_in[:,la,lo] - np.nanmean(arr_in[in_mean:in_mean+72,la,lo])
    return arr_out

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

def detrend(serie, axis):
    
    serie_out = signal.detrend(serie, axis, type='constant')

    return serie_out

def tendencia(array_in):

    array_detrended = np.zeros_like(array_in)
    array_trend = np.zeros_like(array_in[0])
    
    for i in range(array_in.shape[1]):
        for j in range(array_in.shape[2]):
            serie = array_in[:,i,j]
            x0 = np.linspace(0,10, len(serie))
            model = np.polyfit(x0, serie, 1)
            predicted = np.polyval(model, x0)
            array_detrended[:,i,j] = serie - predicted
            array_trend[i,j] = (predicted[1]-predicted[0])/(x0[1]-x0[0])
            
    return array_trend, array_detrended

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interp3D(data, kind='linear'):

    data_interp = np.zeros_like(data)

    for la in range(data.shape[1]):
        for lo in range(data.shape[2]):

            y = data[:,la, lo].copy()
            nans, x= nan_helper(y)
            f = interp1d(x(~nans), y[~nans], kind=kind, copy=False)
            y[nans] = f(x(nans))
            data_interp[:, la, lo] = y

    return data_interp

def plot_EOFs(out, matrix, lat, lon, num, title,labeli, vals, dates_complete, grid=(1,4), \
                fig_size=(8,4), mapa='bwr_r' ,title_num=None,factor=None, points=None):
    
    csfont = {'fontname':'Arial', 'fontsize':12}
    reader = shpreader.Reader('data/HPA/HP.shp')
    lon_ = lon - 360
    fig ,ax= plt.subplots(grid[0] , grid[1], figsize=fig_size,sharex=True, sharey=True,\
                          dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})

    fig = plt.figure(figsize=fig_size, dpi=200)

    for i in range(num):
        
        ax = fig.add_subplot(2, 2, num, projection=ccrs.PlateCarree())
        eof = matrix[:,i].reshape(lat.shape[0], lon.shape[0])

        ax.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
               facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
        ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
          name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
        ax.contourf(lon,lat, eof*mask, cmap=mapa[i],vmax=vals[i],vmin=0)

        gl = ax.gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
                  draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 4, dtype=int))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'fontname':'Arial', 'fontsize': 10}
        gl.xlabel_style = {'fontname':'Arial', 'fontsize': 10}        
        # fig.colorbar(plt.cm.ScalarMappable(cmap='bwr_r', \
        #          norm=plt.Normalize(-1*vals[i],vals[i])),ax=ax.flat[i],orientation='horizontal',\
        #         fraction=0.046, pad=0.08, label=labeli,extend='both')

        if title_num == None:
            ax.set_title(title[i] ,loc='left',**csfont)
        if title_num == 'sequence':
            ax.set_title(title + '' + str(i+1) ,loc='left', **csfont)
        elif title_num == 'lags':
            if factor is not None:
                ax.set_title(title + '' + str((i - num // 2) * factor) ,loc='left', **csfont)
            else:
                ax.set_title(title + '' + str(i - num // 2) ,loc='left', **csfont)
            
        if points is not None:
            points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
            ax.scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())

        fig.colorbar(plt.cm.ScalarMappable(cmap=mapa[i], \
            norm=plt.Normalize(0, vals[i])),ax=ax,orientation='horizontal',\
            fraction=0.046, pad=0.08, label=labeli[i],extend='both', )
    
#     axes = fig.add_subplot(2, 1, 2)
#    # ax.change_geometry(2, 1, 1)
#     # axes = fig.add_axes([0.15, 0.0, 0.8, 0.3])
#     axes.fill_between(dates_complete, -15, 15, where = mask_nan, color='gray', alpha=0.3)
#     axes.plot(dates_complete, estandarizar(serie_TWS_pad), label='Grace', c='r')
#     axes.plot(dates_complete, estandarizar(serie_model_pad), label='H&G mean', c='k', ls=':')
#     axes.plot(dates_complete, estandarizar(serie_modelo_complete_pad), label='fitted H&G', c='k')
#     axes.set_xlabel('time')
#     axes.set_ylabel('TWS anomaly [cm]')
#     axes.set_xlim(dates_complete[0], dates_complete[-1])
#     axes.set_ylim(-15, 15)
#     axes.legend()
#     axes.spines['top'].set_visible(False)
#     axes.spines['right'].set_visible(False)
#     axes.grid(alpha=0.3)
 #   fig.subplots_adjust(bottom=0.3)
    # cbar_ax = fig.add_axes([0.1, 0.0001, 0.8, 0.01])
    # cbar_ax.grid(False)
    # cbar_ax.axis('off')
    # [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
    # [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
    # cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none')

def estandarizar(df, axis=-1):
    dfo = df - np.nanmean(df, axis = axis) / np.nanstd(df, axis = axis)
    return dfo


"**************************************************************************************************"

# GRACE

#data_grace = Dataset('data/GRCTellus.JPL.200204_202107.GLO.RL06M.MSCNv02CRI.nc')
data_grace = Dataset('data/GRCTellus.JPL.200204_202211.GLO.RL06.1M.MSCNv03CRI.nc')
mask = np.genfromtxt('data/mask.csv', delimiter = ',')

#dates_grace = pd.date_range('2002-04-15','2021-07-17', freq='M')
dates_grace = pd.date_range('2002-04-15','2022-11-17', freq='M')

TWS, lat_grace, lon_grace = read_grace(data_grace)

ix_nan = np.argwhere(np.isnan(TWS[:,0,0]))[:, 0]
mask_nan = np.in1d(range(TWS.shape[0]), ix_nan)
mask_nan_m = mask_nan[:208]

dates_complete = pd.to_datetime(pd.date_range('1979-1','2022-11', freq='M'))
ix_start_grace = 279

n_features = TWS.shape[1] * TWS.shape[2]

# MODELO - Parte comentada une los ensembles del modello en archivo .npy

# ensembles_list = sorted(os.listdir('data/model_JPL_ERA5/'))

# TWS_model = np.zeros((len(ensembles_list), time_model.shape[0], TWS.shape[1], TWS.shape[2]))
# for i, file in enumerate(ensembles_list):
#     data_model = Dataset(f'data/model_JPL_ERA5/{file}')
#     TWS_model[i, :, :, :] = read_model(data_model)

# print(TWS_model.shape)

# with open('data/ensembles.npy', 'wb') as f:
#     np.save(f, TWS_model)

data_model = Dataset('data/model_JPL_ERA5/GRACE_REC_v03_JPL_ERA5_monthly_ens082.nc')
TWS_model_best = read_model(data_model)
time_model = np.array(data_model.variables['time'][:]) 
dates_model = np.array([datetime(1901,1,1) + timedelta(days = time_model[i]) for i in range(len(time_model))])
dates_model_grace = dates_model[ix_start_grace:]

ix_end_model = 208

with open('data/ensembles.npy', 'rb') as f:
    TWS_model0 = np.load(f)

TWS_model = np.nanmean(TWS_model0, axis=0)
TWS_model_std = np.nanstd(TWS_model0, axis=0, ddof=1)


# INTERPOLACIÓN, REMOCIÓN DE TENDENCIA LINEAL Y COMPONENTE ESTACIONAL DE TWS DE GRACE

TWS_interp = interp3D(TWS, kind='linear')
TWS_interp2 = interp3D(TWS, kind='quadratic')
TWS_interp3 = interp3D(TWS, kind='cubic')

fig, axes = plt.subplots(figsize=(10,3), dpi=200)

axes.fill_between(dates_grace, -15, 15, where = mask_nan, color='gray', alpha=0.3)
axes.plot(dates_grace, (np.nanmean(TWS_interp, axis=(1,2))), label='linear', c='r')
axes.plot(dates_grace,(np.nanmean(TWS_interp2, axis=(1,2))), label='quadratic', c='k', ls=':')
axes.plot(dates_grace,(np.nanmean(TWS_interp3, axis=(1,2))), label='cubic', c='c', ls=':')
axes.set_xlabel('time')
axes.set_ylabel('TWS anomaly [cm]')
axes.set_xlim(dates_grace[0],dates_grace[-1])
axes.set_ylim(-15, 15)
axes.legend()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/interp_TWS.jpg', dpi=100, transparent=False, edgecolor='none')

TWS_trend, TWS_detrend = tendencia(TWS_interp)
TWS_residual = deseasonal(TWS_detrend)[0] + deseasonal(TWS_detrend)[2] # sumar tendencia y residual
TWS_seasonal_component = deseasonal(TWS_detrend)[1]

# CORTE TEMPORAL DE GRACE Y MODELO PARA ENCAJAR

TWS_model_grace = TWS_model0[:,ix_start_grace:, :, :]
TWS_grace_m = TWS_residual[:ix_end_model, :, :]

TWS_model_grace = TWS_model_grace.reshape(100, TWS_model_grace.shape[1], n_features)
TWS_grace_m = TWS_grace_m.reshape(TWS_grace_m.shape[0], n_features)

print(TWS_grace_m.shape, TWS_model_grace.shape)

# MODELO DE REGRESION RIDGE

modelo = make_pipeline(StandardScaler(with_mean=True), RidgeCV(

            alphas          = np.logspace(-6, 6, 1000),
            fit_intercept   = True,
            store_cv_values = True)
            )


coefs = np.zeros((TWS_model_grace.shape[-1]))*np.nan
gaps = np.zeros((ix_nan.shape[0], TWS_model_grace.shape[-1]))*np.nan
rmse_ridge = np.zeros((TWS_model_grace.shape[-1]))*np.nan
r2_ridge = np.zeros((TWS_model_grace.shape[-1]))*np.nan
modelo_complete = np.zeros((dates_model.shape[0], TWS_model_grace.shape[-1]))*np.nan

TWS_model_complete = TWS_model0.reshape(100, dates_model.shape[0], n_features)

# AJUSTE DEL MODELO Y PREDICCION DE GAPS Y HACIA EL PASADO

for pixel in range(n_features):

    X = TWS_model_grace[:, ~mask_nan_m, pixel].T
    y = TWS_grace_m[~mask_nan_m, pixel]
    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y.reshape(-1,1),
                                            train_size   = 0.8,
                                            shuffle      = True,
                                            random_state=10
                                        )

    modelo.fit(X_train, y_train)

    modelo["standardscaler"].transform(modelo["ridgecv"].coef_.reshape(1, -1))

    predicciones = modelo.predict(X=X_test)
    predicciones = predicciones.flatten()

    gaps[:, pixel] = modelo.predict(TWS_model_grace[:, mask_nan_m, pixel].T).flatten()
    
    modelo_complete[:, pixel] = modelo.predict(TWS_model_complete[:, :, pixel].T).flatten()

    rmse_ridge[pixel] = mean_squared_error(
                y_true  = y_test,
                y_pred  = predicciones,
                squared = False
             )


    r2_ridge[pixel] = r2_score(y_test.flatten(), predicciones)

print("")
print(f"El error (rmse) de test es: {np.mean(rmse_ridge)}")
print(f"el r2 del test es: {np.mean(r2_ridge)}")


# LLENADO DE GAPS

TWS_complete = TWS_residual.copy()
TWS_complete[ix_nan] = gaps.reshape(33, 24, 18)

# LLENADO DE SERIES TEMPORALES CON NANS PARA GRAFICACION

serie_model = np.mean(TWS_model, axis=(1,2))
serie_model_pad = np.pad(serie_model, (0, len(dates_complete) - len(serie_model)), constant_values = np.nan)

serie_TWS = np.mean(TWS_complete, axis=(1,2))
serie_TWS_pad = np.pad(serie_TWS, (ix_start_grace, 0), constant_values = np.nan)

serie_modelo_complete = np.mean(modelo_complete, axis=1)
serie_modelo_complete_pad = np.pad(serie_modelo_complete , (0, len(dates_complete) - len(serie_model)), constant_values = np.nan)

ix_list = np.arange(0, len(dates_complete))
mask_nan = np.in1d(range(len(dates_complete)), ix_nan + ix_start_grace)
mask_nan[:ix_start_grace] = True

# CONFIDENCE BOUNDS

sem = TWS_model_std / np.sqrt(TWS_model0.shape[0])

# Calculate the t-value for the desired confidence level (e.g. 95%)
from scipy.stats import t
t_value = t.ppf(1 - 0.05/2, TWS_model0.shape[0]-1)

# Calculate the confidence interval
lower_ci = TWS_model - t_value * sem
upper_ci = TWS_model + t_value * sem

lower_ci, upper_ci = np.mean(lower_ci, axis=(1,2)), np.mean(upper_ci, axis=(1,2))

lower_ci = np.pad(lower_ci, (0, len(dates_complete) - len(lower_ci)), constant_values=np.nan)
upper_ci = np.pad(upper_ci, (0, len(dates_complete) - len(upper_ci)), constant_values=np.nan)

# GRAFICACION

# fig, axes = plt.subplots(figsize=(10,3), dpi=200)

# axes.fill_between(dates_complete, -15, 15, where = mask_nan, color='gray', alpha=0.3)
# axes.plot(dates_complete, estandarizar(serie_TWS_pad), label='Grace', c='r')
# axes.plot(dates_complete, estandarizar(serie_model_pad), label='H&G mean', c='k', ls=':')
# axes.plot(dates_complete, estandarizar(serie_modelo_complete_pad), label='fitted H&G', c='k')
# axes.set_xlabel('time')
# axes.set_ylabel('TWS anomaly [cm]')
# axes.set_xlim(dates_complete[0], dates_complete[-1])
# axes.set_ylim(-15, 15)
# axes.legend()
# axes.spines['top'].set_visible(False)
# axes.spines['right'].set_visible(False)
# axes.grid(alpha=0.3)
# fig.tight_layout()
# fig.savefig('figures/model_series.jpg', dpi=100, transparent=False, edgecolor='none')


# scoress = np.array([rmse_ridge, r2_ridge])
# plot_EOFs('figures/model_rmse.pdf', scoress.T, lat_grace, lon_grace, 2, [r'RSME', r'R²'],['[cm]', ''], \
#     [4, 1], dates_complete, grid=(1,2), fig_size=(6,4), mapa=['Reds', 'Blues'],title_num=None,factor=None, points=None)


# ADICION DEL COMPONENTE ESTACIONAL A GAPS, FILL FINAL Y EXPORTACION DE TWS EN .npy

ix_nan_gap = ix_nan[20:31]
gaps_season = gaps.reshape(33, 24, 18) #+ TWS_seasonal_component[ix_nan]

TWS_final = TWS_interp2.copy()
TWS_final[ix_nan_gap] = gaps_season[20:31]
TWS_final = TWS_final.reshape(TWS_complete.shape[0], TWS_complete.shape[1] , TWS_complete.shape[2])

with open('data/tws_fill.npy', 'wb') as f:
     np.save(f, TWS_final)

with open('data/fit_model.npy', 'wb') as f:
     np.save(f, modelo_complete)

#%%

def plot_EOFs(out, matrix, lat, lon, num, title,labeli, vals, dates_complete, \
              serie_TWS_pad, serie_model_pad, serie_modelo_complete_pad, grid=(1,4), \
                fig_size=(6,4), mapa='bwr_r', points=None):
    
    csfont = {'fontname':'Arial', 'fontsize' : 12}
    reader = shpreader.Reader('data/HPA/HP.shp')
    lon_ = lon - 360
    fig ,ax= plt.subplots(grid[0] , grid[1], figsize=fig_size,sharex=True, sharey=True,\
                          dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})

    #fig = plt.figure(figsize=fig_size, dpi=200)

    for i in range(num):
        
       # ax = fig.add_subplot(2, 2, num, projection=ccrs.PlateCarree())
        eof = matrix[:,i].reshape(lat.shape[0], lon.shape[0])

        ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
               facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
        ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
          name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
        ax.flat[i].contourf(lon,lat, eof*mask, cmap=mapa[i],vmax=vals[i],vmin=0)

        ax.flat[i].set_title(title[i], loc='left')
        gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=np.linspace(min(lon_) - 1, max(lon_)-1, 3,dtype=int),
                  draw_labels=True,linewidth=1,color='k',alpha=0.3, linestyle='--')          
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator(np.linspace(min(lat)+1, max(lat) -1, 4, dtype=int))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = csfont#{'fontname':'times', 'weight': 'bold', 'fontsize': 9}
        gl.xlabel_style = csfont#{'fontname':'times', 'weight': 'bold', 'fontsize': 9}        
        # fig.colorbar(plt.cm.ScalarMappable(cmap='bwr_r', \
        #          norm=plt.Normalize(-1*vals[i],vals[i])),ax=ax.flat[i],orientation='horizontal',\
        #         fraction=0.046, pad=0.08, label=labeli,extend='both')

        if points is not None:
            points = points.reshape(points.shape[0], points.shape[1], lat.shape[0], lon.shape[0])
            ax.flat[i].scatter((points[i, 1, :, :])*mask, (points[i, 0, :, :])*mask, color="k",s=1,alpha=0.5, transform=ccrs.PlateCarree())

        c_bar = fig.colorbar(plt.cm.ScalarMappable(cmap=mapa[i], \
            norm=plt.Normalize(0, vals[i])),ax=ax.flat[i],orientation='vertical',\
            fraction=0.056, pad=0.08, label=labeli[i],extend='both', )
        c_bar.set_label(labeli[i], font = 'Arial', size=12)

 #   ax_1 = fig.add_subplot(111)
 #   ax_1.change_geometry(2, 1, 1)
  #  ax_1.set_axis_off()
    axes = fig.add_axes([0.13, 0.0001, 0.8, 0.2])
   # axes = fig.add_subplot(2, 1, 2)
    axes.fill_between(dates_complete, -15, 15, where = mask_nan, color='gray', alpha=0.3)
 #   axes.fill_between(dates_complete, lower_ci, upper_ci, color='blue', alpha=0.4)
    axes.plot(dates_complete, estandarizar(serie_TWS_pad), label='TWSa-GRACE', c='r')
    axes.plot(dates_complete, estandarizar(serie_model_pad), label='TWSa-H&G mean', c='k', ls=':')
    axes.plot(dates_complete, estandarizar(serie_modelo_complete_pad), label='TWS-Model', c='k')
#    axes.set_title('c)', loc='left', **csfont)
    axes.set_xlabel('Time', **csfont)
    axes.set_ylabel('TWS anom [cm]', **csfont)
    axes.set_xlim(dates_complete[0], dates_complete[-1])
    axes.set_ylim(-15, 15)
    axes.legend(prop={'family' : 'Arial', 'size' : 8})
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.grid(alpha=0.3)
   # fig.subplots_adjust(bottom=0.1)
    # cbar_ax = fig.add_axes([0.1, 0.0001, 0.8, 0.01])
    # cbar_ax.grid(False)
    # cbar_ax.axis('off')
    # [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
    # [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
    # cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

    axes.text(0.01, 0.7, "a)", transform=fig.transFigure, **csfont)
    axes.text(0.52, 0.7, "b)", transform=fig.transFigure, **csfont)
    axes.text(0.01, 0.25, "c)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=False, edgecolor='none', bbox_inches='tight')

plot_EOFs('figures/model_rmse.pdf', scoress.T, lat_grace, lon_grace, 2,
           [f'RSME = {np.round(np.mean(rmse_ridge), 2)}', f'R$^{{{2}}}$ = {np.round(np.mean(r2_ridge), 2)}'],['[cm]', ''], \
    [4, 1], dates_complete, serie_TWS_pad, serie_model_pad, serie_modelo_complete_pad, grid=(1,2), \
        fig_size=(6,6), mapa=['Reds', 'Blues'])

