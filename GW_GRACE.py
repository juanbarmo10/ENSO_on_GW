
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

    dates_gldas = np.array([init_date + timedelta(days = time_gldas[i]) for i in range(len(time_gldas))])

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

def read_oni(data, dates):

    data.columns=range(0,12)
    t = data_oni.stack()
    year, month = t.index.get_level_values(0).values, t.index.get_level_values(1).values
    t.index = pd.PeriodIndex(year=year, month=month+1, freq='M', name='Fecha')

    return estandarizar(pd.Series(t.loc[dates[0]:dates[1]],name='ONI'))

def read_pdo(data, dates):

    data.index = pd.to_datetime(data.index, format='%Y%m', errors = "ignore")
    return estandarizar(pd.Series(data['Value'].loc[dates[0]:dates[1]],index=data.loc[dates[0]:dates[1]].index))

def estandarizar(df):
    dfo = (df - df.mean())/df.std()
    return dfo

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
    
    serie_out = signal.detrend(serie, axis, type='linear')
    
    return serie_out

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
           # array_interp[:,i,j] = serie

            x0 = np.linspace(0,10, len(array_in[:,0,0]))
      #      y = array_interp[:,lat,lon]
            model = np.polyfit(x0, serie, 1)
            predicted = np.polyval(model, x0)
            array_detrended[:,i,j] = serie - predicted
            array_trend[i,j] = (predicted[11]-predicted[0])/(x0[11]-x0[0])
            
    return array_trend, array_detrended

def basic_stats(array, variable_name="", decimals=2):

    line = f"""{variable_name}
       Mean: {str(np.round(np.nanmean(array), decimals))}
       Maximun: {str(np.round(np.nanmax(array), decimals))}
       Minimun: {str(np.round(np.nanmin(array), decimals))}
       Standard Deviation: {str(np.round(np.nanstd(array), decimals))}"""
    
    print(line)

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

def spatial_mean(arr_in):

    arr_in = np.array(arr_in) if isinstance(arr_in, list) else arr_in
    arr_out = np.nanmean(arr_in, axis=-1)
    return arr_out
        
def corr_matrix(variables, nombres, dates):
    
    lista = np.array([spatial_mean(var) for var in variables]).T
    df = pd.DataFrame(lista, index=dates, columns=nombres)
    return df.corr()

def MK_test(x,eps=0.0001, alpha=0.01):
    
    from scipy.special import ndtri, ndtr
    from scipy.stats import norm, skew, linregress, spearmanr, pearsonr# kendalltau
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
    b_o, a_o, r_o, p_o, stderr  = linregress(x_o,x)
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
 
def MK_test_df(df):
    
    df_out = pd.DataFrame(index = df.columns, columns = ['Trend', 'Slope', 'Z'])
    
    for columna in df.columns:
        df_out.loc[columna,:] = MK_test(df[columna].values)
        
    return df_out

def red_noise_spectrum(serie, significance):

    periodos = FFT(serie)[0]
    periodos = periodos[1:]

    r = acf(serie)[:2]
    S0 = (4 / len(serie)) / (1 + r[1]**2 - 2 * r[1] * np.cos((2 * np.pi) / periodos))

    critical = chi2.ppf(1 - significance, 2)

    return (S0 * critical * (1 - significance)) / 2

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

def fourier_filter(serie, freq_min=24, freq_max=84):
            
    Freq = np.fft.fftfreq(len(serie), d=1) 
    Fpos = Freq[ Freq >= 0 ]
    fourier  = np.fft.fft(serie, norm="forward" )

    fourier[np.where((np.abs(1/  Freq) > (freq_max)) | (np.abs(1 / Freq) < (freq_min)))] = 0
    inverse = np.fft.ifft(fourier)

    periodos = 1 / Fpos
    power = abs(fourier)**2

    return periodos, power[:len(Fpos)], inverse

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
        matrix = np.corrcoef(array.T)
       # print('Correlations shape ' + str(comatrixrr.shape))
        val, vec = linalg.eig(matrix) 
        # PC are obtained from dot product from original matrix and eigenvectors
        PC = (array - np.nanmean(array, axis=0) / np.std(array, axis=0)) @ vec

        PCscore = PC @ np.diag(np.sqrt(val[:]))

        loadings =  PCscore.T / len(array) @ array

        # Sum of squared loading values give variance explained by PC (eigenvalue)
        print('Variance explained by loading 1' , np.sum(loadings[:,0]**2))
        print('Variance explained by loading 2' , np.sum(loadings[:,1]**2))
    

        return loadings, PCscore, val

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

def significanceP(corrmap,lat,lon):
    lat1 = []
    lon1 = []

    for la in range(lat.shape[0]):
        for lo in range(lon.shape[0]):
            if corrmap.reshape(lat.shape[0], lon.shape[0])[la,lo] <= 0.05:
                lat1.append(lat[la]), lon1.append(lon[lo])
    return lat1, lon1

def spatial_plot(out, variables, nombres, dates, mascara=1):
    
    fig, axes = plt.subplots(figsize=(8,3), dpi=200)
    
    for i in range(len(variables)):
        axes.plot(dates, spatial_mean(variables[i]), label=nombres[i], linewidth=1)
    
    axes.set_xlabel(r'$Time$')
    axes.set_ylabel(r'$Anomalies [cm]$')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.grid(color='gray', linestyle='--', alpha=0.3)
    axes.legend(loc=3, ncol=4)

    fig.savefig(out, dpi=100, transparent=False, edgecolor='none')

def plot_MK_test(out, array_in, tendencias,latitudes, longitudes):
    
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
    mapa = ax[0].contourf(longitudes,latitudes, tendencias, cmap='BrBG',vmax=5,vmin=-5)
    
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
      
    fig.colorbar(plt.cm.ScalarMappable(cmap='BrBG', \
             norm=plt.Normalize(-5, 5)),
             ax=ax[0],orientation='horizontal',fraction=0.03, pad=0.07, label=r'$ cm / year$', extend='both')
    
    
    dic = {"+":1,"no trend":0,"-":-1}
    dic_inv =  { val: key for key, val in dic.items()  }
    
    cbar = fig.colorbar(mapa,ax=ax[1],orientation='horizontal',fraction=0.03, pad=0.07)
    cbar.set_ticks([1,0,-1])
    cbar.set_ticklabels([dic_inv[t] for t in [1,0,-1]])
    
    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none')

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

def plot_FFT(out, PC_score,  oni, pdo, labels, numero_PCs=2, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(numero_PCs+2,1, sharex=True, figsize=(4,6), dpi=200)

    csfont = {'fontname':'Arial', 'fontsize' : 12}
    labeli = [r'$ONI$', r'$PDO$']

    for i in range(numero_PCs):
        
        periods = func(PC_score[:,i])[0]
        powers = estandarizar(func(PC_score[:,i])[1])

 #       ax[i].plot(periods[1:], powers[1:], label=r'$PC$ '+str(i+1), c='darkblue')
        ax[i].plot(periods[1:], powers[1:], label=labels[i], c='darkblue')
        ax[i].plot(periods[1:], red_noise_spectrum(PC_score[:, i], 0.05), c='firebrick', ls='--')

    
    if func == fourier_filter:

        ax[numero_PCs].plot(func(oni[:])[0][1:], estandarizar(func(oni, freq_min, freq_max)[1])[1:], label=labeli[0], c='darkblue')
        ax[numero_PCs+1].plot(func(pdo[:])[0][1:], estandarizar(func(pdo, freq_min, freq_max)[1])[1:], label=labeli[1], c='darkblue')
        ax[numero_PCs].plot(periods[1:], red_noise_spectrum(oni, 0.05), c='firebrick', ls='--')
        ax[numero_PCs+1].plot(periods[1:], red_noise_spectrum(pdo, 0.05), c='firebrick', ls='--')
    else:
        ax[numero_PCs].plot(func(oni)[0][1:], estandarizar(func(oni)[1])[1:], label=labeli[0], c='darkblue')
        ax[numero_PCs+1].plot(func(pdo)[0][1:], estandarizar(func(pdo)[1])[1:], label=labeli[1], c='darkblue')
        ax[numero_PCs].plot(periods[1:], red_noise_spectrum(oni, 0.05), c='firebrick', ls='--')
        ax[numero_PCs+1].plot(periods[1:], red_noise_spectrum(pdo, 0.05), c='firebrick', ls='--')
    
    for i in range(len(ax)):
        
        ax[i].axvline(12*2,linestyle='--', color='orange', alpha=0.4)
        ax[i].axvline(12*7, linestyle='dashed', color='orange', alpha=0.4)
        ax[i].legend(loc=0, frameon=False) 
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(color='gray', linestyle='--', alpha=0.3)
     #   ax[i].set_xscale('log')
        ax[i].set_ylim(0, 10)
        ax[i].set_xlim(0, 120)

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    ax[-1].set_xlabel(r'$Period \, [Months]$', **csfont)
    axe.get_xaxis().set_visible(False)
    plt.ylabel(r'$Stand. \, power$', labelpad=20, **csfont)
   # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none')

def plot_cross_corr(out, PC_score, number, PDO, ONI):

    csfont = {'fontname':'Arial', 'fontsize':12}
    fig, ax = plt.subplots(number//2,2,figsize=(8,5), dpi=200, sharey=True)

    for i in range(number):

        ax.flat[i].xcorr(PDO.values,PC_score[:,i],normed=True,maxlags=70, label=r'$PDO$', color='b')
        ax.flat[i].xcorr(ONI.values,PC_score[:,i],normed=True,maxlags=70, color='r', label=r'$ONI$', alpha=0.8, linestyle='--')
        ax.flat[i].set_title(r'$PC$ '+str(i+1), loc='right', **csfont)
        ax.flat[i].spines['top'].set_visible(False)
        ax.flat[i].spines['right'].set_visible(False)
        ax.flat[i].grid(color='gray', linestyle='--', alpha=0.3)
        ax.flat[i].legend(frameon=False, loc=3, ncol=2)
        ax.flat[i].set_xlim(-6,6)
        ax.flat[i].set_ylim(-0.6,0.6)

    ax[-1,-1].set_xlabel(r'$Lag \, [months]$')
    ax[0, 0].set_ylabel(r'$Pearson \; Correlation$')
    fig.tight_layout()
    fig.savefig(out, dpi=100, transparent=False, edgecolor='none')

def plot_map12(out, matrix, title, labeli, vals, cmap='bwr', mask=False, norm=True):
    
    csfont = {'fontname':'Arial', 'weight':'bold'}
    reader = shpreader.Reader('data/HPA/HP.shp')
    mascara = np.genfromtxt('data/mask.csv', delimiter = ',')
    fig ,ax= plt.subplots(2,6,figsize=(8,4),sharex=True, sharey=True,\
                          dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

    for i in range(12):
        
        mes = matrix[i,:].reshape(24,18)

        if mask == True:
            mes = mes * mascara

        ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
               facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
        ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
          name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
        ax.flat[i].set_title(title[i] ,loc='left',fontsize=12, **csfont)
        gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                      draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
        
        if norm:
            ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=-1*vals)
        else:
            ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals[1],vmin=vals[0])
    for k in ax[1,:]:

        gl = k.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                      draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.ylabels_left = False
        gl.ylocator = mticker.FixedLocator([31,33,35,37,39,41,43,45])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'fontname':'Arial', 'weight': 'bold', 'fontsize': 8}
        
    for z in ax[:,0]:   
        gl2 = z.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                      draw_labels=True,linewidth=1,color='k',alpha=0.01, linestyle='--')             
        gl2.xlabels_top = False
        gl2.ylabels_right = False
        gl2.xlabels_bottom = False
        gl2.ylocator = mticker.FixedLocator([31,33,35,37,39,41,43,45])
        gl2.xformatter = LONGITUDE_FORMATTER
        gl2.yformatter = LATITUDE_FORMATTER
        gl2.ylabel_style = {'fontname':'Arial', 'weight': 'bold', 'fontsize': 8}      
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
    cbar_ax.grid(False)
    [tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
    cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

    if norm:
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
                norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
                    fraction=0.9,pad=0.04,aspect=100,shrink=1,label=labeli, extend='both')
    else:
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
                norm=plt.Normalize(vals[0],vals[1])),ax=cbar_ax,orientation='horizontal',\
                    fraction=0.9,pad=0.04,aspect=100,shrink=1,label=labeli, extend='both')        

    fig.savefig(out, dpi=100, transparent=False, edgecolor='none')

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
"**************************************************************************************************"


"**************************************************************************************************"

# GRACE

#data_grace = Dataset('data/GRCTellus.JPL.200204_202107.GLO.RL06M.MSCNv02CRI.nc')
data_grace = Dataset('data/GRCTellus.JPL.200204_202211.GLO.RL06.1M.MSCNv03CRI.nc')

mask = np.genfromtxt('data/mask.csv', delimiter = ',')
#dates_grace = pd.date_range('2002-04-15','2021-07-17', freq='M')
dates_grace = pd.date_range('2002-04-15','2022-11-17', freq='M')

with open('data/tws_fill.npy', 'rb') as f:
    TWS = np.load(f)

_, lat_grace, lon_grace = read_grace(data_grace)

ix_nan = np.argwhere(np.isnan(_[:,0,0]))[:, 0]
mask_nan = np.in1d(range(_.shape[0]), ix_nan)
mask_nan_m = mask_nan[:208]

# GLDAS


data_gldas = Dataset('data/GLDAS_NOAH_1979-2021.nc')
data_gldas2 = Dataset('data/GLDAS_NOAH_2000-2022.nc')

SM1, SWE1, SM_model, SWE_model, dates_gldas = read_gldas(data_gldas, 279, datetime(1948,1,1))
SM2, SWE2, _, _, dates_gldas2 = read_gldas(data_gldas2, 28, datetime(2000,1,1))

SM = np.concatenate([SM1, SM2[SM1.shape[0]:, :, :]], axis=0)
SWE = np.concatenate([SWE1, SWE2[SWE1.shape[0]:, :, :]], axis=0)

# MODELO

data_model = Dataset('data/GRACE_REC_v03_JPL_ERA5_monthly_ens082.nc')
TWS_model_best = read_model(data_model)
time_model = np.array(data_model.variables['time'][:]) 
dates_model = np.array([datetime(1901,1,1) + timedelta(days = time_model[i]) for i in range(len(time_model))])
dates_model_grace = dates_model[279:]


with open('data/fit_model.npy', 'rb') as f:
    TWS_model = np.load(f)

# with open('data/ensembles.npy', 'rb') as f:
#     TWS_model0 = np.load(f)

# TWS_model = np.nanmean(TWS_model0, axis=0)
# TWS_model_std = np.nanstd(TWS_model0, axis=0)


# fig, axes = plt.subplots(figsize=(10,3), dpi=200)

# axes.fill_between(dates_grace, -15, 15, where = mask_nan, color='gray', alpha=0.3)
# axes.plot(dates_grace, (np.nanmean(TWS, axis=(1,2))), label='Grace fit', c='r')
# axes.plot(dates_grace,(np.nanmean(_, axis=(1,2))), label='Grace', c='k', ls=':')
# axes.set_xlabel('time')
# axes.set_ylabel('TWS anomaly [cm]')
# axes.set_xlim(dates_grace[0],dates_grace[-1])
# axes.set_ylim(-15, 15)
# axes.legend()
# axes.spines['top'].set_visible(False)
# axes.spines['right'].set_visible(False)
# axes.grid(alpha=0.3)
# fig.tight_layout()
# fig.savefig('figures/model_series.jpg', dpi=100, transparent=False, edgecolor='none')


# CHIRPS

#data_chirps = Dataset('data/chirps_hpa_int.nc')
data_chirps = Dataset('data/chirps_monthly_gracegrid.nc')
precip = read_grace(data_chirps, 'precip')[0] / 10
time_chirps = np.array(data_chirps.variables['time'][:]) 
dates_chirps = np.array([datetime(1980,1,1) + timedelta(days = int(i)) for i in time_chirps])
dates_chirps = dates_chirps[:-3]
precip = precip[:-3]

# ONI

data_oni = pd.read_csv('data/ONI.csv', index_col=0)
ONI_model = read_oni(data_oni, ['1979-01','2019-07'])
ONI_grace = read_oni(data_oni, ['2002-04','2022-10'])

# PDO

data_pdo = pd.read_csv('data/PDO.csv',index_col=0, header=1)
PDO_model = read_pdo(data_pdo, ['1979-01','2019-07'])
PDO_grace = read_pdo(data_pdo, ['2002-04','2022-10'])


"*********************************************************************************************************"


# ANOMALIES

TWSa = TWS - np.nanmean(TWS, axis=0)
SMa, SMa_model = SM - np.nanmean(SM, axis=0), SM_model - np.nanmean(SM_model, axis=0)
SWEa, SWEa_model = SWE - np.nanmean(SWE, axis=0), SWE_model - np.nanmean(SWE_model, axis=0)

#SMa, SMa_model = anomalies(SM, 20), anomalies(SM_model, 300),        # anomalias para ecuacion 1
#SWEa, SWEa_model = anomalies(SWE, 20), anomalies(SWE_model, 300)

GWa = TWSa - SMa - SWEa    # Ecuacion 1

GWa_N = (mask*GWa)[:, :8, :]
GWa_C = (mask*GWa)[:, 8:15, :]
GWa_S = (mask*GWa)[:, 15:, :]

#Stadistics

variables = [TWSa, SMa, SWEa, SMa_model, SWEa_model]
nombres = [r"TWSa", r"SMa", r"SWEa", r"SMa model", r"SWEa model"]

[basic_stats(variables[i]*mask, nombres[i]) for i in range(len(variables))]



# spatial_plot('', variables, nombres, dates_grace, mascara=mask)

# #plot corr
# fig, axes = plt.subplots(1,2,figsize=(12,3))
# sns.heatmap(corr_matrix(variables[:4], nombres[:4], dates_grace),ax=axes[0] , annot=True, cmap='Blues')
# sns.heatmap(corr_matrix(variables[4:], nombres[4:], dates_model),ax=axes[1] , annot=True, cmap='Blues')


# MK TEST
#plot_MK_test('figures/mk_test_GWa.jpg' ,GWa, tendencia(GWa)[0], lat_grace, lon_grace)


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
    
df

ft_to_cm = 30.48
s_y = -0.15

df_anom = (df*ft_to_cm - (df*ft_to_cm).mean()) * s_y
#df_anom = s_y * ft_to_cm * df.diff()

df_anom = df_anom.interpolate('linear', limit_direction = 'both')
df_anom = df_anom['1979':'06-2021']

df_anom = df_anom.drop('NE04', axis=1)

mann_kendall = MK_test_df(df_anom)
#mann_kendall = MK_test_df(df_gmean_season + df_gmean_residuales)
    
locacion = pd.read_excel('data/loc_pozos.xls', index_col=0)
locacion = locacion.drop('NE04')

s = mann_kendall['Slope'].values.astype('float16')

# Zoom
locacion_zoom = locacion['NE01':'NE12']
s_zoom = mann_kendall['Slope']['NE01':'NE12'].values.astype('float16')

names = df_anom.columns
names_zoom = names[1:10]
names = np.setdiff1d(names, names_zoom)

average = df_anom.groupby(df_anom.index.month).mean()
average.index = pd.date_range('1/15/2020', '1/15/2021', freq='M').strftime('%B')
# average.plot(figsize=(14,8),subplots=True, layout=(6,4), title= 'CICLO ANUAL')

df_trend, df_season, df_residual = deseasonal_df(df_anom)
df_des = df_trend + df_residual

av_north = (average[['NE12', 'NE14', 'NE15', 'NE17', 'NE19']]).mean(axis=1)
av_central = (average[['NE01', 'NE02', 'NE03', 'NE05', 'NE06', 'NE08', 'NE09', 'NE10', 'KS02']]).mean(axis=1)
av_south = (average[['TX08', 'TX08']]).mean(axis=1)


# REMOVE LINEAR TREND

GWa_detrend = detrend(GWa, axis=0)

#precip_a = anomalies(precip, 276)[:-12]
precip_a = precip - np.nanmean(precip, axis=0)
precip_detrend = detrend(precip_a, axis=0)


# ANNUAL CYCLE

#GWa_annual, dates_annual = annual_cycle(estandarizar(GWa_detrend), dates_grace)
GWa_annual, dates_annual = annual_cycle(GWa_detrend, dates_grace)
# plot_map12('figures/anual_cycle_GWa.jpg', GWa_annual , dates_annual, r'$GW \, anomalies [cm]$', 8, cmap='seismic_r', mask=True)
# GWa_annual_mean = [np.nanmean(GWa_annual[i,:]) for i in range(12)]

#precip_annual, dates_annual = annual_cycle(estandarizar(precip_detrend), pd.date_range('1981-01-15','2021-07-17', freq='M'))
precip_annual, dates_annual = annual_cycle(precip_detrend, pd.date_range('1981-01-15','2022-10-17', freq='M'))
# plot_map12('figures/anual_cycle_precip.jpg', precip_annual , dates_annual, r'$Rain \, anomalies [cm]$', [-8, 8], 'PuOr', norm=False)
# precip_annual_mean = [np.nanmean(precip_annual[i,:]) for i in range(12)]

#variables = [GWa_annual*2, precip_annual[-231:, :]]
#nombres = [r"$GWa x2$", r"$Rain$"]

variables = [GWa*mask, GWa_N, GWa_C, GWa_S]
nombres_GRACE = [r"GWa", r"GWa northern HPA", r"GWa central HPA", r"GWa southern HPA"]

#[basic_stats(variables[i], nombres[i]) for i in range(len(variables))]


GWa_annual_N, _ = annual_cycle(GWa_N, dates_grace)
GWa_annual_C, _ = annual_cycle(GWa_C, dates_grace)
GWa_annual_S, _ = annual_cycle(GWa_S, dates_grace)

#variables = [GWa_annual*2, precip_annual[-231:, :]]
#nombres = [r"$GWa x2$", r"$Rain$"]
# variables = [GWa_annual, GWa_annual_N, GWa_annual_C, GWa_annual_S]
# spatial_plot('figures/anual_cycle_serie.jpg', variables, nombres_GRACE, dates_annual, mascara=None)

# REMOVE SEASONAL COMPONENT

GWa_residual = deseasonal(GWa_detrend)[0] + deseasonal(GWa_detrend)[2] # sumar tendencia y residual
GWa_residual = np.reshape(GWa_residual, (247, 432))

precip_residual = deseasonal(precip_detrend)[0] + deseasonal(precip_detrend)[2] # sumar tendencia y residual
precip_residual = np.reshape(precip_residual, (501, 432))

# MODEL

SWEa_model_residual = deseasonal(SWEa_model)[0] + deseasonal(SWEa_model)[2] # sumar tendencia y residual
SWEa_model_residual = np.reshape(SWEa_model_residual, (487, 24 * 18))

SMa_model_residual = deseasonal(SMa_model)[0] + deseasonal(SMa_model)[2] # sumar tendencia y residual
SMa_model_residual = np.reshape(SMa_model_residual, (487, 24 * 18))

GWa_model = TWS_model - SMa_model_residual - SWEa_model_residual

GWa_detrend_model = detrend(GWa_model, axis=0)

colors = ['firebrick', 'royalblue', 'darkturquoise', 'lime']

# Probando suavizado
GWa_residual_m = moving_mean(GWa_residual, 3)[2:]
GWa_detrend_model_m = moving_mean(GWa_detrend_model, 3)[2:]







#%%

# COMPOSITES    SSSS!!!


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
                (df.index <= events['end_date'].loc[i].to_timestamp())
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
    threshold = 0.5

    df = pd.DataFrame(array, index=index_dates)

    el_nino_events, la_nina_events, neutral_events = define_ENSO_phase(ENSO_index, threshold, duration_months)

    df_nino = monthly_composites(df, el_nino_events, months, lag, type)
    df_nina = monthly_composites(df, la_nina_events, months, lag, type)
    df_neutral = monthly_composites(df, neutral_events, months, lag, type)

    if type == "total":
        composites = np.array([df_nino, df_nina, df_neutral])
    elif type == "season":
        composites = np.array(pd.concat([df_nino, df_nina, df_neutral], axis=0))
    else:
        print("check type")

    return composites.T

lag = 3

labels_composites = ['El Niño', 'La Niña', 'Neutral']
total_composites = lag_composites(GWa_residual_m, dates_grace[2:], ONI_grace, lag, "total")
total_composites_m = lag_composites(GWa_detrend_model_m, dates_model[2:], ONI_model, lag, "total")

plot_EOFs(f'figures/total_composites_grace.jpg', total_composites, lat_grace, lon_grace, 3, labels_composites,r'anom',\
        5, grid=(1,3), fig_size=(6,3), points=None)

plot_EOFs(f'figures/total_composites_model.jpg', total_composites_m, lat_grace, lon_grace, 3, labels_composites,r'anom',\
        5, grid=(1,3), fig_size=(6,3), points=None)

seasonal_labels = ["DJF", "MAM", "JJA", "SON"] + [""] * 8
seasonal_composites = lag_composites(GWa_residual_m, dates_grace[2:], ONI_grace, lag, "season")
season_composites_m = lag_composites(GWa_detrend_model_m, dates_model[2:], ONI_model, lag, "season")

plot_EOFs(f'figures/seasonal_composites_grace.jpg', seasonal_composites, lat_grace, lon_grace, 12, seasonal_labels,r'anom',\
        5, grid=(3,4), fig_size=(6,6), points=None)

plot_EOFs(f'figures/seasonal_composites_model.jpg', season_composites_m, lat_grace, lon_grace, 12, seasonal_labels,r'anom',\
        5, grid=(3,4), fig_size=(6,6), points=None)


#%%

precip_residual_m = moving_mean(precip_residual.reshape(501, 24*18), 3)[3:]


lag = 0

total_composites_p = lag_composites(precip_residual_m, pd.date_range('1981-04-15','2022-10-17', freq='M'), read_oni(data_oni, ['1981-04','2022-10']), lag, "total")

plot_EOFs(f'figures/total_composites_precip.jpg', total_composites_p, lat_grace, lon_grace, 3, labels_composites,r'anom',\
        1, grid=(1,3), fig_size=(6,3), points=None)

seasonal_composites_p = lag_composites(precip_residual_m, pd.date_range('1981-04-15','2022-10-17', freq='M'), read_oni(data_oni, ['1981-04','2022-10']), lag, "season")

plot_EOFs(f'figures/seasonal_composites_precip.jpg', seasonal_composites_p, lat_grace, lon_grace, 12, seasonal_labels,r'anom',\
        1, grid=(3,4), fig_size=(6,6), points=None)


# Precip_N = (mask*precip_residual_m.reshape(498, 24, 18))[:, :8, :]
# Precip_C = (mask*precip_residual_m.reshape(498, 24, 18))[:, 8:15, :]
# Precip_S = (mask*precip_residual_m.reshape(498, 24, 18))[:, 15:, :]


# plt.figure(figsize=(10, 12))
# plt.plot(pd.date_range('1981-04-15','2022-10-17', freq='M'), np.nanmean(Precip_N, axis=(1,2)))

#plt.plot(pd.date_range('1981-04-15','2022-10-17', freq='M'), np.nanmean(Precip_C, axis=(1,2)))
#plt.plot(pd.date_range('1981-04-15','2022-10-17', freq='M'), np.nanmean(Precip_S, axis=(1,2)))


#%%
# Analisis sin filtrado

#GWa_tape = np.array([GWa_residual[:, i] * np.hanning(GWa_residual.shape[0]) for i in range(GWa_residual.shape[1])])
#GWa_tape = np.multiply(GWa_residual.T, np.hanning(GWa_residual.shape[0]))

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

            return loadings, coef, val
        
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

# array = GWa_residual_m

matrix_used = 'corr'
# matrix = np.corrcoef(array.T)
# # print('Correlations shape ' + str(comatrixrr.shape))
# val, vec = linalg.eig(matrix) 
# # PC are obtained from dot product from original matrix and eigenvectors
# PC = ((array - np.nanmean(array, axis=0)) / np.std(array, axis=0)) @ vec

# PCscore = PC @ np.diag(np.sqrt(val[:]))
# coef = PCscore[:,:] @ np.linalg.inv(np.diag(val[:]))

# #  coef = (coef - np.nanmean(coef, axis=0) / np.var(coef, axis=0))

# loadings = array.T @ coef / len(array)
# #  loadings = (PCscore.T @ array) / len(array)

# # Sum of squared loading values give variance explained by PC (eigenvalue)
# print('Variance explained by loading 1' , np.sum(loadings[:,0]**2))
# print('Variance explained by loading 2' , np.sum(loadings[:,1]**2))

# print(loadings.shape)
# plt.plot(coef[:,0])

print('TWS GRACE residual')
vec, PC, val = PCA(GWa_residual_m, matrix_used=matrix_used)
loadings, PCscore, val = PCA(GWa_residual_m, normalized=True, matrix_used=matrix_used)

plot_variance_explained('figures/PC_variance_GWa.jpg', val)
#plot_EOFs('figures/EOF_GWa.jpg', loadings, lat_grace, lon_grace, 4, r'EOF',r'$cm / std$',7, fig_size=(8,3),points=None)
#plot_scores('figures/PC_GWa.jpg', PCscore, val, 4, colors , r'$PC$ ', ONI_grace, PDO_grace, dates_grace)

## .. modelo

print('TWS modelo')
vec_model, PC_model, val_model = PCA(GWa_detrend_model_m, matrix_used=matrix_used)
loadings_model, PCscore_model, val_model = PCA(GWa_detrend_model_m, normalized=True, matrix_used=matrix_used)

#%%
#plot_variance_explained('figures/PC_variance_GWa_model.jpg', val_model)
#plot_EOFs('figures/EOF_GWa_model.jpg', loadings_model, lat_grace, lon_grace, 4, r'EOF',r'$cm / std$',7, fig_size=(8,3))
#plot_scores('figures/PC_GWa_model.jpg', PCscore_model, val_model, 4, colors , r'$PC$ ', ONI_model.loc[:'2019-07'], PDO_model.loc[:'2019-07'], dates_model)

# GRACE
#plot_cross_corr('figures/corr_PCix.jpg', PCscore, 4, PDO_grace, ONI_grace)

#modelo
#plot_cross_corr('figures/corr_PCix_model.jpg', PCscore_model, 4, PDO_model.loc[:'2019-07'], ONI_model.loc[:'2019-07'])


# Get rotated matrix from 3 first loadings)
matrix_rot = ortho_rotation(loadings[:,:4])
matrix_rot_model = ortho_rotation(loadings_model[:,:4])

# Rotated loadings and components
# Components not orthonogals
rotated_load = loadings[:,:4] @ matrix_rot
rotated_load_model = loadings_model[:,:4] @ matrix_rot_model

# New component data matrix . pseudo.inverser of transponse of rotaed loadings
rotated_PC = GWa_residual_m @ np.linalg.pinv(rotated_load).T
rotated_PC_model = GWa_detrend_model_m @ np.linalg.pinv(rotated_load_model).T

variance = np.real(np.sum(loadings**2, axis=0))
variance_model = np.real(np.sum(loadings_model**2, axis=0))
print('variance grace: ', np.sum(np.round(variance,2)))
print([f"EOF {i+1} : {np.round(100* variance[i] / np.sum(variance), 1)}" for i in range(4)])
print('variance model: ', np.sum(np.round(variance_model,2)))
print([f"EOF {i+1} : {np.round(100 * variance_model[i] / np.sum(variance_model),1)}" for i in range(4)])

print(" rotated .. ")
variance_rotated = np.real(np.sum(rotated_load**2, axis=0))
variance_model_rotated = np.real(np.sum(rotated_load_model**2, axis=0))
print('variance grace: ', np.sum(np.round(variance_rotated,2)))
print([f"EOF {i+1} : {np.round(100* variance_rotated[i] / np.sum(variance_rotated),1)}" for i in range(4)])
print('variance model: ', np.sum(np.round(variance_model_rotated,2)))
print([f"EOF {i+1} : {np.round(100 * variance_model_rotated[i] / np.sum(variance_model_rotated),1)}" for i in range(4)])


# Recovered data is not equal
rotated_GWa_trunc = rotated_PC @ rotated_load.T
rotated_model_trunc = rotated_PC_model @ rotated_load_model.T

PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T

rotated_PC_selected = np.array([rotated_PC[:,0], rotated_PC[:,2], rotated_PC[:,3]]).T
rotated_PC_model_selected = np.array([rotated_PC_model[:,0], rotated_PC_model[:,2], rotated_PC_model[:,3]]).T

GWa_trunc = (PCscore[:, [0]]) @ loadings[:, [0]].T
GWa_trunc_model = (PCscore_model[:, [0]]) @ loadings_model[:, [0]].T

# para unrotated cov
#componente 3 (2) es el mas poderoso
# 1 y 5 juntos (0 y 4) hacen
# 4 y 5 (3 y 4) tambien muestran algo raro ahi
#

GWa_rotated_trunc = rotated_PC[:,[3]] @ rotated_load[:,[3]].T
GWa_rotated_trunc_model = rotated_PC_model[:,[3]] @ rotated_load_model[:,[3]].T

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

#%%


# EOFS MODEL GRAPH

# matrix = loadings_model
# matrix2 = PCscore_model
# eigenval = val_model

matrix = rotated_load_model
matrix2 = rotated_PC_model
eigenval = variance_model_rotated

dates = dates_model[2:]

oni = ONI_model.loc[:'2019-07'][2:]
pdo = PDO_model.loc[:'2019-07'][2:]

csfont = {'fontname':'Arial', 'fontsize' : 12}
lat = lat_grace
lon = lon_grace
vals = 8
labeli = "cm/std"
colors = colors
label = "PC"
title = ["PC1", "PC2", "PC3", "PC4"]

#csfont = {'fontname':'Arial', 'weight':'bold'}
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


fig.savefig(f'figures/EOF_model_r_{matrix_used}.pdf', dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")
#%%

# EOFS GRACE GRAPH

# matrix = loadings
# matrix2 = PCscore
# eigenval = val

matrix = rotated_load
matrix2 = rotated_PC
eigenval = variance_rotated

dates = dates_grace[2:]
oni = ONI_grace[2:]
pdo = PDO_grace[2:]

csfont = {'fontname':'Arial', 'fontsize' : 12}
lat = lat_grace
lon = lon_grace
vals = 8
labeli = "cm/std"
colors = colors
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
fig.savefig(f'figures/EOF_grace_r_{matrix_used}.pdf', dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")


#%%

# SPECTRAL GRAPH OLD

PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T


def plot_FFT_2c(out, PC_score, PCscore_model, oni,oni_model, pdo, pdo_model, labels, numero_PCs=2, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(numero_PCs+2,2, sharex=True, figsize=(6,6), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(numero_PCs):
        
        periods = func(PC_score[:,i])[0]
        powers = estandarizar(func(PC_score[:,i])[1])

 #       ax[i].plot(periods[1:], powers[1:], label=r'$PC$ '+str(i+1), c='darkblue')
        ax[i, 0].plot(periods[1:], powers[1:], label=labels[i], c='k')
        ax[i, 0].plot(periods[1:], red_noise_spectrum(PC_score[:, i], 0.05), c='firebrick', ls='--')

        periods_model = func(PCscore_model[:,i])[0]
        powers_model = estandarizar(func(PCscore_model[:,i])[1])

 #       ax[i].plot(periods[1:], powers[1:], label=r'$PC$ '+str(i+1), c='darkblue')
        ax[i, 1].plot(periods_model[1:], powers_model[1:], label=labels[i], c='k')
        ax[i, 1].plot(periods_model[1:], red_noise_spectrum(PCscore_model[:, i], 0.05), c='firebrick', ls='--')

    
    if func == fourier_filter:

        ax[numero_PCs, 0].plot(func(oni[:])[0][1:], estandarizar(func(oni, freq_min, freq_max)[1])[1:], label=labeli[0], c='k')
        ax[numero_PCs+1, 0].plot(func(pdo[:])[0][1:], estandarizar(func(pdo, freq_min, freq_max)[1])[1:], label=labeli[1], c='k')
       # ax[numero_PCs, 0].plot(periods[1:], red_noise_spectrum(oni, 0.05), c='firebrick', ls='--')
      #  ax[numero_PCs+1, 0].plot(periods[1:], red_noise_spectrum(pdo, 0.05), c='firebrick', ls='--')

        ax[numero_PCs, 1].plot(func(oni_model[:])[0][1:], estandarizar(func(oni_model, freq_min, freq_max)[1])[1:], label=labeli[0], c='k')
        ax[numero_PCs+1, 1].plot(func(pdo_model[:])[0][1:], estandarizar(func(pdo_model, freq_min, freq_max)[1])[1:], label=labeli[1], c='k')
        ax[numero_PCs, 1].plot(periods_model[1:], red_noise_spectrum(oni_model, 0.05), c='firebrick', ls='--')
        ax[numero_PCs+1, 1].plot(periods_model[1:], red_noise_spectrum(pdo_model, 0.05), c='firebrick', ls='--')

    else:
        ax[numero_PCs, 0].plot(func(oni)[0][1:], estandarizar(func(oni)[1])[1:], label=labeli[0], c='k')
        ax[numero_PCs+1, 0].plot(func(pdo)[0][1:], estandarizar(func(pdo)[1])[1:], label=labeli[1], c='k')
        ax[numero_PCs, 0].plot(periods[1:], red_noise_spectrum(oni, 0.05), c='firebrick', ls='--')
        ax[numero_PCs+1, 0].plot(periods[1:], red_noise_spectrum(pdo, 0.05), c='firebrick', ls='--')
    
        ax[numero_PCs, 1].plot(func(oni_model[:])[0][1:], estandarizar(func(oni_model)[1])[1:], label=labeli[0], c='k')
        ax[numero_PCs+1, 1].plot(func(pdo_model[:])[0][1:], estandarizar(func(pdo_model)[1])[1:], label=labeli[1], c='k')
        ax[numero_PCs, 1].plot(periods_model[1:], red_noise_spectrum(oni_model, 0.05), c='firebrick', ls='--')
        ax[numero_PCs+1, 1].plot(periods_model[1:], red_noise_spectrum(pdo_model, 0.05), c='firebrick', ls='--')

    for axe in ax.flatten():
        
        axe.axvline(12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=9, frameon=False) 
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
     #   ax[i].set_xscale('log')
        axe.set_ylim(0, 10)
        axe.set_xlim(0, 120)

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Period [Months]',labelpad=20, **csfont)
   # axe.get_xaxis().set_visible(False)
    axe.set_ylabel('Stand power', labelpad=20, **csfont)

   # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=False, edgecolor='none', bbox_inches="tight")

plot_FFT_2c('figures/FT_GWa_GWa_model.pdf', PCscore_selected, PCscore_model_selected, ONI_grace, ONI_model.loc[:'2019-07'],\
            PDO_grace, PDO_model.loc[:'2019-07'], ["PC1", "PC3", "PC4"], 3, FFT)


#%%

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

# SPECTRAL GRAPH NEW

chunk_size = 247
n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')
confidence = .95
scale = 'density'
win = 'boxcar'


# PCscore_selected_m = moving_mean(PCscore_selected, 3)
# PCscore_model_selected_m = moving_mean(PCscore_model_selected, 3)

# selected = PCscore[:,[0,2,3]]
# selected_model = PCscore_model[:,[0,2,3]]

selected = rotated_PC[:,[0,2,3]]
selected_model = rotated_PC_model[:,[0,2,3]]

def plot_FFT_2c(out, PC_score, PCscore_model, oni,oni_model, pdo, pdo_model, labels, numero_PCs=2, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(numero_PCs+1,2, sharex=True, figsize=(6,5), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(numero_PCs):
        

        serie = PC_score[:,i]
        serie_model = PCscore_model[:, i]

        freqs, powers = signal.csd(serie, serie, window=win,noverlap = chunk_size/4, nperseg = chunk_size, \
                scaling = scale, detrend = False)
        
        freqs_model, powers_model = signal.csd(serie_model, serie_model, window=win,noverlap = chunk_size/4, nperseg = chunk_size, \
                scaling = scale, detrend = False)
        
        freqs_ONI, powers_ONI = signal.csd(oni, oni, window=win,noverlap = chunk_size/4, nperseg = chunk_size, \
                scaling = scale, detrend = False)

        freqs_ONI_model, powers_ONI_model = signal.csd(oni_model, oni_model, window=win,noverlap = chunk_size/4, nperseg = chunk_size, \
                scaling = scale, detrend = False)

        ax[i, 0].plot(1 / freqs, powers, label=labels[i], c='k')
        ax[i, 0].plot(1 / freqs, red_noise_spectrum(serie, freqs, chunk_size, confidence, scale), c='firebrick', ls='--')
        ax[i, 1].plot(1 / freqs_model, powers_model, label=labels[i], c='k')
        ax[i, 1].plot(1 / freqs_model, red_noise_spectrum(serie_model, freqs_model, chunk_size, confidence, scale), c='firebrick', ls='--')

        print(np.max(powers), np.max(powers_model))    
    ax[numero_PCs, 0].plot(1 / freqs_ONI, powers_ONI, label=labeli[0], c='k')
    ax[numero_PCs, 0].plot(1 / freqs_ONI, red_noise_spectrum(oni, freqs_ONI, chunk_size, confidence, scale), c='firebrick', ls='--')

    ax[numero_PCs, 1].plot(1 / freqs_ONI_model, powers_ONI_model, label=labeli[0], c='k')
    ax[numero_PCs, 1].plot(1 / freqs_ONI_model, red_noise_spectrum(oni_model, freqs_ONI_model, chunk_size, confidence, scale), c='firebrick', ls='--')

    for axe in ax.flatten():
        
        axe.axvline(12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=2, frameon=False) 
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        axe.set_ylim(0, 50)
        axe.set_xlim(0, 120)
        axe.set_xscale('log')

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Period [Months]',labelpad=20, **csfont)
    axe.set_ylabel('Power (Fraction of variance)', labelpad=25, **csfont)

   # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=300, transparent=False, edgecolor='none', bbox_inches="tight")


plot_FFT_2c(f'figures/FT_r_{matrix_used}.pdf', selected, selected_model, ONI_grace[2:], ONI_model.loc[:'2019-07'][2:],\
            PDO_grace, PDO_model.loc[:'2019-07'], ["PC1", "PC3", "PC4"], 3, FFT)


# COHERENCE


confidence = .99

# PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
# PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T


def plot_coherence_2c(out, PC_score, PCscore_model, oni,oni_model, pdo, pdo_model, labels, numero_PCs=2, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(numero_PCs,2, sharex=True, figsize=(6,5), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(numero_PCs):
        

        serie = PC_score[:,i]
        serie_model = PCscore_model[:, i]

        chunk_size = 124
        n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')
        dof = 2*n_chunks
        fval = stats.f.ppf(0.99, 2, dof - 2)
        r2_cutoff = fval / (n_chunks - 1. + fval)

        chunk_size_m = 244
        n_chunks_m = np.round((len(serie_model) * 2 / chunk_size_m) - 1).astype('int')
        dof_m = 2*n_chunks_m
        fval_m = stats.f.ppf(0.99, 2, dof_m - 2)
        r2_cutoff_m = fval_m / (n_chunks_m - 1. + fval_m)

        freqs, Cxy = signal.coherence(serie, oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                       detrend = False)
        
        
        freqs_model, Cxy_model = signal.coherence(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size_m/2, nperseg = chunk_size_m, \
                       detrend = False)
        
        ix_enso = np.intersect1d(np.where(freqs <= 1/24)[0], np.where(freqs >= 1/(8*12))[0])
        ix_enso_m = np.intersect1d(np.where(freqs_model <= 1/24)[0], np.where(freqs_model >= 1/(8*12))[0])

        j = np.argsort(Cxy[ix_enso])[-1]
        j_m = np.argsort(Cxy_model[ix_enso_m])[-1]

       # print(Cxy[ix_enso], j)
        #print(Cxy_model[ix_enso_m], j_m)

        ax[i, 0].plot(1 / freqs, Cxy, label=labels[i], c='k')
        ax[i, 0].hlines(r2_cutoff, 0, 180, linewidth = 2, ls='--', color='r')
        ax[i, 0].plot(1 / freqs[ix_enso][j], Cxy[ix_enso][j],'or',linewidth = 2,markersize = 3)

        ax[i, 1].plot(1 / freqs_model, Cxy_model, label=labels[i], c='k')
        ax[i, 1].hlines(r2_cutoff_m, 0, 180, linewidth = 2, ls='--', color='r')
        ax[i, 1].plot(1 / freqs_model[ix_enso_m][j_m], Cxy_model[ix_enso_m][j_m],'or',linewidth = 2,markersize = 3)

    for axe in ax.flatten():
        
        axe.axvline(12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=9, frameon=False) 
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        #axe.set_ylim(0, 0.01)
        axe.set_xlim(12, 8*12)

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Period [Months]',labelpad=20, **csfont)
    axe.set_ylabel(r'$R^{2}$', labelpad=25, **csfont)

   # plt.xlabel(r'$Period [months]$', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")

plot_coherence_2c(f'figures/COHERENCE_r_{matrix_used}.pdf', selected, selected_model, ONI_grace, ONI_model.loc[:'2019-07'],\
            PDO_grace, PDO_model.loc[:'2019-07'], ["PC1", "PC3", "PC4"], 3, FFT)


# PHASE 

# PCscore_selected = np.array([PCscore[:,0], PCscore[:,2], PCscore[:,3]]).T
# PCscore_model_selected = np.array([PCscore_model[:,0], PCscore_model[:,2], PCscore_model[:,3]]).T


def plot_phase_2c(out, PC_score, PCscore_model, oni,oni_model, pdo, pdo_model, labels, numero_PCs=2, func=FFT, freq_min=None, freq_max=None):
    
    fig, ax = plt.subplots(numero_PCs,2, sharex=True, figsize=(6,5), dpi=300)

    csfont = {'fontname':'Arial', 'fontsize':12}
    labeli = ["ONI", "PDO"]

    for i in range(numero_PCs):
        

        serie = PC_score[:,i]
        serie_model = PCscore_model[:, i]

        chunk_size = 124
        n_chunks = np.round((len(serie) * 2 / chunk_size) - 1).astype('int')
        dof = 2*n_chunks
        fval = stats.f.ppf(0.99, 2, dof - 2)
        r2_cutoff = fval / (n_chunks - 1. + fval)

        chunk_size_m = 244
        n_chunks_m = np.round((len(serie_model) * 2 / chunk_size_m) - 1).astype('int')
        dof_m = 2*n_chunks_m

        freqs, Cxy = signal.coherence(serie, oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                       detrend = False)
        
        freqs, Pxy = signal.csd(serie, oni.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                 scaling = 'density', detrend = False)
        
        P = -np.angle(Pxy, deg = True)
        
        freqs_model, Cxy_model = signal.coherence(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size_m/2, nperseg = chunk_size_m, \
                       detrend = False)
        
        freqs_model, Pxy_model = signal.csd(serie_model, oni_model.values.astype('complex'), window = 'boxcar', noverlap = chunk_size/2, nperseg = chunk_size, \
                 scaling = 'density', detrend = False)

        P_model = -np.angle(Pxy_model, deg = True)

        ix_enso = np.intersect1d(np.where(freqs <= 1/24)[0], np.where(freqs >= 1/(8*12))[0])
        ix_enso_m = np.intersect1d(np.where(freqs_model <= 1/24)[0], np.where(freqs_model >= 1/(8*12))[0])

        j = np.argsort(Cxy[ix_enso])[-1]
        j_m = np.argsort(Cxy_model[ix_enso_m])[-1]


        ax[i, 0].plot(1 / freqs, P, label=labels[i], c='k')
        ax[i, 0].plot(1 / freqs[ix_enso][j], P[ix_enso][j],'or',linewidth = 2,markersize = 3)

        ax[i, 1].plot(1 / freqs_model, P_model, label=labels[i], c='k')
        ax[i, 1].plot(1 / freqs_model[ix_enso_m][j_m], P_model[ix_enso_m][j_m],'or',linewidth = 2,markersize = 3)

        print('-------------------------------------')
        print('            PERIOD ........... PHASE')
        print('   GRACE:     ' + str(np.round(1./freqs[ix_enso][j],0)) + ' months.        ' + str(np.round(P[ix_enso][j])) + ' deg.')
        print('   HYG MODEL:     ' + str(np.round(1./freqs_model[ix_enso_m][j_m],0)) + ' months.        ' + str(np.round(P_model[ix_enso_m][j])) + ' deg.')

    for axe in ax.flatten():
        
        axe.axvline(12*2,linestyle='--', color='darkblue', alpha=0.8)
        axe.axvline(12*7, linestyle='dashed', color='darkblue', alpha=0.8)
        axe.legend(loc=9, frameon=False) 
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.grid(color='gray', linestyle='--', alpha=0.3)
        axe.set_ylim(-185, 185)
        axe.set_xlim(12, 8*12)
        axe.axhline(y=0,color='gray')

    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Period [Months]',labelpad=20, **csfont)
    axe.set_ylabel('Phase difference (degree)', labelpad=30, **csfont)

    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

    axe.text(0.06, 0.92, "a)", transform=fig.transFigure, **csfont)
    axe.text(0.52, 0.92, "b)", transform=fig.transFigure, **csfont)

    fig.tight_layout()
    fig.savefig(out, dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")

plot_phase_2c(f'figures/PHASE_r_{matrix_used}.pdf', selected, selected_model, ONI_grace, ONI_model.loc[:'2019-07'],\
            PDO_grace, PDO_model.loc[:'2019-07'], ["PC1", "PC3", "PC4"], 3, FFT)


#%%


# LAG CORRELATION BETWEEN ONI AND PC SPECTRUMS

fig, ax = plt.subplots(2,2, figsize=(7,5), dpi=200)

#ax.flat[0].xcorr(PDO_grace.values,PCscore[:,0],normed=True,maxlags=70, label='PDO', color='steelblue')
ax.flat[0].xcorr(ONI_grace.values[2:],selected[:,0],normed=True,maxlags=70, color='r', label='ONI', alpha=0.8, linestyle='--')
#ax.flat[1].xcorr(PDO_grace.values,rotated_PC[:,2],normed=True,maxlags=70, label='PDO', color='steelblue')
ax.flat[1].xcorr(ONI_grace.values[2:],selected[:,1],normed=True,maxlags=70, color='r', label='ONI', alpha=0.8, linestyle='--')
#ax.flat[2].xcorr(PDO_model.loc[:'2019-07'].values,PCscore_model[:,0],normed=True,maxlags=70, label='PDO', color='steelblue')
ax.flat[2].xcorr(ONI_model.loc[:'2019-07'][2:].values,selected_model[:,0],normed=True,maxlags=70, color='r', label='ONI', alpha=0.8, linestyle='--')
#ax.flat[3].xcorr(PDO_model.loc[:'2019-07'].values,rotated_PC_model[:,2],normed=True,maxlags=70, label='PDO', color='steelblue')
ax.flat[3].xcorr(ONI_model.loc[:'2019-07'].values[2:],selected_model[:,1],normed=True,maxlags=70, color='r', label='ONI', alpha=0.8, linestyle='--')

for axe in ax.flatten():
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.grid(color='gray', linestyle='--', alpha=0.3)
    axe.legend(frameon=False, loc=2, ncol=2)
    axe.set_xlim(-36,36)
    axe.set_ylim(-0.6,0.6)
    axe.axvline(0,-6,6, lw=5, color='gray', alpha=0.5, zorder=0)

fig.tight_layout()

ax.flat[0].text(0.01, 0.97, "a)", transform=fig.transFigure, **csfont)
ax.flat[0].text(0.5, 0.97, "b)", transform=fig.transFigure, **csfont)
ax.flat[0].text(0.01, 0.49, "c)", transform=fig.transFigure, **csfont)
ax.flat[0].text(0.5, 0.49, "d)", transform=fig.transFigure, **csfont)

fig.savefig(f"figures/LAGcorr_r_{matrix_used}.pdf", dpi=200, transparent=False, edgecolor='none', bbox_inches="tight")

#%%

# CORRELACIONES 


r_precip_grace = np.array([corrmap(precip_residual_m[:len(selected[:,k])], selected[:,k])[0] for k in range(3)])
r_precip_model = np.array([corrmap(precip_residual_m[:len(selected_model[28:,k])], selected_model[28:,k])[0] for k in range(3)])


r_precip = np.concatenate((r_precip_grace, r_precip_model))

labels = ['PC1', 'PC3', 'PC4', 'PC1', 'PC3', 'PC4']
plot_EOFs(f'figures/CorrP_PC_r_{matrix_used}.jpg', r_precip.T, lat_grace, lon_grace, 6, labels,r'Pearson correlation',\
        .5, grid=(2,3), fig_size=(6,6), points=None)

GW_trunc = GWa_rotated_trunc
GW_trunc_model = GWa_rotated_trunc_model

r_GWa_ONI = np.array([corrmap(GW_trunc, np.roll(ONI_grace[2:], i*2))[0] for i in range(-4,4)])
labels = ['Lag -8', 'Lag -6', 'Lag-4', 'Lag -2', '0', 'Lag 2', 'Lag 4', 'Lag 6']
plot_EOFs(f'figures/LAGcorrONI_GW_grace_r_{matrix_used}.jpg', r_GWa_ONI.T, lat_grace, lon_grace, 8, labels,r'Pearson correlation',\
        .6, grid=(2,4), fig_size=(6,4), points=None)


r_GWa_ONI = np.array([corrmap(GW_trunc_model, np.roll(ONI_model[2:], i*2))[0] for i in range(-4,4)])
labels = ['Lag -8', 'Lag -6', 'Lag-4', 'Lag -2', '0', 'Lag 2', 'Lag 4', 'Lag 6']
plot_EOFs(f'figures/LAGcorrONI_GW_model_r_{matrix_used}.jpg', r_GWa_ONI.T, lat_grace, lon_grace, 8, labels,r'Pearson correlation',\
        .6, grid=(2,4), fig_size=(6,4), points=None)




#%%


# GRANGER causality


max_lags =  4 # maximum number of lags
significance = 0.1

p_value_ENSO_GWa = np.zeros((432, max_lags))
p_value_GWa_ENSO = np.zeros((432, max_lags))

for i in range(GWa_trunc.shape[1]):
    granger_ENSO_GWa = grangercausalitytests(pd.DataFrame(np.array([GW_trunc[:,i], ONI_grace[2:]]).T), maxlag=max_lags)
    granger_GWa_ENSO = grangercausalitytests(pd.DataFrame(np.array([ONI_grace[2:], GW_trunc[:,i]]).T), maxlag=max_lags)
    
    for lag in range(1, max_lags + 1):
        p_value_ENSO_GWa[i, lag - 1] = granger_ENSO_GWa[lag][0]['ssr_chi2test'][1]
        p_value_GWa_ENSO[i, lag - 1] = granger_GWa_ENSO[lag][0]['ssr_chi2test'][1]

p_value = np.concatenate([p_value_ENSO_GWa.T, p_value_GWa_ENSO.T])
p_value[p_value > significance] = 0.
p_value[p_value != 0] = 1.

labels = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4']*2
plot_EOFs(f'figures/GRANGER_GW_grace_r_{matrix_used}.jpg', p_value.T, lat_grace, lon_grace, 8, labels,'Granger causality',\
        1, grid=(2,4), fig_size=(6,4), points=None)


max_lags =  4 # maximum number of lags
significance = 0.1

p_value_ENSO_GWa = np.zeros((432, max_lags))
p_value_GWa_ENSO = np.zeros((432, max_lags))

for i in range(GWa_trunc.shape[1]):
    granger_ENSO_GWa = grangercausalitytests(pd.DataFrame(np.array([GW_trunc_model[:,i], ONI_model[2:]]).T), maxlag=max_lags)
    granger_GWa_ENSO = grangercausalitytests(pd.DataFrame(np.array([ONI_model[2:], GW_trunc_model[:,i]]).T), maxlag=max_lags)
    
    for lag in range(1, max_lags + 1):
        p_value_ENSO_GWa[i, lag - 1] = granger_ENSO_GWa[lag][0]['ssr_chi2test'][1]
        p_value_GWa_ENSO[i, lag - 1] = granger_GWa_ENSO[lag][0]['ssr_chi2test'][1]

p_value = np.concatenate([p_value_ENSO_GWa.T, p_value_GWa_ENSO.T])
p_value[p_value > significance] = 0.
p_value[p_value != 0] = 1.

labels = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4']*2
plot_EOFs(f'figures/GRANGER_GW_model_r_{matrix_used}.jpg', p_value.T, lat_grace, lon_grace, 8, labels,'Granger causality',\
        1, grid=(2,4), fig_size=(6,4), points=None)


# %%


# trend figure}

import matplotlib.gridspec as gridspec

csfont = {'fontname':'Arial', 'fontsize' : 12} 

array_in = GWa
longitudes = lon_grace
latitudes = lat_grace
trends = tendencia(GWa)[0]


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

ax1.scatter(longitudes_wells, latitudes_wells, np.abs(s)*10, c= s, norm=plt.Normalize(-8,8),cmap='RdYlGn', zorder=5)
ax1.set_extent([-105, -97, 32, 44])

#ax1.set_title('a)',loc='left', **csfont)

style = {'fontname':'Arial', 'fontsize' : 9, 'rotation' : 20, 'weight': 'normal'} 

for i, txt in enumerate(names):
    ax1.annotate(txt, (np.setdiff1d(longitudes_wells, longitudes_zoom, assume_unique=True)[i] - .3, \
                        np.setdiff1d(latitudes_wells, latitudes_zoom, assume_unique=True)[i] - .28) , **style, zorder=6)
    
ax_zoomin = ax1.inset_axes([-0.1, 0.45, 0.4, 0.3])
ax_zoomin.set_xlim([-102.2, -100.5])
ax_zoomin.set_ylim([40, 41.2])
ax_zoomin.scatter(longitudes_zoom, latitudes_zoom, np.abs(s_zoom)*10, c= s_zoom, norm=plt.Normalize(-8,8),cmap='RdYlGn', zorder=5)

ax_zoomin.xaxis.set_visible(False)
ax_zoomin.yaxis.set_visible(False)

ax_zoomin.set_facecolor('tan')
for i, txt in enumerate(names_zoom):
    ax_zoomin.annotate(txt, (longitudes_zoom[i]-.03, latitudes_zoom[i]+.03), **style, zorder=6)

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
        facecolor='none', edgecolor=[0,0,0,1],linewidth=1., zorder=1)
    
    axe.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=1)) 
    
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
        facecolor='tan', edgecolor=[0,0,0,1],linewidth=0.2, zorder=0)

cbar_ax2 = fig.colorbar(plt.cm.ScalarMappable(cmap='BrBG',norm=plt.Normalize(-5, 5)), ax=ax2,
             orientation='horizontal',fraction=0.03, pad=0.11, extend='both')

cbar_ax2.set_label("cm / year", labelpad=0.01, font='Arial', fontsize=11)
#cbar_ax2.set_ticklabels([-5, 0, 5], **csfont)

cbar_ax1 = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-20, 20)),ax=ax1, \
                        fraction=0.065, extend='both', pad=0.05)
cbar_ax1.set_label("cm/year", labelpad=0.01, font='Arial', fontsize=12, rotation=90)

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

fig.savefig('figures/trends.pdf', dpi=200, edgecolor='none', bbox_inches='tight')


#%%

## annual cycle GWA 
# 
# """"""""""""""""""

from matplotlib.ticker import FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter 


#def plot_map12(out, matrix, title, labeli, vals, cmap='bwr', mask=False, norm=True):
#plot_map12('figures/anual_cycle_GWa.jpg', GWa_annual , dates_annual, r'$GW \, anomalies [cm]$', 8, cmap='seismic_r', mask=True)
variables = [GWa_annual_N, GWa_annual_C, GWa_annual_S]
nombres_GRACE = ["GWa North", "GWa Central", "GWa South"]
months = ['J', 'F', 'M', 'A', 'M ', ' J', ' J ', 'A', 'S', 'O', 'N', 'D']

csfont = {'fontname':'Arial', 'fontsize' : 12}
reader = shpreader.Reader('data/HPA/HP.shp')
mascara = np.genfromtxt('data/mask.csv', delimiter = ',')

cmap = 'seismic_r'
matrix = GWa_annual
title = dates_annual
labeli = "GW anomalies [cm]"
vals = 10


fig ,ax= plt.subplots(3,6,figsize=(7.8,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    
    mes = matrix[i,:].reshape(24,18)

  #  if mask == True:
  #      mes = mes * mascara

    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    
#if norm:
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=-1*vals)
    # else:
    #     ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals[1],vmin=vals[0])
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

#fig.suptitle('a)', x=0.05, y=0.9,**csfont)
# fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

# #if norm:
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(-1*vals,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1,label=labeli, extend='both')

ax_b = fig.add_axes([0.1, 0.0001, 0.41, 0.23])


colors_s = ['r', 'darkblue', 'k']
for i in range(len(variables)):
    ax_b.plot(title, spatial_mean(variables[i]), label=nombres_GRACE[i], color=colors_s[i])   

#ax_b.set_title('b)', loc='left', **csfont)
ax_b.set_xlabel('Time')
ax_b.set_ylabel('GWa [cm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
#ax_b.legend(loc=3)
ax_b.set_ylim(-4, 4)
     
ax_c = fig.add_axes([0.55, 0.0001, 0.4, 0.23])
#axes = fig.add_subplot(3, 2, 5)
#axes.fill_between(dates_complete, -15, 15, where = mask_nan, color='gray', alpha=0.3)
ax_c.plot(estandarizar(av_north), label='North', c='r')
ax_c.plot(estandarizar(av_central), label='Central', c='darkblue', ls='-')
ax_c.plot(estandarizar(av_south), label='South', c='k')
#ax_c.set_title('c)', loc='left', **csfont)
ax_c.set_xlabel('Time', **csfont)
#ax_c.set_ylabel('TWS anomaly [cm]', **csfont)
ax_c.set_ylim(-2, 2)
ax_c.legend(prop={'family' : 'Arial'})
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(alpha=0.3)
     
month_fmt = DateFormatter('%b')
def m_fmt(x, pos=None):
    return month_fmt(x)[0]

ax_b.set_xticklabels(months)
ax_c.set_xticklabels(months)
# ax_b.xaxis.set_major_locator(MonthLocator())
# ax_b.xaxis.set_major_formatter(FuncFormatter(m_fmt(title)))

# ax_c.xaxis.set_major_locator(MonthLocator())
# ax_c.xaxis.set_major_formatter(FuncFormatter(m_fmt(title)))

ax_c.text(0.04, 0.86, "a)", transform=fig.transFigure, **csfont)
ax_c.text(0.04, 0.24, "b)", transform=fig.transFigure, **csfont)
ax_c.text(0.5, 0.24, "c)", transform=fig.transFigure, **csfont)

#fig.subplots_adjust(hspace=1)

#fig.tight_layout()
fig.savefig('figures/GWa_annual_cycle.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')



#%%


# ANUAL CYCLE OF PRECIPITATION


csfont = {'fontname':'Arial', 'fontsize' : 12}
reader = shpreader.Reader('data/HPA/HP.shp')
mascara = np.genfromtxt('data/mask.csv', delimiter = ',')

precip_ = precip
precip_annual2, dates_annual = annual_cycle(precip_, pd.date_range('1981-01-01','2022-10-01', freq='M'))

precip_to_m,_ = annual_cycle(precip_*mask, pd.date_range('1981-01-01','2022-10-01', freq='M'))

Precip_N = (mask*precip_to_m.reshape(12, 24, 18))[:, :8, :]
Precip_C = (mask*precip_to_m.reshape(12, 24, 18))[:, 8:15, :]
Precip_S = (mask*precip_to_m.reshape(12, 24, 18))[:, 15:, :]

Precip_N_mean = spatial_mean(Precip_N.reshape(12, Precip_N.shape[1] * Precip_N.shape[2]))
Precip_C_mean = spatial_mean(Precip_C.reshape(12, Precip_C.shape[1] * Precip_C.shape[2]))
Precip_S_mean = spatial_mean(Precip_S.reshape(12, Precip_S.shape[1] * Precip_S.shape[2]))
#print(np.sum(spatial_mean(precip_to_m)), 'cm/year in total')
print(np.sum(Precip_N_mean), \
      'cm/year in north')
print(np.sum(Precip_C_mean), \
      'cm/year in central')

print(np.sum(Precip_S_mean), \
      'cm/year in south')



#%%

cmap = 'Blues'
matrix = precip_annual2
matrix2 = precip_to_m
title = dates_annual
labeli = "Rain [cm]"
vals = 12


fig ,ax= plt.subplots(3,6,figsize=(7,6),sharex=False, sharey=False,\
                        dpi=200,subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(12):
    
    mes = matrix[i,:].reshape(24,18)

  #  if mask == True:
  #      mes = mes * mascara

    ax.flat[i].add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
            facecolor='none', edgecolor=[0,0,0,1],linewidth=1)
    ax.flat[i].add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
        name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=0.5))
    ax.flat[i].set_title(title[i] ,loc='left', **csfont)
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-103,-100,-97],
                    draw_labels=False,linewidth=1,color='k',alpha=0.3, linestyle='--')
    
#if norm:
    ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals,vmin=0)
    # else:
    #     ax.flat[i].contourf(lon_grace,lat_grace, mes, cmap=cmap,vmax=vals[1],vmin=vals[0])
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

#fig.suptitle('a)', x=0.05, y=0.9,**csfont)
# fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.1, 0.31, 0.8, 0.05])
cbar_ax.axis('off')
[tick.set_visible(False) for tick in cbar_ax.xaxis.get_ticklabels()]
[tick.set_visible(False) for tick in cbar_ax.yaxis.get_ticklabels()]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)

# #if norm:
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, \
        norm=plt.Normalize(0,vals)),ax=cbar_ax,orientation='horizontal',\
            fraction=0.9,pad=0.02,aspect=100,shrink=1, extend='both')


ax_b = fig.add_axes([0.1, 0.0001, 0.8, 0.23])


#ax_b.bar(title, spatial_mean(matrix2) , color='dodgerblue')

df_precip_mean = pd.DataFrame(np.stack((Precip_N_mean, Precip_C_mean, Precip_S_mean), axis=1),\
                              index=title, columns=['North', 'Central', 'South'])
df_precip_mean.plot(kind="bar", rot=0, ax=ax_b,  color=['steelblue', 'k', 'dodgerblue'])

#ax_b.set_title('b)', loc='left', **csfont)
ax_b.set_xlabel('Time')
ax_b.set_ylabel('Rain [cm]')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(color='gray', linestyle='--', alpha=0.3)
#ax_b.legend(loc=3)
ax_b.set_ylim(0, 10)

ax_b.text(0.45, 0.28, labeli, transform=fig.transFigure, **csfont) # color bar label, la puse aca para facilidad

ax_b.text(0.02, 0.85, "a)", transform=fig.transFigure, **csfont)
ax_b.text(0.02, 0.28, "b)", transform=fig.transFigure, **csfont)

#fig.tight_layout()
fig.savefig('figures/precip_annual_cycle.pdf', dpi=200, transparent=False, edgecolor='none',bbox_inches='tight')
