
#%%

import os
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import scipy.signal as signal
from scipy.fft import fft, fftfreq


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

def deseasonal(df):
    
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

def detrend(serie):
    
    serie_out = signal.detrend(serie, type='linear')
    
    return serie_out

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
#         # Si hay tendencia positiva
        if Zmk >= Z_1:
         #    tendencia.append('Existe tendencia positiva')
             return 1., np.round(b_o,2), Zmk
#         # Si hay tendencia positiva
        else:
#             return -.,b_o,p,Zmk
              return -1., np.round(b_o,2), Zmk
        
#         #return +1., Zmk
    
    # No hay tendencia en la serie
    else:       
        return 0,np.round(b_o,2), Zmk
    
def MK_test_df(df):
    
    df_out = pd.DataFrame(index = df.columns, columns = ['Trend', 'Slope', 'Z'])
    
    for columna in df.columns:
        df_out.loc[columna,:] = MK_test(df[columna].values)
        
    return df_out

def detrend_df(df):
    
    df_out = pd.DataFrame(index = df.index, columns = df.columns)
    
    for columna in df.columns:
 
        df_out[columna] =  signal.detrend(df[columna].values, type='linear')  
            
    return df_out

def plot_map_slopes(s, locacion, locacion_zoom, names, names_zoom, title, label, cmap='RdYlGn'):
    
    longitudes = locacion["LONGITUDE"].values
    latitudes = locacion["LATITUDE"].values

    longitudes_zoom = locacion_zoom["LONGITUDE"].values
    latitudes_zoom = locacion_zoom["LATITUDE"].values

    reader = shpreader.Reader('data/HPA/HP.shp')
    fig = plt.figure(figsize=(7,9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    
    ax.add_feature(cfeature.ShapelyFeature(reader.geometries(),ccrs.PlateCarree()),\
               facecolor='powderblue', edgecolor=[0,0,0,1],linewidth=.25, zorder=0)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural',scale='50m',
          name='admin_1_states_provinces_lines',edgecolor='k', facecolor='none',linewidth=1))
          #ax.clabel(mapa, inline=2, fontsize=12)
    ax.set_title(title,loc='left',fontsize=16, fontweight='bold')
    ax.scatter(longitudes, latitudes, np.abs(s)*10, c= s, norm=plt.Normalize(-8,8),cmap=cmap)
    ax.set_extent([-105, -97, 32, 44], crs=ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(),xlocs=[-106,-104,-102,-100,-98,-96],
                  draw_labels=True,linewidth=2,color='k',alpha=0.3, linestyle='--')             
    gl.top_labels = False
    gl.right_labels = False
     #gl.xlines = False 
    gl.ylocator = mticker.FixedLocator([31,33,35,37,39,41,43,45])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        
    style = dict(size=8, color='k', weight='bold', rotation=20)
    
    for i, txt in enumerate(names):
        ax.annotate(txt, (np.setdiff1d(longitudes, longitudes_zoom, assume_unique=True)[i] + .03, \
                          np.setdiff1d(latitudes, latitudes_zoom, assume_unique=True)[i] + .03) , **style)
     
    ax_zoomin = ax.inset_axes([-0.1, 0.45, 0.4, 0.3])
    ax_zoomin.set_xlim([-102.2, -100.5])
    ax_zoomin.set_ylim([40, 41.2])
    ax_zoomin.set_facecolor('powderblue')
    ax_zoomin.scatter(longitudes_zoom, latitudes_zoom, np.abs(s_zoom)*10, c= s_zoom, norm=plt.Normalize(-8,8),cmap=cmap)

    ax_zoomin.xaxis.set_visible(False)
    ax_zoomin.yaxis.set_visible(False)
    
    for i, txt in enumerate(names_zoom):
        ax_zoomin.annotate(txt, (longitudes_zoom[i]+.03, latitudes_zoom[i]+.03), **style)
    
    ax.indicate_inset_zoom(ax_zoomin, edgecolor="black", lw=2)

    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-20, 20)),ax=ax, label=label, extend='both')
        
    plt.show()

def FFT(serie):

    FTT = np.fft.fft(serie, norm = 'forward')

    Freq = np.fft.fftfreq(len(serie), 1)
    Fpos = Freq[ Freq > 0 ]
    periodos = 1 / Fpos
    Power = np.abs(FTT) ** 2
    
    df = Freq[1] - Freq[0]
    P = np.sum(Power) * df

    return (periodos, Power[:len(Fpos)])
 
def plot_FFT(PC_score, names,  oni, pdo, numero_PCs=2, func=FFT):
    
    fig, ax = plt.subplots(numero_PCs+2,1, sharex=True, figsize=(5,8))

    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    csfont = {'fontname':'Arial', 'weight':'bold'}

    plt.xlabel('Period [Months]', **csfont)
    labeli = ['ONI', 'PDO']
#    mode = (oni_p, pdo_p)

    dof= 3
    for i in range(numero_PCs):
        
     #   ax[i].set_xscale('log')
        #ax[i].set_ylim(0,)
        ax[i].axvline(12*2,linestyle='--', color='r', alpha=0.4)
        ax[i].axvline(12*7, linestyle='dashed', color='r', alpha=0.4)
        ax[i].plot(func(PC_score[:,i])[0], func(PC_score[:,i])[1], \
                        label= names[i], color='royalblue')
  #      ax[i].axhline(chi_val_95,color='0.4',linestyle='--')
    #    ax[i].fill_between(1//FFT(rotated_PC[:,i])[2],FFT(rotated_PC[:,i])[3]*confidence_fft(dof)[0]\
    #                       ,FFT(rotated_PC[:,i])[3]*confidence_fft(dof)[1],alpha=0.3)
        ax[i].legend(loc=2, frameon=False) 
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(color='gray', linestyle='--', alpha=0.3)
    

    ax[numero_PCs].plot(func(oni[:])[0], func(oni[:])[1], label=labeli[0])
#    ax[numero_PCs].axhline(chi_val_95,color='0.4',linestyle='--')
    ax[numero_PCs+1].plot(func(pdo[:])[0], func(pdo[:])[1], label=labeli[1])
  #  ax[numero_PCs+1].axhline(chi_val_95,color='0.4',linestyle='--')

    for axes in ax.flat[numero_PCs:]:
        
        axes.axvline(12*2,linestyle='--', color='r', alpha=0.4)
        axes.axvline(12*6, linestyle='dashed', color='r', alpha=0.4)
        axes.legend(loc=2)
    #    axes.set_ylim(0,100)
        axes.set_xlim(1,210)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.grid(color='gray', linestyle='--', alpha=0.3)
    
    axe = fig.add_subplot(111,frameon=False)
    axe.grid(False)
    axe.tick_params(axis=u'both', which=u'both',length=0)
    axe.set_xlabel('Period [Months]', labelpad=20, **csfont)
    axe.get_xaxis().set_visible(False)
    plt.ylabel('Power', labelpad=20, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]

def fourier_filter(serie, freq_min=24, freq_max=84):
            
    # Freq, S = scipy.signal.welch(serie, window='boxcar',return_onesided=False, nperseg=array_filtrado.shape[0])
    # S[np.where((np.abs(1/Freq)>(freq_max)) | (np.abs(1/Freq) < (freq_min)))]= 0
    # array_filtrado[:,lat,lon] = np.fft.ifft(S)
    Freq = np.fft.fftfreq(len(serie), d=1) 
    fourier  = np.fft.fft(serie, norm="forward" )

    fourier[np.where((np.abs(1/  Freq) > (freq_max)) | (np.abs(1 / Freq) < (freq_min)))] = 0
    inverse = np.fft.ifft(fourier)

    periodos = 1 / Freq
    power = abs(fourier)**2

    return periodos, power, inverse

def fourier_filter_df(df, freq_min=24, freq_max=84):
    
    df_out = pd.DataFrame(index = df.index, columns = df.columns)
    
    for columna in df.columns:
 
        df_out[columna] = fourier_filter(df[columna].values, freq_min, freq_max)[2]
            
    return df_out

def plot_scores(df, number, label, oni, pdo, dates):

    "matrix: PC score vector"
    "eigenval: eigenvalue of the val matrix"
    
    ncols = 3
    nrows = (number // ncols) + 1

    fig, ax = plt.subplots(nrows, ncols,figsize=(16,7), sharex=True, sharey=True)
    for i in range(number):

        ax.flat[i].plot(dates, df.values[:,i], label=df.columns[i])
 #       ax.flat[i].set_xlabel('Time [Months]')
 #       ax.flat[i].set_ylabel('Amplitude [cm]')
#        ax.flat[i].set_title(str(per_var) + '% Total variance', fontweight='bold', loc='left')
        ax.flat[i].spines['top'].set_visible(False)
        ax.flat[i].spines['right'].set_visible(False)
        ax.flat[i].grid(color='gray', linestyle='--', alpha=0.3)
        ax.flat[i].legend(frameon=False, loc=2)
        ax0 = ax.flat[i].twinx()
        ax0.plot(dates, oni.values, label='ONI', color='k')
        ax0.fill_between(dates,pdo.values[:],0, color='grey', alpha=0.5, label='PDO')
  #      ax0.set_ylabel('SST anomalies [°C]')
        ax0.spines['top'].set_visible(False)
 #       ax0.spines['right'].set_visible(False)
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
    axe0.set_ylabel('SST anomalies [°C]',labelpad=25)
    axe.set_xlabel('Time [Months]', labelpad=25)
  #  axe.set_ylabel('', labelpad=55, **csfont)
    [tick.set_visible(False) for tick in axe.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe.yaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe0.xaxis.get_ticklabels()]
    [tick.set_visible(False) for tick in axe0.yaxis.get_ticklabels()]

    plt.show()

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

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
df_anom = df_anom['1979':'06-2018']

df_anom = df_anom.drop('NE04', axis=1)


# INTERPOLAR Y DESCOMPONER CON ANOMALIAS EN PERIODO DE GRACE

df_gmean_ = (df - df.mean()) * ft_to_cm * s_y

df_gmean_ = df_gmean_.interpolate('linear', limit_direction = 'both')[1:]
df_gmean_ = df_gmean_['1979':'06-2021']
df_gmean_.describe()

df_gmean_ = df_gmean_.drop('NE04', axis=1)
#df_gmean_ = detrend_df(df_gmean_)

#df_gmean_.plot(figsize=(14,8), subplots=True, layout=(6,4), title='ANOMALIAS')
#plt.show()

df_gmean_trend = deseasonal(df_gmean_)[0]
#df_gmean_trend.plot(figsize=(16,7), subplots=True, layout=(7,3), title='TENDENCIA')
#plt.show()

df_gmean_season = deseasonal(df_gmean_)[1]
#df_gmean_season.plot(figsize=(14,8), subplots=True, layout=(6,4), title='CLIMATOLOGIA')
#plt.show()

df_gmean_residuales = deseasonal(df_gmean_)[2]
#df_gmean_residuales.plot(figsize=(16,7), subplots=True, layout=(7,3), title='RESIDUALES')
#plt.show()

#%%

# CICLO ANUAL

# average = df_gmean_.groupby(df_gmean_residuales.index.month).mean()
# average.index = pd.date_range('1/15/2020', '1/15/2021', freq='M').strftime('%B')
# average.plot(figsize=(14,8),subplots=True, layout=(6,4), title= 'CICLO ANUAL')

#%%
# MANN KENDALL

mann_kendall = MK_test_df(df_anom)
#mann_kendall = MK_test_df(df_gmean_season + df_gmean_residuales)
    
locacion = pd.read_excel('data/loc_pozos.xls', index_col=0)
locacion = locacion.drop('NE04')

# trend = []
# df_detrended = pd.DataFrame(index=df_anom.index, columns=df_anom.columns)

# for i, well in enumerate(df_anom.columns):
#     serie = df_anom[well]
#     nans, x = nan_helper(np.array(serie))
#     serie[nans]= np.interp(x(nans), x(~nans), serie[~nans])

#     x0 = np.linspace(0,10, len(serie))
# #      y = array_interp[:,lat,lon]
#     model = np.polyfit(x0, serie, 1)
#     predicted = np.polyval(model, x0)
#     df_detrended[well] = serie - predicted
#     trend.append((predicted[11]-predicted[0])/(x0[11]-x0[0]))

s = mann_kendall['Slope'].values.astype('float16')

# Zoom
locacion_zoom = locacion['NE01':'NE12']
s_zoom = mann_kendall['Slope']['NE01':'NE12'].values.astype('float16')

names = df_anom.columns
names_zoom = names[1:10]
names = np.setdiff1d(names, names_zoom)

plot_map_slopes(s, locacion, locacion_zoom, names, names_zoom, "","")


# SUMAR TENDENCIA Y RESIDUALES (EN LA TENDENCIA DE LOESS PUEDE HABER INFORMACION)

df_gmean_deseasonal = df_gmean_trend + df_gmean_residuales
df_gmean_deseasonal.plot(figsize=(14,8), subplots=True, layout=(6,4), title='TREND + RESIDUALES')
plt.show() 


#%%

# fig, axes = plt.subplots(3,1,figsize=(14,12))

# sns.heatmap(df_anom.corr(), ax=axes[0] , annot=True, cmap='Blues')
# sns.heatmap(df_gmean_deseasonal.corr(), ax=axes[1] , annot=True, cmap='Blues')
# sns.heatmap(df_gmean_residuales.corr(), ax=axes[2] , annot=True, cmap='Blues')

# plt.show()



#%%

"*****************************************************************************************************"

# ONI

data_oni = pd.read_csv('data/ONI.csv', index_col=0)
ONI_model = read_oni(data_oni, ['1979-01','2021-06'])

# PDO

data_pdo = pd.read_csv('data/PDO.csv', index_col=0, header=1)
PDO_model = read_pdo(data_pdo, ['1979-01','2021-06'])


"*********************************************************************************************************"

#plot_scores(c, 19, '', ONI, PDO, df_gmean_trend.index)

#%%
# ESTOY ESTANDARIZANDO PC SCORE, ESO SE HACE?
# pARECE QUE HAY SEÑAL PERO AJÁ

#plot_FFT(df_gmean_deseasonal.values, df_gmean_deseasonal.columns, ONI, PDO, 10)

plot_FFT(df_gmean_deseasonal.values, df_gmean_deseasonal.columns, ONI_model, PDO_model, 10, fourier_filter)


#%%

# FILTRANDO

df_filter = fourier_filter_df(df_gmean_deseasonal)

plot_FFT(df_filter.values, df_filter.columns, ONI_model, PDO_model, 10, FFT)

plot_scores(df_filter, 19, '', ONI_model, PDO_model, df_filter.index)


average = df_filter.groupby(df_filter.index.month).mean()
average.index = pd.date_range('1/15/2020', '1/15/2021', freq='M').strftime('%B')
average.plot(figsize=(14,8),subplots=True, layout=(6,4), title= 'CICLO ANUAL')