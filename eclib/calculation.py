import pandas as pd
import numpy as np
from eclib.preprocessing import create_bins

def means(df, df_bins=None, step=None, start=None, stop=None, prefix=False):
    '''
    Считает в 'df' среднее по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащий данные, для которых будет производится расчет средних значний.
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'ser' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'ser' (не рекомендуется, см. create_bins).
        Default: None.
    prefix : bool
        Если True, приписывает префик "_mean" к названием колонок выходного DataFrame.  
        Default: None
    
    Returns
    -------
    df_mean : pd.DataFrame 
        DataFrame, содержащий осредненные значения по входному df.
    '''
    
    if not df_bins: 
        df_bins = create_bins(step, df, start, stop)
        
    df_mean = df.groupby(df_bins, observed=True).mean()

    df_mean.index = df_mean.index.map(lambda x: x.left)

    if prefix: 
        df_mean = df_mean.add_prefix('_mean')

    return df_mean



def pulsations(ser, df_bins=None, step=None, start=None, stop=None, df_means=None, inplace=None):
    '''
    Рассчитывает в 'ser' пульсации по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащий данные, для которых будет производится расчет пульсаций.
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'ser' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'ser' (не рекомендуется, см. create_bins).
        Default: None.
    df_means : pd.DataFrame or pd.Series
        Датафрейм или временной ряд, содержащий средние значения для расчета пульсаций.
        Default: None,  
    inplace : bool; optional
        Если False, сделает копию 'ser', если True перезапишет 'ser'.
        Default: None
    
    Returns
    -------
    puls : pd.DataFrame
        Объект идентичный входному 'ser', содержащий рассчитанные пульсации.
    
    '''
    
    if not inplace: 
        ser = ser.copy()

    if not df_bins: 
        df_bins = create_bins(ser, step, start, stop)

    if not df_means:
        df_means = means(ser, df_bins)

    ind = ser.index[-1]
    
    for bins, group in ser.groupby(df_bins, observed=True):
        ser.loc[group.index, df_means.columns] -= df_means.loc[bins.left]
        ind = group.index[-1]
        
    puls = ser.loc[:ind]
    
    return puls



def stat_moments(puls, df_bins=None, step=None, start=None, stop=None):
    '''
    Рассчитывает по 'puls' турбулентные моменты по периодам осреднения 'df_bins' или от 'start' до 'stop' с шагом 'step'.
    
    Parameters
    ----------
    puls : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащий данные, для которых будет производится расчет пульсаций.
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'ser' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'ser' (не рекомендуется, см. create_bins).
        Default: None.
    
    Returns
    -------
    moments
    '''
    if not df_bins: 
        df_bins = create_bins(ser, step, start, stop)
        
    moments = puls.prod(axis=1, skipna=False)
    moments.name = ''.join(puls.columns)
    
    moments = means(moments, df_bins)
        
    return moments
