import pandas as pd
import numpy as np
from eclib.preprocessing import create_bins

def counts(df, df_bins=None, step=None, start=None, stop=None, prefix=None):
    '''
    Считает в 'df' количество не пустых значений по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащие данные, для которых будет производиться фильтрация воротами.
    limit : int, float
        Размер допустимого отклонения от среднего. 
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'df' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'df' (не рекомендуется, см. create_bins).
        Default: None.
    prefix : bool
        Если True, приписывает префикc "_count" к названиям колонок выходного DataFrame.  
        Default: None
    
    Returns
    -------
    df_count : pd.DataFrame 
        DataFrame, содержащий осредненные значения по входному df.
    '''
    
    if not df_bins: 
        df_bins = create_bins(step, df, start, stop)
        
    df_count = df.groupby(df_bins, observed=True).count()

    df_count.index = df_count.index.map(lambda x: x.left)

    if prefix: 
        df_count = df_count.add_prefix('_count')

    return df_count



def kurtosis(df, df_bins=None, step=None, start=None, stop=None, prefix=None):
    '''
    Считает в 'df' коэффициент эксцесса по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащие данные, для которых будет рассчитываться коэффициент эксцесса.
    limit : int, float
        Размер допустимого отклонения от среднего. 
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'df' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'df' (не рекомендуется, см. create_bins).
        Default: None.
    prefix : bool
        Если True, приписывает префикc "_kurt" к названиям колонок выходного DataFrame.  
        Default: None
    
    Returns
    -------
    df_kurt : pd.DataFrame 
        DataFrame, содержащий осредненные значения по входному 'df'.
    '''
    
    if not df_bins: 
        df_bins = create_bins(step, df, start, stop)
    
    df_kurt = df.groupby(df_bins, observed=True).agg(pd.Series.kurt)

    df_kurt.index = df_kurt.index.map(lambda x: x.left)

    if prefix: 
        df_kurt = df_kurt.add_prefix('_kurt')

    return df_kurt



def skewness(df, df_bins=None, step=None, start=None, stop=None, prefix=None):
    '''
    Считает в 'df' коэффициент асимметрии по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащий данные, для которых будет рассчитываться коэффициент асимметрии.
    limit : int, float
        Размер допустимого отклонения от среднего. 
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'df' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'df' (не рекомендуется, см. create_bins).
        Default: None.
    prefix : bool
        Если True, приписывает префик "_skew" к названиям колонок выходного DataFrame.  
        Default: None
    
    Returns
    -------
    df_skew : pd.DataFrame 
        DataFrame, содержащий осредненные значения по входному 'df'.
    '''
    
    if not df_bins: 
        df_bins = create_bins(step, df, start, stop)
        
    df_skew = df.groupby(df_bins, observed=True).skew()

    df_skew.index = df_skew.index.map(lambda x: x.left)

    if prefix: 
        df_skew = df_skew.add_prefix('_skew')

    return df_skew



def angle_of_attack_counts(df, df_bins=None, step=None, start=None, stop=None, u_name='u', v_name='v', w_name='w', minaa=-30, maxaa=30):
    '''
    Считает в 'df' количество углов атаки за пределами 'minaa' и 'maxaa' по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Входной датафрейм, содержащий данные, для которых будет рассчитываться угол атаки.
    df_bins : pandas.core.arrays.categorical.Categorical; optional
        Объект, содержащий границы интервалов осреднения. Если не задан, используются 'step', 'start', 'stop'.  
        Default: None.
    step : int, float, Timedelta; optional
        Длина интервала осреднения. 
        Используется, если не задан 'df_bins'. Если не заданы 'step' и 'df_bins', программа закончится ошибкой.
        Default: None.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'df' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'df' (не рекомендуется, см. create_bins).
        Default: None.
    u_name : str; optional
        Имя колонки 'df' содержащий u компоненту скорости ветра.
        Default: 'u'
    v_name : str; optional
        Имя колонки 'df' содержащий v компоненту скорости ветра.
        Default: 'v'
    u_name : str; optional
        Имя колонки 'df' содержащий w компоненту скорости ветра.
        Default: 'w'
    minaa : int, float; optional
        Минимально допустимый угол атаки.
        Default: -30
    maxaa : int, float; optional
        Максимально допустимый угол атаки.
        Default: 30

    Returns
    -------
    bad_angle_counts :  pd.Series 
        Временная серия, содержащая количество углов атаки за пределами 'minaa' и 'maxaa' по периодам осреднения 'df_bins'.
    angles : pd.Series
        Временная серия, содержащая моментальные углы атаки за весь период
        
    '''
    if not df_bins: 
        df_bins = create_bins(step, df, start, stop)
        
    ws = (df[u_name]**2+df[v_name]**2)**(1/2)
    angles = np.degrees(np.arctan(df[w_name]/ws))
    bad_angle_counts = ((angles < minaa) | (angles > maxaa)).groupby(df_bins, observed=True).sum()
    bad_angle_counts.index = bad_angle_counts.index.map(lambda x: x.left)
    
    return bad_angle_counts, angles