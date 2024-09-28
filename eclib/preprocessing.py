import pandas as pd
import numpy as np
from scipy import stats

def create_bins(ser, step, start=None, stop=None):
    '''
    Генерирует интервалы осреднения временной серии 'ser' от 'start' до 'stop' с шагом 'step'.
    
    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Датафрейм или временной ряд, содержащий данные, для которых будет производиться разбиение на интервалы осреднения.
    step : int, float, Timedelta
        Длина интервала разбиения. 
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. Если неуказано, берется первый индекс 'ser'.
        Default: None
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. Если неуказано, берется последний индекс 'ser'.
        Default: None

    Returns
    -------
    df_bins : pandas.core.arrays.categorical.Categorical
        Объект, содержащий границы интервалов осреднения ser.
    
    Функция поддреживает работу с 'ser', содержащими пропуски.
    Если 'start' и/или 'stop' не задан, функция определит его автоматически на основе 'ser' как первый и последний индекс массива.
    Сгенерированные границы периодов осреднения включают левое значение периода осреднения "[" и не включают правое ")".   
     
    Во избежание неточностей в выделяемых периодах осреднения стартовое значение 'start' и конечное значение 'stop' рекомендуется задавать явно.
    Генерация границ интервалов происходит по средствам функции np.arange(start, stop, step). 
    Как следстиве, генерируемые интервалы не включают границу 'stop', что может привести к потере последнего периода осреднения при автоматическом определении значения 'stop'.
    Например, если необоходимо получить разбиение по индексам от 0 до 100 с шагом 10, необоходимо задавать stop = 100.001 (немного превышающий требуемое значение).
    Если задать stop = 100, последний период осреднения будет [80:90) (см. документацию numpy к функции np.arange).  
    '''
    
    if not start: 
        start = ser.index[0]
    if not stop: 
        stop = ser.index[-1]
    
    new_indices = np.arange(start, stop, step)

    df_bins = pd.cut(ser.index, bins=new_indices, right=False, include_lowest=True)
    
    return df_bins



def find_start_and_lengt(mask):
    '''
    Находит все последовательности элементов, удовлетворяющих 'mask'. 
    Выводит начало 'starts', и длину 'lengths' для всех последовательностей элементов, удовлетворяющих 'mask'.

    Parameters
    ----------
    mask : двоичный массив
        Массив, содержащий масску элементов, для которых определяются 'starts' и 'lengths'.  

    Returns
    -------
    starts : list
        Список индексов начала последовательностей, удовлетворяющих 'mask'. 
    lengths : list
        Список длин последовательностей, удовлетворяющих 'mask'.
    '''
    
    starts = []
    lengths = []
    
    start = None
  
    for i, is_True in enumerate(mask):
        # если встречаем первый раз, запоминаем индекс как начало последовательности
        if is_True and start is None:
            start = i
        # если после этого встречаем нормальное значение, 
        # записываем индекс начала последовательности и разницу текущего индекса и индекса начала последовательности
        elif not is_True and start is not None:
            length = i - start
            starts.append(start)
            lengths.append(length)
            start = None

    # Если массив заканчивается на True, 
    # записываем индекс начала и последний индекс массива - индекс начала последовательности
    if start is not None:
        length = len(mask) - start
        starts.append(start)
        lengths.append(length)
    
    return starts, lengths



def absolute_limits_filtration(ser, ulim, blim, logger=None, inplace=False):
    '''
    Заменяет в 'ser' все значения, превышающие пороговые значения 'ulim', 'blim' на пустые (np.nan).

    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Датафрейм или временной ряд, содержащий данные, для которых будет производиться фильтрация пиков.
    ulim : int, float
        Верхнее пороговое значение для 'ser'.    
    blim : int, float
        Нижнее пороговое значение для 'ser'.
    logger : logging.Logger; optional
        Если задан, записывает лог. 
        Default: None.
    inplace : bool; optional
        Если False, сделает копию 'ser', если True перезапишет 'ser'.
        Default: False.   

    Returns
    -------
    ser : pd.DataFrame or pd.Series
        Объект аналогичный 'ser' с пустыми значениями вместо пиков. 

    Функция поддреживает работу с 'ser', содержащими пропуски.
    Поддерживает логирование.
    '''
    
    if not inplace:
        ser = ser.copy()

    if logger:
        before = ser.count()
    
    ser[(ser<blim)|(ser>ulim)] = np.nan
    
    if logger:
        after = ser.count()
        logger.info(f'Удалено значений {ser.name}: {before-after}, {np.round((before-after)/before*100)} %')
    
    return ser



def gates_filtration(ser, limit, df_bins=None, step=None, start=None, stop=None, logger=None, inplace=False):
    '''
    Осуществляет в 'ser' удаление значений лежащих за пределами "среднее значение +- 'limit'" по периодам осреднения 'df_bins'.

    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
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
        Используется, если не задан 'df_bins'. Если None, берется первый индекс 'ser' (не рекомендуется, см. create_bins). 
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. 
        Используется, если не задан 'df_bins'. Если None, берется последний индекс 'ser' (не рекомендуется, см. create_bins).
        Default: None.
    logger : logging.Logger; optional
        Если задан, записывает лог. 
        Default: None.
    inplace : bool; optional
        Если False, сделает копию ser, если True перезапишет 'ser'.
        Default: False.  
    
    Returns
    -------
    ser : pd.DataFrame or pd.Series
        Объект аналогичный 'ser' с пустыми значениями (np.nan) вместо пиков.

    Функция поддерживает ввод 'ser' только в формате pd.DataFrame и pd.Series, если формат не соответсвует указанному, функция сообщит о несовпадении формата и вернет пустой вывод.
    Функция поддерживает как числовые, так и временные индексы 'ser'.
    Функция поддреживает работу с 'ser', содержащими пропуски.
    Для применения опции inplace, используйте ввод ser=df или ser = df['val']. Функция не перезапишет исходный 'ser', если использовать ввод ser = df[['val']], даже при inplace = True.   
    '''
    if isinstance(ser, pd.DataFrame):
        return ser.apply(lambda x: drop_outliers(x, limit, df_bins, step, start, stop, logger, inplace))
    
    elif isinstance(ser, pd.Series):
        
        if not inplace: 
            ser = ser.copy()
        
        if not df_bins:
            df_bins = create_bins(ser, step, start, stop)
            
        # создает массивы границ превышений для каждого обрабатываемого периода осреднения
        ser_mean = ser.groupby(df_bins, observed=False).mean()
        ser_min = ser_mean - limit
        ser_max = ser_mean + limit
        ser_bins = ser_mean.index

        # цикл по периодам осреднения
        for bin, max, min in zip(ser_bins, ser_max, ser_min):

            # определяет маску превышений
            mask = (ser[bin.left:bin.right] > max) | (ser[bin.left:bin.right] < min) 
            
            # находит начало и длину всех превышений
            starts, lengths = find_start_and_lengt(mask)

            # цикл по превышениям
            for start, length in zip(starts, lengths):
        
                end = start + length
                ser[bin.left:bin.right].iloc[start:end] = np.nan
    
            # считает в периоде осреднения количесвто удаленных превышений 
            outliers_count = np.sum(lengths)
            
            if logger and outliers_count > 0:
                
                logger.info(f'Value: {ser.name}, period: {bin}, outliers count: {outliers_count} ({np.round(outliers_count/len(ser[bin.left:bin.right])*100,2)}%)')
     
        return ser
    else: 
        print(f"{type(ser)} - недопустимый формат 'ser'. Аргумент 'ser' должен быть pd.DataFrame или pd.Series")
        return 



def sigmas_filtration(ser, nsig=3.5, n=3, iterations=3, df_bins=None, step=None, start=None, stop=None, logger=None, inplace=False):
    '''
    Осуществляет в 'ser' удаление значений лежащих за пределами "mean +- 'nsig' * std" по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Входной датафрейм или временной ряд, содержащие данные, для которых будет производиться фильтрация пиков.
    nsig : int, float optional
        Размер допустимого отклонения от среднего в стандатных отклонениях. 
        Default: 3.5.
    n : int, float; optional
        Если длина отклонения превышает 'n', пик считается значимым и не удаляется.
        Default: 3.
    iterations : int; optional
        Количество итераций фильтраций. 
        Default: 3.
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
    logger : logging.Logger; optional
        Если задан, записывает лог. 
        Default: None.
    inplace : bool; optional
        Если False, сделает копию 'ser', если True, перезапишет 'ser'.
        Default: False.  
    
    Returns
    -------
    ser : pd.DataFrame or pd.Series
        Объект аналогичный 'ser' с пустыми значениями (np.nan) вместо пиков.
    
    Функция поддерживает ввод 'ser' только в формате pd.DataFrame и pd.Series, если формат не соответсвует указанному, функция сообщит о несовпадении формата и вернет пустой вывод.
    Функция поддерживает как числовые, так и временные индексы 'ser'.
    Функция поддреживает работу с 'ser', содержащими пропуски.
    Функция обрабатывает один и тот же период осреднения пока не достигнуто заданное количество итераций 'iterations' или пока все пики не будут удалены.
    Для применения опции inplace, используйте ввод ser=df или ser=df['val']. Функция не перезапишет исходный 'ser', если использовать ввод ser = df[['val']], даже при inplace = True.   
    '''
    
    if isinstance(ser, pd.DataFrame):
        return ser.apply(lambda x: drop_spikes(x, nsig, n, iterations, df_bins, step, start, stop, logger, inplace))
    
    elif isinstance(ser, pd.Series):
        
        if not inplace: 
            ser = ser.copy()
        
        if not df_bins:
            df_bins = create_bins(ser, step, start, stop)

        end_loop = True
        
        # цикл по итерациям
        for iteration in range(iterations):
            
            # создает массивы границ превышений для каждого обрабатываемого периода осреднения
            ser_mean = ser.groupby(df_bins, observed=False).mean()
            ser_std = ser.groupby(df_bins, observed=False).std()
            ser_min = ser_mean - nsig * ser_std
            ser_max = ser_mean + nsig * ser_std
            ser_bins = ser_mean.index

            # цикл по периодам осреднения
            for bin, max, min in zip(ser_bins, ser_max, ser_min):

                # определяет маску превышений
                mask = (ser[bin.left:bin.right] > max)|(ser[bin.left:bin.right] < min)

                # находит начало и длину всех превышений
                starts, lengths = find_start_and_lengt(mask)

                outliers_count = 0
                
                # цикл по превышениям
                for start, length in zip(starts, lengths):

                    if length <= n:
                        end = start + length
                        ser[bin.left:bin.right].iloc[start:end] = np.nan
                        outliers_count += length # считает в периоде осреднения количество обработанных значений
                 
                # если в периоде осреднения есть обработанные значения, флаг окончания итерационного цикла False, 
                # если в периоде осреднения нет обработанных значений, переходим к следующему периоду 
                if outliers_count > 0:
                    end_loop = False
                else:
                    continue

                if logger:
                    logger.info(f'Value: {ser.name}, period: {bin}, iteration {iteration+1}, spikes count: {outliers_count}')
               
            # если флаг окончания итерационного цикла True, цикл заканчивается, если False, меняется на True
            if end_loop:
                break
            else:
                end_loop = True
                
        return ser
    
    else: 
        print(f'{type(ser)} - недопустимый формат ser. Аргумент ser должен быть pd.DataFrame или pd.Series')
        return 



def detrend(ser, df_bins=None, step=None, start=None, stop=None, mode='detrend', min_val=3, logger=None, inplace=False):
    '''
    Осуществялет удаление тренда из данных 'ser' по периодам осреднения 'df_bins'.
    
    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Входной датафрейм или таймсерия, содержащие данные, для которых будет производиться детрендинг.
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
    mode : {'trend', 'detrend', 'dwm'}; optional
        Удаляет тренд, если mode = 'detrend'. Сохряняет только вычисленный тренд, если mode = 'trend'.
        Вычитает тренд, но оставляет среднее, если mode = 'dwm'.  
        Default: 'detrend'. 
    min_val: int; optional
        Минимальное количество непустых значений за период осреднения. Если значений меньше, весь период заполняется np.nan. 
        Default: 3.
    logger : logging.Logger; optional
        Если задан, записывает лог. 
        Default: None.
    inplace : bool; optional
        Если False, сделает копию 'ser', если True перезапишет 'ser'.
        Default: False.
    
    Returns
    -------
    ser_detrend : pd.DataFrame or pd.Series
        Объект аналогичный 'ser' c преобразованным трендом.
    
    Функция поддерживает ввод 'ser' только в формате pd.DataFrame и pd.Series, если формат не соответсвует указанному, функция сообщит о несовпадении формата и вернет пустой вывод.
    Функция поддерживает как числовые, так и временные индексы 'ser'.
    Функция поддреживает работу с 'ser', содержащими пропуски.
    Для применения опции 'inplace', используйте ввод ser = df или ser = df['val']. Функция не перезапишет исходный 'ser', если использовать ввод ser = df[['val']], даже при inplace = True.   
    
    '''
    if isinstance(ser, pd.DataFrame):
        return ser.apply(lambda x: detrend(x, df_bins, step, start, stop, mode, min_val, logger, inplace))
    
    elif isinstance(ser, pd.Series):
    
        if not inplace:
            ser = ser.copy()
        
        if not df_bins:
            df_bins = create_bins(ser, step, start, stop)

        ser_mean = ser.groupby(df_bins, observed=False).mean()
        ser_count = ser.groupby(df_bins, observed=False).count()
        ser_bins = ser_mean.index
        
        for bin, count, mean in zip(ser_bins, ser_count, ser_mean):

            if count >= min_val:

                y = ser[bin.left:bin.right]
                x = np.arange(len(y))

                not_nan = np.logical_not(np.isnan(y))

                slope, intercept, r_value, p_value, std_err = stats.linregress(x[not_nan], y[not_nan])
                
                if logger:
                    logger.info(f'Value: {ser.name}, period: {bin}, slope {slope}, intercept {intercept}')

                trend = slope * x + intercept
                
                if mode == 'trend': 
                    ser[bin.left:bin.right] = trend
                elif mode == 'dwm': 
                    ser[bin.left:bin.right] = y - trend + mean   
                else: 
                    ser[bin.left:bin.right] = y - trend
                    
            else:
                ser[bin.left:bin.right] = np.nan
        
        return ser
    
    else: 
        logger.error(f'{type(ser)} - недопустимый формат ser. Аргумент ser должен быть pd.DataFrame или pd.Series')
        return



def fillgaps(ser, inplace=False):
    if inplace:
        ser.interpolate(inplace=inplace)
    else: 
        return ser.interpolate()
    
    # # цикл по периодам осреднения
    # for bin in df_bins:

    #     # определяет маску превышений
    #     mask = ser[bin.left:bin.right].isna()

    #     # находит начало и длину всех превышений
    #     starts, lengths = find_start_and_lengt(ser[bin.left:bin.right], mask)

        
    #     # цикл по превышениям
    #     for start, length in zip(starts, lengths):

    #         if length == 1:
    #             ser[bin.left:bin.right].iloc[start] = (ser[bin.left:bin.right].iloc[start-length]+ser[bin.left:bin.right].iloc[start+length])/2
    #         else:
    #             end = start + length
                
    #             y = ser[bin.left:bin.right].iloc[start-length:end+length]
    #             x = np.arange(len(y))

    #             gap = np.isnan(y)
    #             not_gap = np.logical_not(gap)

    #             slope, intercept, _, _, _ = stats.linregress(x[not_gap], y[not_gap])
                
    #             if logger:
    #                 logger.info(f'Value: {ser.name}, period: {bin}, slope {slope}, intercept {intercept}')

    #             ser[bin.left:bin.right].iloc[start:end] = slope * x[gap] + intercept

    # return ser



def axis_rotations(df, D=2, df_bins=None, step=None, start=None, stop=None, u_name='u', v_name='v', w_name='w', logger=None, inplace=False):
    '''
    Поворачивает оси компонент скорости ветра вдоль преобладающего потока по периодам осреднения 'df_bins'.  
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, содержащий компоненты скорости u, v, w.  
    D : int; optional
        Количество поворотов осей. Для измерений над неоднородной поверхностью рекомендуется использовать 1 или 2 поворота (Finnigan, 2004)
        Default: 2.
    step : int, float, Timedelta
        Длина интервала осреднения.
    start : int, float, Timestamp; optional
        Начало обрабатываемого периода. Если неуказано, берется первый индекс df (не рекомендуется, см. bins).
        Default: None.
    stop : int, float, Timestamp; optional
        Конец обрабатываемого периода. Если неуказано, берется последний индекс df (не рекомендуется, см. bins).
        Default: None.
    u_name : str; optional
        Название колонки df, содержащей компоненту скорости u.
        Default: 'u'.
    v_name : str; optional
        Название колонки df, содержащей компоненту скорости v.
        Default: 'v'.
    w_name : str; optional
        Название колонки df, содержащей компоненту скорости w.
        Default: 'w'.
    logger : logging.Logger; optional
        Если задан, записывает лог. 
        Default: None.
    inplace : bool; optional
        Если False, сделает копию df, если True перезапишет df.
        Default: False.
        
    Returns
    -------
    df_rot_comp : pandas.core.frame.DataFrame
        Датафрейм идентичный df, содержащий развернутые компоненты скорости u, v, w.
    angles : pd.DataFrame
        Датафрейм, содержащий углы поворота для каждого вектора скорости за все периоды осреднения.
    '''
    
    if not inplace:
        df = df.copy()

    if D not in [1,2,3]:
        if logger:
            logger.error("Недопустимое количество поворотов осей, 'D' должно быть от 1 до 3.")
        return
    
    angles = pd.DataFrame()

    if not df_bins:
        df_bins = create_bins(ser, step, start, stop)
    
    if D >= 1:
        df_mean = df[[u_name, v_name]].groupby(df_bins, observed=False).mean()
        angles['Theta'] = np.arctan(df_mean[v_name] / df_mean[u_name])
                          
        rotation(df, u_name, v_name, angles.Theta)
        if logger:
            logger.info("Поворот вокруг оси 'z' выполнен.")
        
    if D >= 2:
        df_mean = df[[u_name, w_name]].groupby(df_bins, observed=False).mean()
        angles['Phi'] = np.arctan(df_mean[w_name] / df_mean[u_name])                
        
        rotation(df, u_name, w_name, angles.Phi)
        if logger:
            logger.info("Поворот вокруг оси 'y' выполнен.")
    
    # if D == 3:
    #     u1u2 = stat_moments(df[[v_name, w_name]], df_bins)
    #     u1u1 = stat_moments(df[[v_name, v_name]], df_bins)
    #     u2u2 = stat_moments(df[[w_name, w_name]], df_bins)
    #     angles['Psi'] = 0.5 * np.arctan(2 * u1u2 / (u1u1 - u2u2))
                                 
    #     rotation(df, v_name, w_name, angles.Psi)
    #     log.info("Поворот вокруг оси 'x' выполнен.")
    
    angles.index = angles.index.categories.left
    
    return df, angles



def rotation(df, u1, u2, angles):
    '''
    Осуществляет поворот компонент скорости df['u1'] и df['u2'] на угол 'angles' по всем периодам осреднения.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, содержащий компоненты скорости 'u1' и 'u2'.
    u1 : str
        Название колонки 'df', содержащей первую компоненту скорости.
    u2 : str
        Название колонки 'df', содержащей вторую компоненту скорости.
    angles : pd.Series
        Временная серия со значениями углов поворота для каждого интервала осреднения. 
    
    Returns
    -------
    df : pd.DataFrame 
    Объект идентичный входному 'df', содержащий развернутые компоненты скорости 'u1', 'u2'.
    '''
    
    sin = np.sin(angles)
    cos = np.cos(angles)

    U1 = df[u1].copy()
    U2 = df[u2].copy()

    for ind in angles.index:
        df.loc[ind.left:ind.right, u1] = U1.loc[ind.left:ind.right] * cos[ind] + U2.loc[ind.left:ind.right] * sin[ind]
        df.loc[ind.left:ind.right, u2] = -U1.loc[ind.left:ind.right] * sin[ind] + U2.loc[ind.left:ind.right] * cos[ind]
        
    return df
    