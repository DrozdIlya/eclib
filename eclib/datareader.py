import glob
import xarray as xr
import pandas as pd
import numpy as np

def read_all_files(func, files_pattern, logger=None):
    '''
    Последовательно считывает функцией 'func' файлы, удовлетворяющие 'files_pattern', и объединяет их в 'df'.
    
    Parameters
    ----------
    func : function
        Читалка для одного файла (например  nc_to_df).
    files_pattern : str
        Шаблон полного имени файлов. 
    logger : logging.Logger; optional
        Объект вывода для лога.
        Default: None
        
    Returns
    -------
    df : pd.DataFrame
        Датафрейм, содержащий данные из всех считанных файлов. 
    
    Необходимо, чтобы считываемые файлы были однотипными и все читались функцией 'func'.
    При ошибке считывания файла выдаст сообщение об ошибке и продолжит считывание со следующего файла.
    Поддерживает логирование.
    '''
    
    files = glob.glob(files_pattern)
    files.sort()
    
    df = pd.DataFrame()

    if logger:
        logger.info(f'Files found: {len(files)}')
    i=0
    for file in files:
        try:
            new_df = func(file)
            df = pd.concat([df,new_df])
            if logger:
                logger.info(f'{file} has been read')
            i+=1
        except:
            if logger:
                logger.error(f'Error: {file}')
            break
    
    df = df.astype('float64')
    df.index = pd.to_datetime(df.index)
    if logger:
        logger.info(f'Files read: {i}')
    return df



def nc_to_df(file):
    '''
    Считывает netcdf файлы и конвертирует их в 'df'.
    
    Parameters
    ----------
    file : str
        Полное имя файла.
    
    Returns
    -------
    df : pd.DataFrame
        Датафрейм, содержащий данные из считанного файла.
    '''
    
    ds = xr.open_dataset(file)

    df = ds.to_dataframe()
    
    return df
