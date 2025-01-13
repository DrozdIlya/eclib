import eclib.datareader as dr
import eclib.preprocessing as pp
import eclib.calculation as ec
import eclib.dataquality as dq
from datetime import timedelta
import pandas as pd
import os



def processing(df, avg_period, start, stop, output_path = '.', inplace = False):

    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)
    step = timedelta(minutes=avg_period) 
    stop += timedelta(seconds=1)
    
    if inplace:
        df1 = df
    else:
        df1 = df.copy()
          
    df_bins = pp.create_bins(df1, step, start, stop)

    counts_before_processing = dq.counts(df, df_bins)
    
    pp.absolute_limits_filtration(df1.t, ulim = 40, blim = -40, inplace = True)
    pp.absolute_limits_filtration(df1.u, ulim = 30, blim = -30, inplace = True)
    pp.absolute_limits_filtration(df1.v, ulim = 30, blim = -30, inplace = True)
    pp.absolute_limits_filtration(df1.w, ulim = 5 , blim = -5 , inplace = True)

    pp.gates_filtration(df1.t, limit = 5 , df_bins = df_bins, inplace = True)
    pp.gates_filtration(df1.u, limit = 20, df_bins = df_bins, inplace = True)
    pp.gates_filtration(df1.v, limit = 20, df_bins = df_bins, inplace = True)
    pp.gates_filtration(df1.w, limit = 5 , df_bins = df_bins, inplace = True)

    pp.detrend(df1, df_bins, mode = 'dwm', inplace = True)

    pp.sigmas_filtration(df1.t, nsig = 3.5, n = 20, iterations = 10, df_bins = df_bins, inplace = True)
    pp.sigmas_filtration(df1.u, nsig = 3.5, n = 20, iterations = 10, df_bins = df_bins, inplace = True)
    pp.sigmas_filtration(df1.v, nsig = 3.5, n = 20, iterations = 10, df_bins = df_bins, inplace = True)
    pp.sigmas_filtration(df1.w, nsig = 5  , n = 20, iterations = 10, df_bins = df_bins, inplace = True)

    counts_before_gapfilling = dq.counts(df1, df_bins)
    pp.fillgaps(df1, inplace = True)
    counts_after_gapfilling = dq.counts(df1, df_bins)

    bad_angles_counts, _ = dq.angle_of_attack_counts(df1, df_bins, minaa = -30, maxaa = 30)

    _, angles_of_rotations = pp.axis_rotations(df1, D = 2, df_bins = df_bins, inplace = True)

    data_availability_flags = (counts_before_gapfilling / counts_before_processing * 100) < 80
    
    bad_angles_flags = (bad_angles_counts / counts_after_gapfilling.w * 100) > 10

    skew = dq.skewness(df1, df_bins)
    skew_flags = (skew < -2) | (skew > 2)
    
    kurt = dq.kurtosis(df1, df_bins)
    kurt_flags = kurt > 8

    hard_flags = (data_availability_flags + skew_flags + kurt_flags)
    hard_flags[['u','v','w']] = hard_flags[['u','v','w']].add(bad_angles_flags, axis=0)

    if output_path:
        counts_before_processing.to_csv(f'{output_path}/{start.date()}-{stop.date()}_counts_before_processing_{avg_period}min.csv')
        counts_before_gapfilling.to_csv(f'{output_path}/{start.date()}-{stop.date()}_counts_before_gapfilling_{avg_period}min.csv')
        counts_after_gapfilling.to_csv(f'{output_path}/{start.date()}-{stop.date()}_counts_after_gapfilling_{avg_period}min.csv')
        bad_angles_counts.to_csv(f'{output_path}/{start.date()}-{stop.date()}_bad_angles_counts_{avg_period}min.csv')
        skew.to_csv(f'{output_path}/{start.date()}-{stop.date()}_skewness_{avg_period}min.csv')
        kurt.to_csv(f'{output_path}/{start.date()}-{stop.date()}_kurtosis_{avg_period}min.csv')
        hard_flags.to_csv(f'{output_path}/{start.date()}-{stop.date()}_hard_flags_{avg_period}min.csv')
    
    return df1

    
    
def calculation(df, avg_period, start, stop, output_path = '.', inplace = False):

    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)
    step = timedelta(minutes=avg_period) 
    stop += timedelta(seconds=1)

    if inplace:
        df1 = df
    else:
        df1 = df.copy()
    
    df_bins = pp.create_bins(df1, step, start, stop)

    df1_means = ec.means(df1, df_bins)

    ec.pulsations(df1, df_bins, df_means=df1_means, inplace=True)

    df1_means['uu']     = ec.stat_moments(df1[['u']*2], df_bins)
    df1_means['vv']     = ec.stat_moments(df1[['v']*2], df_bins)
    df1_means['ww']     = ec.stat_moments(df1[['w']*2], df_bins)
    df1_means['tt']     = ec.stat_moments(df1[['t']*2], df_bins)
    df1_means['wu']     = ec.stat_moments(df1[['w','u']], df_bins)
    df1_means['wv']     = ec.stat_moments(df1[['w','v']], df_bins)
    df1_means['wt']     = ec.stat_moments(df1[['w','t']], df_bins)
    df1_means['wu_h']   = (df1_means.wu ** 2 + df1_means.wv ** 2) ** 0.5
    
    P = 101325  # [Pa]
    R = 287 # [J/(kg·K)]
    rho = P / (R * (df1_means.t + 273.15))  # [kg/m3]
    Cp = 1005  # [J/(kg·K)]
    
    df1_means['H']      = rho * Cp * df1_means.wt  # [W/m2]
    df1_means['tau']    = rho * df1_means.wu_h  # [N/m2]
   
    df1_means['u_star'] = df1_means.wu_h ** 0.5
    df1_means['L']      = -(df1_means.t + 273.15) * df1_means.u_star ** 3 / ( 9.8 * 0.4 * df1_means.wt) 
    df1_means['TKE']    = (df1_means.uu + df1_means.vv + df1_means.ww) / 2
    df1_means['A']      = df1_means.ww / (df1_means.uu + df1_means.vv + df1_means.ww) 
    df1_means['wuu']    = ec.stat_moments(df1[['w','u','u']], df_bins)
    df1_means['wvv']    = ec.stat_moments(df1[['w','v','v']], df_bins)
    df1_means['wtt']    = ec.stat_moments(df1[['w','t','t']], df_bins)
    df1_means['wwt']    = ec.stat_moments(df1[['w','w','t']], df_bins)
    df1_means['uuu']    = ec.stat_moments(df1[['u']*3], df_bins)
    df1_means['vvv']    = ec.stat_moments(df1[['v']*3], df_bins)
    df1_means['www']    = ec.stat_moments(df1[['w']*3], df_bins)
    df1_means['ttt']    = ec.stat_moments(df1[['t']*3], df_bins)

    if output_path:
        df1_means.to_csv(f'{output_path}/{start.date()}-{stop.date()}_moments_{avg_period}min.csv')
        
        
    return df1_means

if __name__ == '__main__':

    input_data = './test_data/msu/01/MSU_A1_*.nc'
    start = '2023-01-01'
    stop = '2023-02-01'
    avg_period = 30
    output_path = '.'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print('Data reading...')
    df = dr.read_all_files(func=dr.nc_to_df, files_pattern=input_data)
    df.rename(columns={'temp': 't'}, inplace=True)

    print('Data processing...')
    processing(df, avg_period, start, stop, output_path, inplace=True)

    print('Fluxes computation...')
    df_moments = calculation(df, avg_period, start, stop, output_path)
    df_moments.to_csv('output.csv')
    
