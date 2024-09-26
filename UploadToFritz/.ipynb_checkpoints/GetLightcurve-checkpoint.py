import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
#pd.options.mode.copy_on_write = True
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import multiprocessing as mp
from tqdm import tqdm
cosmo = FlatLambdaCDM(name='Planck18', H0=67.66, Om0=0.30966, Tcmb0=2.7255, Neff=3.046, m_nu=[0.  , 0.  , 0.06]* u.eV, Ob0=0.04897)
cores = 8

directory = '/hildafs/projects/phy220048p/leihu/AstroWork/DESIRT/IMPROC/SEARCH'
_21A_dir = f'{directory}/DESIRT21A/2021-06-10/NORMAL/Candidates'
_22A_dir = f'{directory}/DESIRT22A/2024-01-27/NORMAL/Candidates'
_22B_dir = f'{directory}/DESIRT22B/2024-01-18/NORMAL/Candidates'
_23A_dir = f'{directory}/DESIRT23A/2024-01-30/NORMAL/Candidates'
_23B_dir = f'{directory}/DESIRT23B/2024-01-30/NORMAL/Candidates'

Semdirlist = [_21A_dir,_22A_dir,_22B_dir,_23A_dir,_23B_dir]

sem_list = ['21A','22A','22B','23A','23B']

crossmatch_df =  pd.read_csv('../CrossMatch/DESIRT_one_arcsecond_crossmatch_final_list.csv')

usecols = ['MJD_OBS', 'MAG_FPHOT', 'LIM_MAG5', 'FILTER', 'MAGERR_FPHOT', 'STATUS_FPHOT']

num_plots = 100000 # how many plots to plot
plot_count = 0
mjd_threshold = 0.01 # Cutoff for time separation to combine data for color
bogus_mag_cutoff = 10 # Anything brighter than this will be removed from the data

bright = []

def make_name_dict(d1, d2, d3, d4, d5):
    name_dict = {
        d1: _21A_dir,
        d2: _22A_dir,
        d3: _22B_dir,
        d4: _23A_dir,
        d5: _23B_dir,
    }
    return name_dict

def get_semester_names(crossmatch, standard_name):
    transient_info = crossmatch[crossmatch['STANDARD_NAME'] == standard_name]
    # Get the individual names for each semester
    d1=transient_info['21A'].iloc[0]
    d2=transient_info['22A'].iloc[0]
    d3=transient_info['22B'].iloc[0]
    d4=transient_info['23A'].iloc[0]
    d5=transient_info['23B'].iloc[0]
    return (d1, d2, d3, d4, d5)

def full_dataframe_from_names(d1, d2, d3, d4, d5):
    transient_dfs = []
    names = make_name_dict(d1, d2, d3, d4, d5)
    
    for name in names:
        if not pd.isna(name):
            # We have data for this semester
            transient = name
            
            sem = [d1,d2,d3,d4,d5].index(name)
            
            #sem = sem_list[int([d1,d2,d3,d4,d5].)]
            use_dir = names[name]
            
            #full_file = ''
            
            for file in os.listdir(Semdirlist[sem]):
                
                mid = file.find('_') + 1
                
                if file[mid:(mid + 23)] == transient:
                    
                    full_file = file
            
           
            transient_data = table(Table.read(f"{use_dir}/{full_file}"))
            # Isolate filter from DIFNAME and Real from AlertID and remove DIFNAME/AlertID columns
            #transient_data= transient_data[transient_data['REAL']==True]
            
            valid_mask = ~np.isnan(transient_data['MAG_FPHOT'])
            keep_real_data = np.zeros_like(transient_data['MAG_FPHOT'], dtype=bool)
            keep_real_data[valid_mask] = transient_data['MAG_FPHOT'][valid_mask]>=bogus_mag_cutoff
            
            transient_data = transient_data[keep_real_data]
            
            transient_data.remove_columns(['PixA_THUMB_TEMP', 'PixA_THUMB_SCI', 'PixA_THUMB_DIFF'])
            
            transient_df = transient_data.to_pandas()
            semester_origin = transient_df.shape[0] * [transient]
            transient_df["SEMESTER_NAME"] = semester_origin
            
            transient_dfs.append(transient_df)

    return pd.concat(transient_dfs)

def full_dataframe(input_list):
    standard_name, df = input_list
    d1, d2, d3, d4, d5 = get_semester_names(df, standard_name)
    lc_df = full_dataframe_from_names(d1, d2, d3, d4, d5)
    standard_name_list = lc_df.shape[0] * [standard_name]
    lc_df["STANDARD_NAME"] = standard_name_list
    return lc_df

def table(transient_data):
    
    FILT = np.array(transient_data["FILTER"]).astype(str)
    DETTAG = np.array(transient_data["STATUS_FPHOT"]).astype(str)
    MAG = np.array(transient_data["MAG_FPHOT"]).astype(float)
    eMAG = np.array(transient_data["MAGERR_FPHOT"]).astype(float)
    TIME = np.array(transient_data["MJD_OBS"]).astype(float)
    LM5 = np.array(transient_data["LIM_MAG5"]).astype(float)

    fMASK_ND = DETTAG == "m"  # Non-Detections: SNR < 3 and no alert
    fMASK_PD = DETTAG == "q"  # Questionable-Detections: others
    fMASK_CD = DETTAG == "p"  # Convincing-Detection: SNR > 5 with alert
    
    valid_mask = ~np.isnan(eMAG)
    fMASK_SNR = np.zeros_like(eMAG, dtype=bool)
    fMASK_SNR[valid_mask] = 1.0857 / eMAG[valid_mask] > 2.
    
    fMASK_D = np.logical_or(
        np.logical_and(fMASK_PD, fMASK_SNR), 
        fMASK_CD
    )
    
    transient_data['REAL'] = fMASK_D
    transient_data['LIMIT'] = fMASK_ND
    #transient_data['Q'] = fMASK_PD
    
    return transient_data

    
    """
    
    # REAL is true whenever the row has an AlertID
    StarCoordIn = SkyCoord(np.array(transient_data['XWIN_WORLD'])*u.deg,np.array(transient_data['YWIN_WORLD'])*u.deg)
    sep = SkyCoord(np.median(transient_data[transient_data["AlertID"].notnull()]['XWIN_WORLD'])*u.deg,np.median(transient_data[transient_data["AlertID"].notnull()]['YWIN_WORLD'])*u.deg).separation(StarCoordIn).to('arcsec').value
    transient_data['REAL'] = pd.DataFrame(sep <= 0.3)
    #transient_data['REAL'] = transient_data['AlertID'].notnull()
    return transient_data
    
    """