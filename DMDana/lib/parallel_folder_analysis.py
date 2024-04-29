import os
import logging
import pandas as pd
from DMDana.lib.DMDparser import DMD
from multiprocessing import Pool
from typing import List, Dict, Callable

def check_and_create_folder(path: str) -> str:
    """Create a folder if it doesn't exist and return the folder path."""
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def save_database(path: str, df: pd.DataFrame) -> None:
    """Save a pandas DataFrame to an Excel file."""
    df.to_excel(path, index=False)

class FolderAnalysis:
    """A class to perform analysis on DMD folders using dynamically provided functions."""
    
    def __init__(self, DMD_instance: DMD, index: int, analysis_functions: List[Callable], file_prefix: str):
        """Initialize the FolderAnalysis with a DMD instance, an index, analysis functions, and file prefix."""
        self.index = index
        self.DMD_instance = DMD_instance
        self.analysis_functions = analysis_functions
        self.file_prefix=file_prefix
        self.current_folder = os.getcwd()
        self.df = pd.DataFrame()
        self.df.loc[self.index, 'folder'] = self.DMD_instance.DMD_folder
        self.subfolder = check_and_create_folder(os.path.join(self.current_folder, f"{self.index}"))
        os.chdir(self.subfolder)
        self.general_logger = logging.getLogger(f"FolderAnalysis_{self.index}")
        logging.basicConfig(
            level=logging.INFO,
            filename=self.current_folder+'/%d/%s_folder_%d.log'%(index,file_prefix,index),
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            filemode='w',force=True)
        self.general_logger.info('Analysis started')

    def get_function_logger(self, function_name: str) -> logging.Logger:
        """Create a function-specific logger."""
        logger = logging.getLogger(f"FolderAnalysis_{self.index}_{function_name}")
        return logger

    def run_analysis(self) -> None:
        """Run all analysis functions provided during initialization."""
        for function in self.analysis_functions:
            function_name= function.__name__
            func_logger = self.get_function_logger(function_name)
            func_logger.info(f"Running {function_name}")
            try:
                function(self, func_logger)
            except Exception as e:
                func_logger.error(f"Error in {function_name}")
                func_logger.exception(e)
                self.df.loc[self.index, f'{function_name}_status'] = 'Failed'
                continue
            self.df.loc[self.index, f'{function_name}_status'] = 'Succeeded'

    def finalize_and_return(self) -> pd.DataFrame:
        """Finalize analysis by saving results and returning the DataFrame."""
        os.chdir(self.current_folder)
        status = 'Succeeded' if 'Failed' not in self.df.loc[self.index].values else 'Failed but ran to the end'
        self.df.loc[self.index, 'organize_status'] = status
        self.general_logger.info(f"Analysis finished with status: {status}")
        save_database(f'./{self.index}/'+self.file_prefix+f'_database_out_{self.index}.xlsx', self.df)
        return self.df

DF_FILE_PATH_OUT = 'database_out.xlsx'
FOLDERS_FILE = 'folders'

def read_folders(file_path: str) -> List[str]:
    """Read a file containing folder names and return a list of folders."""
    with open(file_path, 'r') as file:
        folders = [line.strip() for line in file if line.strip()]
    return folders

def parallel_folder_analysis(analysis_functions: List[Callable], core_num: int = 1, index_shift: int = 0,file_prefix:str='default') -> pd.DataFrame:
    """Execute FolderAnalysis class instances in parallel."""
    folders = read_folders(FOLDERS_FILE)
    DMD_list = [DMD(folder) for folder in folders]
    file_prefix=file_prefix
    # Use multiprocessing for parallel processing
    with Pool(core_num) as pool:
        results = pool.starmap(
            subfunc,
            [(DMD_instance, i + index_shift, analysis_functions,file_prefix) for i, DMD_instance in enumerate(DMD_list)]
        )
    
    # Combine results and save to an Excel file
    df = pd.concat(results).sort_index()
    save_database(os.path.join(file_prefix+'_'+DF_FILE_PATH_OUT), df)
    return df

def subfunc(DMD_instance: DMD, index_after_shift: int, analysis_functions: List[Callable],file_prefix:str) -> pd.DataFrame:
    """Subfunction for parallel execution; each subprocess will invoke this function."""
    analysis_instance = FolderAnalysis(DMD_instance, index_after_shift, analysis_functions,file_prefix=file_prefix)
    analysis_instance.run_analysis()
    return analysis_instance.finalize_and_return()
