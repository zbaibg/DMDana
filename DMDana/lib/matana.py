from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

from DMDana.lib import DMDparser

from .constant import *


class JDFTx_MatrixAnalyzer:
    """
    Class for analyzing JDFTx matrix data related to electronic structure calculations.

    Attributes:
        folder (str): The folder path containing the data files.
        kmesh_num (Tuple[int, int, int]): The number of k-points in each direction (knx, kny, knz).
        nb_dft (int): The number of DFT bands.
        vmat_dft (np.ndarray): Matrix of DFT data loaded from files.
        kveclist (np.ndarray): List of k-vectors associated with the matrix.
        vmat_minus_k (np.ndarray): Matrix of -k corresponding to the given k-points.
    """

    def __init__(self, folder: str, kmesh_num: Tuple[int, int, int], nb_dft: int):
        """
        Initializes the JDFTx_MatrixAnalyzer object.

        Parameters:
            folder (str): The folder path containing the data files.
            kmesh_num (Tuple[int, int, int]): The number of k-points in each direction (knx, kny, knz).
            nb_dft (int): The number of DFT bands.
        """
        self.folder = folder
        self.kmesh_num = kmesh_num  # (knx, kny, knz)
        self.nb_dft = nb_dft
        self.vmat_dft: np.ndarray
        self.kveclist: np.ndarray
        self.vmat_minus_k: np.ndarray

        self.load_data()
        self.vmat_minus_k = get_mat_of_minus_k(self.vmat_dft, self.kmesh_num, self.kveclist)

    def load_data(self) -> None:
        """
        Loads the DFT matrix data and k-vectors from specified files within the given folder.
        """
        vmat_path = f'{self.folder}/totalE.momenta'
        kvec_path = f'{self.folder}/totalE.kPts'

        self.vmat_dft = mat_init(vmat_path, (np.prod(self.kmesh_num), 3, self.nb_dft, self.nb_dft), np.complex_)
        self.kveclist = np.loadtxt(kvec_path, usecols=(2, 3, 4))
        self.kveclist = shift_kvec(self.kveclist)

        assert validate_kveclist(self.kveclist, self.kmesh_num), "Validation of kveclist failed."


class DMD_MatrixAnalyzer:
    """
    Class for analyzing DMD (Density Matrix Decomposition) matrix data from electronic structure computations.

    Attributes:
        DMDfolder (str): Path to the folder containing the DMD data files.
        kmesh (np.ndarray): Number of k-points in each direction (x, y, z).
        nb (int): Number of bands.
        bBot_dm (int): Lower band index for DMD.
        bTop_dm (int): Upper band index for DMD.
        nb_dm (int): Number of DMD bands.
        nk (int): Number of k-points.
        kstep (np.ndarray): Step size between k-points in each direction.
        denmat (np.ndarray): Density matrix from DMD calculations.
        vmat (np.ndarray): Potential matrix from DMD calculations.
        Emat (np.ndarray): Energy matrix from DMD calculations.
        kveclist (np.ndarray): List of k-vectors.
    """

    def __init__(self, DMDfolder: str):
        """
        Initializes the DMD_MatrixAnalyzer object.

        Parameters:
            DMDfolder (str): The folder path containing the DMD data files.
        """
        self.DMDfolder = DMDfolder
        self.kmesh = np.zeros(3, dtype=int)
        self.load_basic_data()

    def load_basic_data(self) -> None:
        """
        Loads all the necessary matrix data and configurations from the DMD files.
        """
        # Define paths to the data files
        paths: Dict[str, str] = {
            'vmat': f'{self.DMDfolder}/ldbd_data/ldbd_vmat.bin',
            'denmat': f'{self.DMDfolder}/restart/denmat_restart.bin',
            'Emat': f'{self.DMDfolder}/ldbd_data/ldbd_ek.bin',
            'size_data': f'{self.DMDfolder}/ldbd_data/ldbd_size.dat',
            'kvec': f'{self.DMDfolder}/ldbd_data/ldbd_kvec.bin'
        }
        
        # Read size data from file and store it
        self.nb, self.bBot_dm, self.bTop_dm, self.nk, \
            self.kmesh[0], self.kmesh[1], self.kmesh[2] = DMDparser.read_text_from_file(
                paths['size_data'],
                ["# nb nv bBot_dm"] * 3 + ["# nk_full"] * 4,
                [0, 2, 3, 1, 2, 3, 4],
                True,
                [int] * 7
            )
        
        self.nb_dm = self.bTop_dm - self.bBot_dm
        self.kstep = 1 / self.kmesh
        self.denmat = mat_init(paths['denmat'], (self.nk, self.nb_dm, self.nb_dm), np.complex_)
        self.vmat = mat_init(paths['vmat'], (self.nk, 3, self.nb_dm, self.nb), np.complex_)[:, :, :, self.bBot_dm:self.bTop_dm]
        self.Emat = mat_init(paths['Emat'], (self.nk, self.nb), float)[:, self.bBot_dm:self.bTop_dm]
        self.kveclist = mat_init(paths['kvec'], (self.nk, 3), float)
        self.kveclist = shift_kvec(self.kveclist)

        # Validate data integrity and properties
        assert validate_kveclist(self.kveclist, tuple(self.kmesh)), "K-vector list validation failed."
        assert checkhermitian(self.denmat), 'Density matrix is not Hermitian.'
        assert checkhermitian(self.vmat), 'Potential matrix is not Hermitian.'
        
        
def checkhermitian(mat: np.ndarray) -> bool:
    """
    Checks if a matrix is Hermitian along its last two dimensions.

    Parameters:
    - mat (numpy.ndarray): The matrix to be checked, which should be at least two-dimensional.

    Returns:
    - bool: Returns True if the matrix is Hermitian, False otherwise. A matrix is Hermitian
            if it is equal to its complex conjugate transpose.
    """
    # Reorder the last two dimensions of the matrix to prepare for transpose
    axessequence = np.arange(mat.ndim)
    axessequence[-2:] = axessequence[-2:][::-1]
    
    # Check if the original matrix is close to its conjugate transpose within numerical tolerance
    return np.isclose(np.transpose(mat, axes=axessequence).conj(), mat, atol=1e-13, rtol=0).all()


def shift_kvec(kveclist: np.ndarray) -> np.ndarray:
    """
    Shifts k-vectors to ensure all components are within the range [-0.5, 0.5).

    Parameters:
    - kveclist (numpy.ndarray): The array of k-vectors.

    Returns:
    - numpy.ndarray: The shifted array of k-vectors.
    """
    # Shift each k-vector component to be within the specified range
    return (kveclist + 0.5) % 1 - 0.5


def get_mat_of_minus_k(mat: np.ndarray, kmesh_num: Tuple[int, int, int], kveclist: np.ndarray) -> np.ndarray:
    """
    Constructs a matrix corresponding to the negative k-vectors from a given matrix.

    Parameters:
    - mat (np.ndarray): The input matrix whose rows/columns correspond to k-points. Expected to be a complex matrix.
    - kmesh_num (Tuple[int, int, int]): The number of k-points in each direction of the k-space mesh (knx, kny, knz).
    - kveclist (np.ndarray): The list of k-vectors associated with the matrix rows/columns. Expected to contain float entries.

    Returns:
    - np.ndarray: A new matrix where each entry is mapped to the corresponding -k vector from the input matrix.
    """
    # Shift k-vectors to standard range and calculate mesh indices
    kveclist_shifted = shift_kvec(kveclist)
    kmeshlist = np.array(np.round(kveclist_shifted * kmesh_num), dtype=int)
    
    # Initialize mapping from mesh indices to matrix indices
    kmesh_to_knum = np.full(kmesh_num, None)
    for knum_tmp, idx in enumerate(kmeshlist):
        kmesh_to_knum[tuple(idx)] = knum_tmp

    # Map each k-point to its corresponding -k-point in the matrix
    mat_minus_k = np.empty(mat.shape, dtype=mat.dtype)
    for knum_tmp in range(len(kmeshlist)):
        idx = tuple(-kmeshlist[knum_tmp] % kmesh_num)  # Negative index adjusted by mesh size
        minusknum_tmp = kmesh_to_knum[idx]
        mat_minus_k[minusknum_tmp] = mat[knum_tmp]

    return mat_minus_k

def plot_hist(data: np.ndarray, 
              title: str, 
              ylabel: str, 
              xlabel: str, 
              logbin: int, 
              density: bool, 
              scale: str, 
              dpi: int) -> None:
    """
    Plots a histogram of the provided data with extensive customization for display properties.

    Parameters:
    - data (numpy.ndarray): Data to be plotted in the histogram.
    - title (str): Title of the histogram plot.
    - ylabel (str): Label for the Y-axis.
    - xlabel (str): Label for the X-axis.
    - logbin (int): Number of logarithmic bins in the histogram.
    - density (bool): If True, normalize the histogram; otherwise, count frequencies.
    - scale (str): Scale of the plot axes ('linear', 'log', etc.).
    - dpi (int): Resolution of the plot in dots per inch.
    """
    plt.figure(dpi=dpi)
    hist, bins = np.histogram(data, bins=logbin, density=density)
    plt.plot(bins[1:] * 0.5, hist)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def mat_init(path: str, shape: Tuple[int, ...], dtype: Type) -> np.ndarray:
    """
    Initializes a matrix from a binary file.

    Parameters:
    - path (str): File path to the binary data.
    - shape (Tuple[int, ...]): Expected shape of the matrix. Tuple elements represent dimensions sizes.
    - dtype (dtype): Data type of the matrix elements, e.g., np.int32, np.float64.

    Returns:
    - numpy.ndarray: The matrix loaded from the specified binary file.

    Raises:
    - AssertionError: If the number of elements read from the file does not match the expected product of the shape dimensions.
    """
    # Read the binary file content into a numpy array
    mat_raw = np.fromfile(path, dtype=dtype)
    assert len(mat_raw) == np.prod(shape), f"File {path} size does not match expected size {shape}"

    # Reshape the raw data to the specified shape
    mat = mat_raw.reshape(shape, order='C')
    return mat

def validate_kveclist(kveclist: np.ndarray, kmesh_num: Tuple[int, int, int]) -> bool:
    """
    Validates that all k-points in the kveclist are within the specified kmesh grid and their opposite points are also present.

    Parameters:
    - kveclist (numpy.ndarray): The list of k-vectors to validate.
    - kmesh_num (Tuple[int, int, int]): The dimensions of the k-mesh grid (knx, kny, knz), specifying the number of k-points in each direction.

    Returns:
    - bool: True if all k-points and their opposites are valid within the kmesh, False otherwise.
    """
    # Create a grid of k-points for the kmesh
    kmesh_points = np.mgrid[0:kmesh_num[0], 0:kmesh_num[1], 0:kmesh_num[2]].reshape(3, -1).T / np.array(kmesh_num) - 0.5
    
    # Check if every k-vector in kveclist is close to any point in the kmesh_points
    k_in_mesh = np.array([np.any(np.all(np.isclose(kvec, kmesh_points, rtol=1.e-5, atol=0), axis=1)) for kvec in kveclist])
    
    # Check if the opposite of every k-vector in kveclist is close to any other k-vector in kveclist
    k_in_mesh_opposite = np.array([np.any(np.all(np.isclose(shift_kvec(-kvec), kveclist, rtol=1.e-5, atol=0), axis=1)) for kvec in kveclist])

    return np.all(k_in_mesh) and np.all(k_in_mesh_opposite)


class Plot_mat:
    """
    A base class for creating and managing plots for matrix data, particularly for visualizing 
    k-space vectors or similar scientific data.

    Attributes:
        kveclist (np.ndarray): An array of k-vectors, typically representing points in a space.
        fig (matplotlib.figure.Figure): The figure object where the plot(s) will be drawn.
        title (str): Title of the overall figure.
        ylabel (str): Label for the Y-axis of the plot.
        dpi (int): Dots per inch, setting the resolution of the figure.
        figsize (Tuple[int, int]): Dimensions of the figure (width, height) in inches.
    """

    def __init__(self, kveclist: Union[np.ndarray, list], dpi: int = 100, 
                 figsize: Tuple[int, int] = (8, 6), title: str = '', ylabel: str = ''):
        """
        Initializes the Plot_mat object with specified parameters and sets up the plotting environment.
        
        Parameters:
            kveclist (Union[np.ndarray, list]): List or array of k-vectors. If a list is provided,
                                                it will be converted to an np.ndarray.
            dpi (int, optional): Resolution of the plot in dots per inch. Default is 100.
            figsize (Tuple[int, int], optional): Size of the figure (width, height) in inches.
                                                 Default is (8, 6).
            title (str, optional): Title of the overall figure. Default is an empty string.
            ylabel (str, optional): Label for the Y-axis of the plot. Default is an empty string.
        """
        self.kveclist = np.array(kveclist)
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        self.title = title
        self.ylabel = ylabel

    def postprocess(self, colorbartmp: plt.cm.ScalarMappable) -> None:
        """
        Adds a color bar to the plot and sets titles and labels.

        Parameters:
            colorbartmp (matplotlib.cm.ScalarMappable): The ScalarMappable instance from which the colorbar is created.
        """
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values as necessary to fit layout
        self.fig.colorbar(colorbartmp, cax=cbar_ax)
        self.fig.suptitle(self.title)
        if self.ylabel:  # Ensuring ylabel is not empty
            plt.ylabel(self.ylabel)
    def show(self):
        """
        Displays the figure.
        """
        plt.show()

    def savefig(self, path: str):
        """
        Saves the figure to a file.

        Parameters:
            path (str): The path (and filename) to save the figure to.
        """
        self.fig.savefig(path)

    def close(self):
        """
        Closes the figure, freeing up resources.
        """
        plt.close(self.fig)

class Plot_mat2D(Plot_mat):
    """
    Class for visualizing 2D matrices representing data along specified paths in k-space.
    It visualizes data points across multiple bands and directions using color gradients,
    ideal for examining electronic structures in materials science and physics.

    Inherits:
        Plot_mat: A base plotting class that provides a figure and common functionalities.

    Attributes:
        data (np.ndarray): The y-values for the plot, indexed by bands, directions, and k-points.
                           Shape: (num_bands, num_directions, num_kpoints).
        colors (np.ndarray): The colors corresponding to the data values, same shape as data.
        kpaths (List[np.ndarray]): List of arrays, each representing a path through k-space.
                                   Each array in the list is of shape (M, 3), where M is the number of points in the path.
        labels (List[List[str]]): List of lists, each containing labels for the points in corresponding kpaths.
        separate (bool, optional): If True, plots each band-direction combination in a separate subplot.
                                   Default is True.
        gap (float, optional): Gap between consecutive k-paths on the x-axis. Default is 0.1.
        tolerance (float): Numerical tolerance used for matching k-points to k-paths.
        title (str, optional): Title for the entire figure. Default is an empty string.
        ylabel (str, optional): Label for the Y-axis of the plot. Default is an empty string.
        vmin (float, optional): Minimum value for color scaling. Default is the minimum value in the data.
        vmax (float, optional): Maximum value for color scaling. Default is the maximum value in the data.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 100.
        figsize (Tuple[int, int], optional): Size of the figure (width, height) in inches. Default is (8, 6).
    """

    def __init__(self, kveclist: Union[np.ndarray, List[List[float]]], data: np.ndarray, colors: np.ndarray,
                 kpaths: List[np.ndarray], labels: List[List[str]], tolerance: float,
                 separate: bool = True, title: str = '', ylabel: str = '', gap: float = 0.1,
                 vmin: Union[None, float] = None, vmax: Union[None, float] = None,
                 dpi: int = 100, figsize: Tuple[int, int] = (12, 8)):
        """
        Initializes the Plot_mat2D object with specified parameters and setups the plotting environment.

        Parameters:
            kveclist (Union[np.ndarray, List[List[float]]]): List or array of k-vectors. Shpe: (num_kpoints, 3).
            data (np.ndarray): The y-values for the plot, indexed by bands, directions, and k-points. Shape: (num_bands, num_directions, num_kpoints).
            colors (np.ndarray): The colors corresponding to the data values, same shape as data. Shape: (num_bands, num_directions, num_kpoints).
            kpaths (List[np.ndarray]): List of arrays, each representing a path through k-space. Shape: (num_paths, M, 3).
                                       Each array in the list is of shape (M, 3), where M is the number of points in the path.
            labels (List[List[str]]): List of lists, each containing labels for the points in corresponding kpaths. Shape: (num_paths, M).
            tolerance (float): Numerical tolerance used for matching k-points to k-paths.
            separate (bool, optional): If True, plots each band-direction combination in a separate subplot.
                                       Default is True.
            title (str, optional): Title for the entire figure. Default is an empty string.
            ylabel (str, optional): Label for the Y-axis of the plot. Default is an empty string.
            gap (float, optional): Gap between consecutive k-paths on the x-axis. Default is 0.1.
            vmin (Union[None, float], optional): Minimum value for color scaling. Default is the minimum value in the data.
            vmax (Union[None, float], optional): Maximum value for color scaling. Default is the maximum value in the data.
            dpi (int, optional): Resolution of the plot in dots per inch. Default is 100.
            figsize (Tuple[int, int], optional): Size of the figure (width, height) in inches. Default is (8, 6).
        """
        super().__init__(kveclist, dpi=dpi, figsize=figsize, title=title, ylabel=ylabel)
        self.data = np.array(data)
        self.colors = np.array(colors)
        self.kpaths = kpaths
        self.labels = labels
        self.separate = separate
        self.gap = gap
        self.tolerance = tolerance
        self.vmin = vmin if vmin is not None else self.data.min()
        self.vmax = vmax if vmax is not None else self.data.max()

        self.num_bands, self.num_directions, _ = data.shape
        self.setup_figure()
        
    def setup_figure(self):
        """
        Set up the figure with subplots configured for separate bands and directions if specified.
        """
        self.axlist = []
        if self.separate:
            for b in range(self.num_bands):
                for d in range(self.num_directions):
                    ax = self.fig.add_subplot(self.num_bands, self.num_directions, b * self.num_directions + d + 1)
                    self.axlist.append(ax)
        else:
            ax = self.fig.add_subplot(111)
            self.axlist = [ax] * (self.num_bands * self.num_directions)

        self.plot_data()

    def plot_data(self):
        """
        Plots the data on the prepared subplots.
        """
        colorbartmp = None  # To capture the last colorbar mappable for postprocess
        for index, ax in enumerate(self.axlist):
            b, d = divmod(index, self.num_directions)
            for path, x_positions in zip(self.kpaths, self.process_paths()):
                k_indices, projections = self.match_kpoints_to_path(path)
                for k_index, projection, x_pos in zip(k_indices, projections,x_positions):
                    colorbartmp = ax.scatter(x_pos + projection, self.data[b, d, k_index], c=self.colors[b, d, k_index], vmin=self.vmin, vmax=self.vmax, cmap='viridis')

        if colorbartmp:
            self.postprocess(colorbartmp)  # Use the inherited postprocess method


    def postprocess(self, colorbartmp: plt.cm.ScalarMappable):
        """
        Adds a color bar to the plot, adjust xlims, and sets titles and labels.

        Parameters:
            colorbartmp (matplotlib.cm.ScalarMappable): The ScalarMappable instance from which the colorbar is created.
        """
        if self.separate:
            for index, ax in enumerate(self.axlist):
                b, d = divmod(index, self.num_directions)
                ax.set_xticks([x for x_of_path in self.process_paths() for x in x_of_path], [])
                ax.set_xlim(0, self.process_paths()[-1][-1])
                
                if b == self.num_bands - 1:
                    ax.set_xlabel('k-path')
                    if self.labels:
                        ax.set_xticks([x for x_of_path in self.process_paths() for x in x_of_path], [label for label_of_path in self.labels for label in label_of_path])
                if b == 0:
                    ax.set_title(f'Direction {['x', 'y', 'z'][d]}')
            self.fig.supylabel(self.ylabel)
        else:
            self.axlist[0].set_xticks([x for x_of_path in self.process_paths() for x in x_of_path], [])
            self.axlist[0].set_xlim(0, self.process_paths()[-1][-1])
            self.axlist[0].set_xlabel('k-path')
            if self.labels:
                self.axlist[0].set_xticks([x for x_of_path in self.process_paths() for x in x_of_path], [label for label_of_path in self.labels for label in label_of_path])
            self.axlist[0].set_ylabel(self.ylabel)
            self.axlist[0].set_title(self.title)
        super().postprocess(colorbartmp)

    def process_paths(self):
        """
        Calculate x-axis positions for plotting based on the specified k-paths.
        """
        x_positions = []
        x_offset = 0
        for path in self.kpaths:
            distances = LA.norm(np.diff(path, axis=0), axis=1)
            distances = np.insert(distances, 0, 0)
            cumulative_distances = np.cumsum(distances) + x_offset
            x_positions.append(cumulative_distances)
            x_offset = cumulative_distances[-1] + self.gap
        return x_positions

    def match_kpoints_to_path(self, path: np.ndarray):
        """
        Finds k-points from kveclist that are close to the given k-path.
        Returns indices of matched k-points and their projections along the path.
        """
        k_indices = []
        projections = []
        for start, end in zip(path[:-1], path[1:]):
            direction = end - start
            projected_distances = np.dot((self.kveclist - start), direction) / np.linalg.norm(direction)
            perpendicular_distances = np.linalg.norm(np.cross(self.kveclist - start, direction / np.linalg.norm(direction)), axis=1)
            mask = (projected_distances >= 0) & (projected_distances <= np.linalg.norm(direction)) & (perpendicular_distances <= self.tolerance)
            k_indices.append(np.where(mask)[0])
            projections.append(projected_distances[mask])
        return k_indices, projections



class Plot_mat3D(Plot_mat):
    """
    A class for visualizing 3D matrices representing data along specified paths in k-space.
    It visualizes data points across multiple bands and directions using color gradients.

    Inherits:
        Plot_mat: A base plotting class that provides figure setup and post-processing.

    Attributes:
        kveclist (np.ndarray): List of all k-vectors of Kmesh to be analyzed. Shape: (total number of k-points, number of dimensions).
        color_b_k (np.ndarray): Colors corresponding to the data points. Shape: (total number of k-points).
        vmin (Optional[float]): Minimum color scale value. Defaults to None.
        vmax (Optional[float]): Maximum color scale value. Defaults to None.
        title (str): Title of the plot. Defaults to an empty string.
        rownum (int): Number of rows in the subplot grid. Defaults to 1.
        index_str_for_subtitle (str): String for subplot subtitles. Defaults to 'i'.
        subtitle_on (bool): Whether to display subtitles. Defaults to True.
        dpi (int): Dots per inch for plot resolution. Defaults to 100.
        figsize (Tuple[int, int]): Dimensions of the figure (width, height) in inches. Defaults to (12, 6).
    """

    def __init__(self, kveclist: np.ndarray, color_b_k: np.ndarray, 
                 vmin: Optional[float] = None, vmax: Optional[float] = None, 
                 title: str = '', rownum: int = 1, index_str_for_subtitle: str = 'i', 
                 subtitle_on: bool = True, dpi: int = 100, 
                 figsize: Tuple[int, int] = (12, 6)):
        """
        Initializes the Plot_mat3D object with specified parameters and sets up the plotting environment.

        Parameters:
            kveclist (np.ndarray): List or array of k-vectors.
            color_b_k (np.ndarray): Colors corresponding to the data points.
            vmin (Optional[float]): Minimum color scale value. If None, defaults to the minimum value in the data.
            vmax (Optional[float]): Maximum color scale value. If None, defaults to the maximum value in the data.
            title (str): Title of the plot. Defaults to an empty string if not provided.
            rownum (int): Number of rows in the subplot grid. Defaults to 1.
            index_str_for_subtitle (str): String for subplot subtitles. Defaults to 'i'.
            subtitle_on (bool): Whether to display subtitles. Defaults to True.
            dpi (int): Dots per inch for plot resolution. Defaults to 100.
            figsize (Tuple[int, int]): Dimensions of the figure (width, height) in inches. Defaults to (12, 6).
        """
        super().__init__(kveclist=kveclist, dpi=dpi, figsize=figsize, title=title)
        self.color_b_k = np.array(color_b_k)
        self.vmin = vmin
        self.vmax = vmax
        self.rownum = rownum
        self.index_str_for_subtitle = index_str_for_subtitle
        self.subtitle_on = subtitle_on

        # Verify rownum is valid
        b_num = len(color_b_k)
        assert b_num >= rownum, 'rownum should be equal to or smaller than the number of figures'

        # Calculate subplot layout
        columnnum = b_num // rownum + 1 if b_num % rownum != 0 else b_num // rownum

        # Create subplots
        axlist = [self.fig.add_subplot(rownum, columnnum, i, projection='3d') for i in range(1, b_num + 1)]

        # Determine color scale range
        colormax = np.max(np.abs(color_b_k)) if vmax is None else vmax
        colormin = -np.max(np.abs(color_b_k)) if vmin is None else vmin

        # Plot data
        for b, color_k in enumerate(color_b_k):
            ax = axlist[b]
            colorbartmp = ax.scatter(kveclist[:, 0], kveclist[:, 1], kveclist[:, 2], c=color_k, s=5, cmap='rainbow', vmin=colormin, vmax=colormax)

        # Post-process subplots
        for ax_i, ax in enumerate(axlist):
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('kz')
            if self.subtitle_on:
                ax.set_title('%s %d' % (self.index_str_for_subtitle, ax_i))

        super().postprocess(colorbartmp)
