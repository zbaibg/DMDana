import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np

import DMDana.lib.constant as const
from DMDana.lib.matana import (DMD_MatrixAnalyzer, JDFTx_MatrixAnalyzer,
                               Plot_mat2D, Plot_mat3D, checkhermitian,
                               get_mat_of_minus_k, mat_init, shift_kvec,
                               validate_kveclist)

if __name__ == '__main__':
    # Set up dataset directory path.
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data')
    assert os.path.exists(dataset_dir), "Data directory not found"
else:
    # Import dataset directory if not main.
    from ..dataset import dataset_dir

def generate_kveclist(kmesh_num):
    """Generate a list of k-points based on kmesh_num."""
    kvecs = np.mgrid[0:kmesh_num[0], 0:kmesh_num[1], 0:kmesh_num[2]].reshape(3, -1).T
    kvecs = kvecs / np.array(kmesh_num) - 0.5
    return kvecs

def get_output_dir(subfolder=''):
    """Get or create the output directory optionally with a subfolder."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if subfolder == '':
        return output_dir
    else:
        subfolder_path = os.path.join(output_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        return subfolder_path

class UtilityFunctionTests(unittest.TestCase):
    """Test utility functions within the matana library."""

    def test_check_hermitian_true(self):
        """Test that a Hermitian matrix is recognized correctly."""
        mat = np.array([[1, 2j], [-2j, 1]])
        self.assertTrue(checkhermitian(mat))

    def test_check_hermitian_false(self):
        """Test that a non-Hermitian matrix is recognized correctly."""
        non_herm_mat = np.array([[1, 2j], [2j, 1]])
        self.assertFalse(checkhermitian(non_herm_mat))

    def test_shift_kvec(self):
        """Test the shift_kvec function to adjust k-point coordinate ranges."""
        kveclist = np.array([[0.9, 0.1, -0.1], [-0.9, -0.1, 0.1]])
        expected = np.array([[-0.1, 0.1, -0.1], [0.1, -0.1, 0.1]])
        shifted_kvecs = shift_kvec(kveclist)
        np.testing.assert_array_almost_equal(shifted_kvecs, expected)

    def test_get_mat_of_minus_k(self):
        """Test the get_mat_of_minus_k function for correct matrix handling."""
        kveclist = np.array([[0, 0, 0], [0.25, 0, 0], [-0.25, 0, 0]])
        kmesh_num = (4, 4, 4)
        mat = np.array([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)])
        expected = np.array([np.eye(2), 3 * np.eye(2), 2 * np.eye(2)])
        minus_k = get_mat_of_minus_k(mat, kmesh_num, kveclist)
        np.testing.assert_array_equal(minus_k, expected)

    def test_validate_kveclist(self):
        """Test the validate_kveclist function to ensure k-point lists are correct."""
        kmesh_num = (2, 2, 2)
        kveclist = generate_kveclist(kmesh_num)
        self.assertTrue(validate_kveclist(kveclist, kmesh_num))
        invalid_kveclist = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.3, 0.3, 0.3]])
        self.assertFalse(validate_kveclist(invalid_kveclist, kmesh_num))

class MatInitTests(unittest.TestCase):
    def setUp(self):
        """Set up test environment for matrix initialization tests."""
        self.output_dir = get_output_dir()
        self.file_path = os.path.join(self.output_dir, 'test_data.bin')
        self.expected_shape = (4, 5)
        num_elements = np.prod(self.expected_shape)
        self.data = (np.random.rand(num_elements) + 1j * np.random.rand(num_elements)).astype(np.complex64)
        self.data.tofile(self.file_path)

    def tearDown(self):
        """Clean up files after tests."""
        os.remove(self.file_path)

    def test_read_complex_matrix_from_file(self):
        """Test reading a complex matrix from a file and ensure it matches the expected shape and data type."""
        result_matrix = mat_init(self.file_path, self.expected_shape, np.complex64)
        self.assertEqual(result_matrix.shape, self.expected_shape)
        self.assertTrue(np.iscomplexobj(result_matrix))
        np.testing.assert_array_equal(result_matrix, self.data.reshape(self.expected_shape))

class JDFTxMatrixAnalyzerTests(unittest.TestCase):
    def setUp(self):
        """Initialize the JDFTx matrix analyzer test setup."""
        self.kmesh_num = [8, 8, 8]
        self.band_number = 10
        self.num_kpoints = np.prod(self.kmesh_num)
        self.DFTfolder = os.path.join(dataset_dir, 'GaAs_DFT_nosym')

    def test_initialization(self):
        """Test initialization of JDFTx Matrix Analyzer."""
        analyzer = JDFTx_MatrixAnalyzer(folder=self.DFTfolder, kmesh_num=self.kmesh_num, nb_dft=self.band_number)
        self.assertEqual(analyzer.vmat_dft.shape, (self.num_kpoints, 3, self.band_number, self.band_number))
        self.assertEqual(len(analyzer.kveclist), self.num_kpoints)

class DMDMatrixAnalyzerTests(unittest.TestCase):
    def setUp(self):
        """Set up DMD matrix analyzer for testing."""
        self.DMDfolder = os.path.join(dataset_dir, 'GaAs_DFT', 'init_NkMult2', 'DMD', 'ExEy0.00050-eph')
        self.band_number = 6
        self.num_kpoints = 15

    def test_initialization(self):
        """Test initialization of DMD Matrix Analyzer."""
        analyzer = DMD_MatrixAnalyzer(self.DMDfolder)
        self.assertEqual(analyzer.vmat.shape, (self.num_kpoints, 3, self.band_number, self.band_number))
        self.assertEqual(len(analyzer.kveclist), self.num_kpoints)

class PlotFunctionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class for plot function tests."""
        cls.output_dir = get_output_dir('plots')

    def test_plot_mat2d_creation(self):
        """Test creation of a 2D plot."""
        file_path = os.path.join(self.output_dir, 'plot_mat2d.png')
        kmesh_num = [10, 10, 10]
        kveclist = generate_kveclist(kmesh_num)
        y_of_b_d_kmi = np.random.random((4, 3, np.prod(kmesh_num)))
        color_of_b_d_kmi = np.random.random((4, 3, np.prod(kmesh_num)))
        kpaths = [np.array([[0, 0, 0], [0.5, 0.5, 0.5]])]
        labels = [["Start", "End"]]
        title='Test 2D Plot'
        ylabel='Y-axis'

        plot = Plot_mat2D(kveclist, y_of_b_d_kmi, color_of_b_d_kmi, kpaths, labels, 0.05, title=title, ylabel=ylabel)
        plot.savefig(file_path)
        plot.close()

        self.assertTrue(os.path.exists(file_path))

    def test_plot_mat2d_creation_as_whole(self):
        """Test creation of a 2D plot as a whole figure."""
        file_path = os.path.join(self.output_dir, 'plot_mat2d_whole.png')
        kmesh_num = [10, 10, 10]
        kveclist = generate_kveclist(kmesh_num)
        y_of_b_d_kmi = np.random.random((4, 3, np.prod(kmesh_num)))
        color_of_b_d_kmi = np.random.random((4, 3, np.prod(kmesh_num)))
        kpaths = [np.array([[0, 0, 0], [0.5, 0.5, 0.5]])]
        labels = [["Start", "End"]]
        title='Test 2D Plot'
        ylabel='Y-axis'

        plot = Plot_mat2D(kveclist, y_of_b_d_kmi, color_of_b_d_kmi, kpaths, labels, 0.05, separate=False, title=title, ylabel=ylabel)
        plot.savefig(file_path)
        plot.close()

        self.assertTrue(os.path.exists(file_path))
    
    def test_plot_mat3d_creation(self):
        """Test creation of a 3D plot."""
        file_path = os.path.join(self.output_dir, 'plot_mat3d.png')
        kveclist = generate_kveclist([5, 5, 5])
        color_b_k = np.random.rand(4, len(kveclist))

        plot = Plot_mat3D(kveclist, color_b_k)
        plot.savefig(file_path)
        plot.close()

        self.assertTrue(os.path.exists(file_path))
    
class RealDataPlotTests(unittest.TestCase):
    """Test plotting from real data for 2D and 3D graphs."""

    def setUp(self):
        """Set up for real data plot tests."""
        self.output_dir = get_output_dir('plots')
        self.DMDfolder = os.path.join(dataset_dir, 'GaAs_DFT', 'init_NkMult2', 'DMD', 'ExEy0.00050-eph')
        self.analyzer = DMD_MatrixAnalyzer(self.DMDfolder)

    def test_2d_plot_from_real_data(self):
        """Test creation of a 2D plot using real data from a DMD Matrix Analyzer."""
        y_of_b_d_kmi = np.real(np.einsum('kdbb->bdk', self.analyzer.vmat))  # Diagonal
        color_of_b_d_kmi = np.tile(self.analyzer.Emat.T[:,None,:], (1, 3, 1)).real * const.Hatree_to_eV
        kpaths = [np.array([[0, 0, 0], [0.02, 0.02, 0.02]])]
        labels = [["Gamma", "X"]]

        plot = Plot_mat2D(self.analyzer.kveclist, y_of_b_d_kmi, color_of_b_d_kmi, kpaths, labels, 0.02, title='Diagonal momentum matrix', ylabel='Energy (eV)')
        plot_path = os.path.join(self.output_dir, 'real_data_2d_plot.png')
        plot.savefig(plot_path)
        self.assertTrue(os.path.exists(plot_path))

    def test_3d_plot_from_real_data(self):
        """Test creation of a 3D plot using real data."""
        color_b_k = np.real(np.einsum('kdbb->bdk', self.analyzer.vmat)[:,0,:])  # Diagonal of x

        plot = Plot_mat3D(self.analyzer.kveclist, color_b_k)
        plot_path = os.path.join(self.output_dir, 'real_data_3d_plot.png')
        plot.savefig(plot_path)
        self.assertTrue(os.path.exists(plot_path))

if __name__ == '__main__':
    unittest.main()