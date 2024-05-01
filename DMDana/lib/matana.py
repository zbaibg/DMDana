from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from .constant import *
from DMDana.lib import DMDparser
class JDFTx_MatrixAnalyzer:
    def __init__(self, folder, kmesh_num, nb_dft):
        self.folder = folder
        self.kmesh_num = kmesh_num  # (knx, kny, knz)
        self.nb_dft = nb_dft
        self.load_data()
        self.vmat_minus_k = get_mat_of_minus_k(self.vmat_dft, self.kmesh_num, self.kveclist)

    def load_data(self):
        vmat_path = f'{self.folder}/totalE.momenta'
        kvec_path = f'{self.folder}/totalE.kPts'
        self.vmat_dft = mat_init(vmat_path, (np.prod(self.kmesh_num), 3, self.nb_dft, self.nb_dft), complex)
        self.kveclist = np.loadtxt(kvec_path, usecols=(2, 3, 4))
        self.kveclist = shift_kvec(self.kveclist)
        assert validate_kveclist(self.kveclist, self.kmesh_num)

class DMD_MatrixAnalyzer:
    def __init__(self, DMDfolder):
        self.DMDfolder = DMDfolder
        self.kmesh = np.zeros(3, dtype=int)
        self.load_basic_data()

    def load_basic_data(self):
        paths = {
            'vmat': f'{self.DMDfolder}/ldbd_data/ldbd_vmat.bin',
            'denmat': f'{self.DMDfolder}/restart/denmat_restart.bin',
            'Emat': f'{self.DMDfolder}/ldbd_data/ldbd_ek.bin',
            'size_data': f'{self.DMDfolder}/ldbd_data/ldbd_size.dat',
            'kvec': f'{self.DMDfolder}/ldbd_data/ldbd_kvec.bin'
        }
        self.nb, self.bBot_dm, self.bTop_dm, self.nk, self.kmesh[0], self.kmesh[1], self.kmesh[2] = DMDparser.read_text_from_file(
            paths['size_data'],
            ["# nb nv bBot_dm"] * 3 + ["# nk_full"] * 4,
            [0, 2, 3, 1, 2, 3, 4],
            True,
            [int] * 7)
        self.nb_dm = self.bTop_dm - self.bBot_dm
        self.kstep = 1 / self.kmesh
        self.denmat = mat_init(paths['denmat'], (self.nk, self.nb_dm, self.nb_dm), complex)
        self.vmat = mat_init(paths['vmat'], (self.nk, 3, self.nb_dm, self.nb), complex)[:, :, :, self.bBot_dm:self.bTop_dm]
        self.Emat = mat_init(paths['Emat'], (self.nk, self.nb), float)[:, self.bBot_dm:self.bTop_dm]
        self.kveclist = mat_init(paths['kvec'], (self.nk, 3), float)
        self.kveclist = shift_kvec(self.kveclist)
        assert validate_kveclist(self.kveclist, self.kmesh)
        assert checkhermitian(self.denmat), 'denmat is not hermitian'
        assert checkhermitian(self.vmat), 'vmat is not hermitian'

def checkhermitian(mat):
    """Checks if the last two axes of the matrix are hermitian"""
    axessequence = np.arange(mat.ndim)
    axessequence[-2:] = axessequence[-2:][::-1]
    return np.isclose(np.transpose(mat, axes=axessequence).conj(), mat, atol=1e-13, rtol=0).all()
    
def shift_kvec(kveclist):
    """Shifts k-vectors to the range [-0.5, 0.5)"""
    kveclist = (kveclist + 0.5) % 1 - 0.5
    return kveclist

def get_mat_of_minus_k(mat, kmesh_num, kveclist):
    """Returns a matrix of -k corresponding to given k-points"""
    kveclist = shift_kvec(kveclist)
    kmeshlist = np.array(np.round(kveclist * kmesh_num), dtype=int)
    kmesh_to_knum = np.full(kmesh_num, None)
    knum = len(kmeshlist)
    for knum_tmp, (i1, i2, i3) in enumerate(kmeshlist):
        kmesh_to_knum[i1, i2, i3] = knum_tmp

    mat_minus_k = np.full(mat.shape, None)
    for knum_tmp in range(knum):
        minusknum_tmp = kmesh_to_knum[-kmeshlist[knum_tmp, 0], -kmeshlist[knum_tmp, 1], -kmeshlist[knum_tmp, 2]]
        mat_minus_k[minusknum_tmp] = mat[knum_tmp]
    assert (mat_minus_k != None).all(), 'mat_minus_k is calculated correctly, maybe some kpoints of the DMD k-list do not have minus-kpoints in the list'
    return mat_minus_k

def plot_hist(data, title, ylabel, xlabel, logbin, density, scale, dpi):
    """Plots histogram"""
    plt.figure(dpi=dpi)
    hist, bins = np.histogram(data, bins=logbin, density=density)
    plt.plot(bins[1:] * 0.5, hist)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.xticks(logbin)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
        
def mat_init(path, shape, dtype):
    """Initializes matrix from a binary file"""
    mat_raw = np.fromfile(path, dtype=dtype)
    assert len(mat_raw) == np.prod(shape), f'{path} size does not match expected size'
    mat = mat_raw.reshape(shape, order='C')
    return mat

def validate_kveclist(kveclist, kmesh_num):
    """
    Checks if all k-points in kveclist are on the specified kmesh and contains their opposites.
    """
    # Create kmesh points
    kmesh_points = np.mgrid[0:kmesh_num[0], 0:kmesh_num[1], 0:kmesh_num[2]].reshape(3, -1).T / np.array(kmesh_num) - 0.5
    
    # Check if all points in kveclist are in kmesh_points
    k_in_mesh = np.array([np.any(np.all(np.isclose(kvec, kmesh_points, rtol=1.e-5, atol=0), axis=1)) for kvec in kveclist])
    k_in_mesh_opposite = np.array([np.any(np.all(np.isclose(shift_kvec(-kvec), kveclist, rtol=1.e-5, atol=0), axis=1)) for kvec in kveclist])

    return np.all(k_in_mesh) and np.all(k_in_mesh_opposite)

class Plot_mat(object):
    def __init__(self,kveclist,dpi=100,figsize=(8,6),title='',ylabel=''):
        self.kveclist=np.array(kveclist)
        self.fig=plt.figure(dpi=dpi,figsize=figsize)
        self.title=title
        self.ylabel=ylabel
        pass
    def postprocess(self,colorbartmp):
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.fig.colorbar(colorbartmp, cax=cbar_ax)
        self.fig.suptitle(self.title)
        self.fig.supylabel(self.ylabel)
        pass
    def show(self):
        self.fig.show()
    def savefig(self,path):
        self.fig.savefig(path)
    def close(self):
        plt.close(self.fig)

class Plot_mat2D(Plot_mat):
    def __init__(self,kveclist,y_of_b_d_kmi:np.ndarray,color_of_b_d_kmi:np.ndarray,kpaths_pi_kpi,labelpath_pi_kpi,distance_tolerance_to_klines,seperate=True,title='',ylabel='',x_gap_between_kpaths=0.1,vmin=None,vmax=None,dpi=150,figsize=(8,6)):
        #y_of_b_d_k color_of_b_d_k should be of the same shape: (n,total_k_number)
        #d is the index of the columns, it means directions in the application of current plots
        #b is the index of the rows, it means bands in the application of current plots
        #pi is the index of the kpaths, a kpath is a list of kpoints. 
        #kp mean kpoints on k-paths
        #kpi are the index of kp 
        #There are one k-line in the k-space between one pair of the kpoints on kpaths. So one kpath consist several k-lines. The program would search for denser kpoints close to each k-lines from the kmesh of DMD.
        #km mean kpoints in k-mesh. These are the kpoints of the DMD Kmesh that are found close to each k-line. 
        #kmi are the index of km corresponding to the original kvec list of DMD data.
        #kmj are the index of km corresponding to the sequences stored in this class.
        # Note there are two K sampling concepts here: Kmesh vs Kpath.
        super().__init__(kveclist=kveclist,title=title,ylabel=ylabel,dpi=dpi,figsize=figsize)
        self.distance_of_km_to_klines=distance_tolerance_to_klines
        self.y_of_b_d_kmi=np.array(y_of_b_d_kmi)
        self.color_of_b_d_kmi=np.array(color_of_b_d_kmi)
        self.vmin=vmin
        self.vmax=vmax
        self.num_of_rows=self.y_of_b_d_kmi.shape[0]
        self.num_of_colomns=self.y_of_b_d_kmi.shape[1]
        self.kpaths_pi_kpi=kpaths_pi_kpi
        self.seperate=seperate
        self.axlist=[]

        if self.seperate:
            self.axlist=[[self.fig.add_subplot(self.num_of_rows,self.num_of_colomns,b*self.num_of_colomns+d+1) for d in range(self.num_of_colomns)] for b in range(self.num_of_rows)]
        else:
            self.axlist=[[self.fig.add_subplot(111)]*self.num_of_colomns]*self.num_of_rows

        for pi in range(len(self.kpaths_pi_kpi)):
            assert len(self.kpaths_pi_kpi[pi])==len(labelpath_pi_kpi[pi]), 'kpath and labelpath should have the same length'
        
        template_list_pi_kpi=[[[]for kpi in range(len(pi)-1)] for pi in self.kpaths_pi_kpi]
        
        self.kmilist_of_pi_kpi_kmj=deepcopy(template_list_pi_kpi)          
        self.parallel_projection_distance_from_km_to_klines_of_pi_kpi_kmj=deepcopy(template_list_pi_kpi) 
        self.color_of_b_d_pi_kpi_kmj=[[deepcopy(template_list_pi_kpi)  for d in range(self.num_of_colomns)] for b in range(self.num_of_rows)]
        self.xlabel_of_allpaths_pi_kpi=[]
        self.x_of_allpaths_of_pi_kpi=[]
    
        largest_x_of_last_path=0
        for pi in range(len(kpaths_pi_kpi)):
            kpath_kpi=self.kpaths_pi_kpi[pi]
            Labelpath_kpi=labelpath_pi_kpi[pi]
            kpathshift_kpi=np.roll(kpath_kpi,1,axis=0)
            kpathshift_kpi[0,:]=kpath_kpi[0]
            len_kpath_kpi=LA.norm(kpath_kpi-kpathshift_kpi,axis=1)
            len_kpath_sum_kpi=np.cumsum(len_kpath_kpi)
            self.x_of_allpaths_of_pi_kpi.append(len_kpath_sum_kpi+largest_x_of_last_path)
            self.xlabel_of_allpaths_pi_kpi.append(Labelpath_kpi)
            largest_x_of_last_path+=len_kpath_sum_kpi[-1]+x_gap_between_kpaths

        self.get_kmi_and_parallel_projection_distance()
        max_color, min_color=self.get_color_range()
        colorbartmp = self.plot(max_color, min_color)
        self.postprocess(colorbartmp)
   
    def get_kmi_and_parallel_projection_distance(self):
        for pi,kpath in enumerate(self.kpaths_pi_kpi):
            for kpi in range(len(kpath)-1):
                k1=kpath[kpi]
                k2=kpath[kpi+1]
                kmi,parallel_projection_distance_from_km_to_klines=self.get_km_near_kline(k1,k2,self.distance_of_km_to_klines)
                self.kmilist_of_pi_kpi_kmj[pi][kpi]=kmi
                self.parallel_projection_distance_from_km_to_klines_of_pi_kpi_kmj[pi][kpi]=parallel_projection_distance_from_km_to_klines
    def get_color_range(self):
        for b in range(self.num_of_rows):
            for d in range(self.num_of_colomns):
                for pi,kpath in enumerate(self.kpaths_pi_kpi):
                    for kpi in range(len(kpath)-1):
                        self.color_of_b_d_pi_kpi_kmj[b][d][pi][kpi]=self.color_of_b_d_kmi[b][d][self.kmilist_of_pi_kpi_kmj[pi][kpi]]
        max_color=np.max(np.abs(np.array(self.color_of_b_d_pi_kpi_kmj))) if self.vmin is None else self.vmax
        min_color=-np.max(np.abs(np.array(self.color_of_b_d_pi_kpi_kmj))) if self.vmin is None else self.vmin
        return max_color,min_color
    def plot(self, max_color, min_color):
        for b in range(self.num_of_rows):
            for d in range(self.num_of_colomns):
                ax=self.axlist[-b-1][d]
                for pi,kpath in enumerate(self.kpaths_pi_kpi):
                    for kpi in range(len(kpath)-1):
                        #print(path_i,func_i,k_i)
                        kmilist_of_kmj=self.kmilist_of_pi_kpi_kmj[pi][kpi]
                        color=self.color_of_b_d_pi_kpi_kmj[b][d][pi][kpi]
                        dis_on_line_of_kmj=self.parallel_projection_distance_from_km_to_klines_of_pi_kpi_kmj[pi][kpi]
                        #print(dis_on_line)
                        colorbartmp=ax.scatter(self.x_of_allpaths_of_pi_kpi[pi][kpi]+dis_on_line_of_kmj,self.y_of_b_d_kmi[b][d][kmilist_of_kmj],c=color\
                                    ,s=5,vmin=min_color,vmax=max_color,cmap='rainbow')
        return colorbartmp
    def postprocess(self,colorbartmp):
        x_of_allpaths_of_pi_kpi_flatten=[x for x_of_path_kpi in self.x_of_allpaths_of_pi_kpi for x in x_of_path_kpi]
        xlabel_of_allpaths_pi_kpi_flatten=[xlabel for xlabel_of_path_kpi in self.xlabel_of_allpaths_pi_kpi for xlabel in xlabel_of_path_kpi]
        if self.seperate:
            for b in range(self.num_of_rows):
                for d in range(self.num_of_colomns):
                    ax=self.axlist[-b-1][d]
                    ax.set_xlim(0,x_of_allpaths_of_pi_kpi_flatten[-1])
                    ax.set_xticks(x_of_allpaths_of_pi_kpi_flatten,[])
            for d in range(self.num_of_colomns):
                self.axlist[-1][d].set_xlabel('kpath')
                self.axlist[-1][d].set_xticks(x_of_allpaths_of_pi_kpi_flatten,xlabel_of_allpaths_pi_kpi_flatten)      
        else:
            ax=self.axlist[0][0]
            ax.set_xlim(0,x_of_allpaths_of_pi_kpi_flatten[-1])
            ax.set_xticks(x_of_allpaths_of_pi_kpi_flatten,[])
        super().postprocess(colorbartmp)

    def get_km_near_kline(self,k1,k2,distance):
        kvec_of_line=k2-k1
        kveclist_minus_k1=self.kveclist-k1
        kvec_of_line_norm=LA.norm(kvec_of_line)
        dis_paral=np.einsum('kd,d->k',kveclist_minus_k1,kvec_of_line)/kvec_of_line_norm
        kveclist_project_paral=np.einsum('k,d->kd',dis_paral,kvec_of_line)/kvec_of_line_norm
        kveclist_projct_perp=kveclist_minus_k1-kveclist_project_paral
        dis_perp=LA.norm(kveclist_projct_perp,axis=1)
        in_paral_range=np.logical_and(0<=dis_paral,dis_paral<=LA.norm(kvec_of_line))
        in_perp_range=dis_perp<=distance
        final_range=np.logical_and(in_paral_range,in_perp_range)
        knum_list=np.arange(len(self.kveclist))[final_range]
        return knum_list,dis_paral[final_range]
class Plot_mat3D(Plot_mat):
    def __init__(self,kveclist,color_b_k,vmin=None,vmax=None,title='',rownum=1,index_str_for_subtitle='i',subtitle_on=True,dpi=100,figsize=(12,6)):
        #init
        super().__init__(kveclist=kveclist,dpi=dpi,figsize=figsize,title=title)
        b_num=len(color_b_k)
        assert b_num>=rownum, 'rownum should be equal to or smaller than the number of figures'
        columnnum=b_num//rownum+1 if b_num%rownum!=0 else b_num//rownum
        axlist = [self.fig.add_subplot(rownum,columnnum,i,projection='3d') for i in range(1,b_num+1)]
        colormax=np.max(np.abs(color_b_k)) if vmax is None else vmax
        colormin=-np.max(np.abs(color_b_k)) if vmin is None else vmin
        #plot
        for b,color_k in enumerate(color_b_k):
            ax=axlist[b]
            colorbartmp=ax.scatter(kveclist[:,0],kveclist[:,1],kveclist[:,2],c=color_k,s=5,cmap='rainbow',vmin=colormin,vmax=colormax)
            #ax.set_xlim(-0.5,0.5)
            #ax.set_ylim(-0.5,0.5)
            #ax.set_zlim(-0.5,0.5)
            
        #postprocess
        for ax_i,ax in enumerate(axlist):
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('kz')
            if subtitle_on:
                ax.set_title('%s %d'%(index_str_for_subtitle,ax_i))

        super().postprocess(colorbartmp)