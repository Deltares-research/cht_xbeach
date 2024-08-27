## python file to read XBeach output and write to netcdf file

# $Id: xbeach.py 12827 2016-07-29 14:49:32Z janjaapmeijer.x $
# $Date: 2016-07-29 16:49:32 +0200 (vr, 29 jul 2016) $
# $Author: janjaapmeijer.x $
# $Revision: 12827 $
# $HeadURL: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/OpenEarthTools/openearthtools/io/xbeach/xbeach.py $
# $Keywords: $

import array,os,numpy,sys
# from pupynere import netcdf_file as nc
from scipy.io.netcdf import netcdf_file as nc
import numpy as np
import datetime
from optparse import OptionParser
import pandas as pd
import math
from pyproj import CRS
from pyproj import Transformer
import time

from cht.misc.geometry import Point

# constants
fmt = '%Y-%m-%d %H:%M:%S'

class XBeach:
    
    def __init__(self, input_file=None, crs=None, get_boundary_coordinates=True):
 
#        self.epsg                     = epsg
        self.params                   = Params()
        self.crs                      = crs
        self.grid                     = None
        self.flow_boundary_point      = []
        self.wave_boundary_point      = []
        self.observation_point        = []
        self.obstacle                 = []
        
        if input_file:
            self.path   = os.path.dirname(input_file)
            self.params = self.params.fromfile(filename=input_file)
            self.get_boundary_points(get_boundary_coordinates)

    def load(self, input_file):
        # Reads sfincs.inp and attribute files
        self.params = self.params.fromfile(filename=input_file)
        self.read_attribute_files()
        
    def read_attribute_files(self):
        pass

    def get_boundary_points(self, get_boundary_coordinates):

        self.flow_boundary_point = []
        self.wave_boundary_point = []
        
        if get_boundary_coordinates:
            xfile = os.path.join(self.path, self.params["xfile"])
            yfile = os.path.join(self.path, self.params["yfile"])        
            xg = np.loadtxt(xfile)
            yg = np.loadtxt(yfile)
            x0 = xg[0, 0]
            y0 = yg[0, 0]
            x1 = xg[0, -1]
            y1 = yg[0, -1]
            x2 = xg[-1, 0]
            y2 = yg[-1, 0]
            x3 = xg[-1, -1]
            y3 = yg[-1, -1]
        else:
            x0 = 0.0
            y0 = 0.0
            x1 = 0.0
            y1 = 0.0
            x2 = 0.0
            y2 = 0.0
            x3 = 0.0
            y3 = 0.0
                
        if self.params["tideloc"] == 1:
            x = x0 + 0.5*(x2 - x0)
            y = y0 + 0.5*(y2 - y0)
            self.flow_boundary_point.append(BoundaryPoint(x, y))
        elif self.params["tideloc"] == 2:
            x = x0
            y = y0
            self.flow_boundary_point.append(BoundaryPoint(x, y))
            x = x2
            y = y2
            self.flow_boundary_point.append(BoundaryPoint(x, y))
        else:
            # Assume tideloc==4
            x = x0
            y = y0
            self.flow_boundary_point.append(BoundaryPoint(x, y))
            x = x2
            y = y2
            self.flow_boundary_point.append(BoundaryPoint(x, y))
            x = x1
            y = y1
            self.flow_boundary_point.append(BoundaryPoint(x, y))
            x = x3
            y = y3
            self.flow_boundary_point.append(BoundaryPoint(x, y))

        x = x0 + 0.5*(x2 - x0)
        y = y0 + 0.5*(y2 - y0)
        self.wave_boundary_point.append(BoundaryPoint(x, y))

    def write_flow_boundary_conditions(self, file_name=None):

        # Write tide.txt
        if not file_name:
            if not self.params["zs0file"]:
                return
            file_name = os.path.join(self.path,
                                     self.params["zs0file"])
            
        if not file_name:
            return
        
        # Build a new DataFrame
        df = pd.DataFrame()
        for point in self.flow_boundary_point:
            df = pd.concat([df, point.data], axis=1)
        
        # add constant water level at tideloc locations without water levels    
        if df.shape[1] < self.params["tideloc"]:
            for i in np.arange(df.shape[1],self.params["tideloc"]):
                df.insert(int(i),str(i),-0.5)
            
        tmsec = pd.to_timedelta(df.index.values - self.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")

    def write_wave_boundary_conditions(self, file_name=None, option="sp2"):

        if option=="timeseries":
            
            # Write jons table
            if not file_name:
                if not self.params["bcfile"]:
                    return
                file_name = os.path.join(self.path,
                                         self.params["bcfile"])
                
            if not file_name:
                return
            
            # Build a new DataFrame
            df = pd.DataFrame()
            for point in self.wave_boundary_point:
                df = pd.concat([df, point.data], axis=1)
            #tmsec = pd.to_timedelta(df.index - self.tref, unit="s")
            #df.index = tmsec.total_seconds()
            df.to_csv(file_name,
                      index=False,
                      sep=" ",
                      header=False,
                      float_format="%0.3f")
        
        else:
            # 2D spectra
            # Bit more work ...
            data = self.wave_boundary_point[0].data
            sz = np.shape(data.point_spectrum2d)
            nt   = sz[0]
            nphi = sz[1]
            nsig = sz[2]
            fname = os.path.join(self.path, "sp2list.txt")
            sp2files = []
            dt      = (data.time[1]-data.time[0]).values.astype('timedelta64[s]').astype(np.float)
            t       = pd.to_datetime(data.time.values)
            it0     = np.where(t>=self.tref)[0][0]
            with open(fname, 'w') as f:               
                f.write("FILELIST\n")
                for it in range(it0, nt):                    
                    ff = "xb.t" + str(it).zfill(4) + ".sp2"
                    sp2files.append(ff)
                    f.write(str(dt) + " 1.0 " + ff + "\n")
                f.close()
                
            for it, sp2file in enumerate(sp2files):
                
                with open(os.path.join(self.path, sp2file), 'w') as f:
                    
                    f.write("SWAN   1                                Swan standard spectral file, version\n")
                    f.write("$   Data produced by SWAN version 40.51AB\n")
                    f.write("$   Project:                 ;  run number:\n")
                    f.write("TIME                                    time-dependent data\n")
                    f.write("     1                                  time coding option\n")
                    f.write("1\n")
                    f.write(str(self.wave_boundary_point[0].geometry.x) + " " + str(self.wave_boundary_point[0].geometry.y) + "\n")
                    f.write("AFREQ                                   absolute frequencies in Hz\n")
                    f.write(str(nsig) + "\n")
                    for ifreq in range(0, nsig):
                        f.write(f'{data.sigma.values[ifreq]:.4f}' + "\n")
                    f.write("NDIR                                   spectral nautical directions in degr\n")
                    f.write(str(nphi) + "\n")
                    for itheta in range(0, nphi):
                        # Convert to nautical, coming from
                        phi = np.mod(270.0 - data.theta.values[itheta], 360.0)
                        f.write(f'{phi:.1f}' + "\n")
                    f.write("QUANT\n")
                    f.write("     1                                  number of quantities in table\n")
                    f.write("EnDens                                  energy densities in J/m2/Hz/degr\n")
                    f.write("J/m2/Hz/degr                            unit\n")
                    f.write("   -0.9900E+02                          exception value\n")
                    tsec = data.time.values[it + it0].astype(datetime.datetime)
                    tstr = datetime.datetime.utcfromtimestamp(tsec/1000000000).strftime("%Y%m%d.%H%M%S")
                    f.write(tstr + "\n")
                    f.write("FACTOR\n")
                    sp2 = np.transpose(data.point_spectrum2d.values[it + it0, :, :])                 
                    # Convert from wave action to energy
                    for ifreq in range(0, nsig):
                        sp2[ifreq,:] = sp2[ifreq,:]*data.sigma.values[ifreq]*1024.0*9.81
                    mxmx = np.max(sp2)
                    fac = mxmx/990099
                    fac = max(fac, 1.e-12)
                    f.write(f'{fac:.8e}' + "\n")                    
                    sp2 = (sp2/fac).astype(int)
                    sp2 = np.clip(sp2, 1.0, None)
                    np.savetxt(f, sp2, fmt="%.7i")
                    f.close()

    def make_index_tiles(self, path, zoom_range=None):
        """Make tiles for different zoom levels by saving the index of the xbeach grid cell
        to be used for each world png raster"""
        from cht.tiling.tiling import deg2num
        from cht.tiling.tiling import num2deg
        import cht.misc.fileops as fo
        from scipy.spatial import distance

        if not zoom_range:
            zoom_range = [0, 13]

        npix = 256

        # Compute lon/lat range
        lon_range, lat_range = self.bounding_box(crs=CRS.from_epsg(4326))

        # get grid orientation
        xb, yb = self.grid_coordinates(loc='cor')
        alpha = math.atan2(yb[0, -1] - yb[0, 0], xb[0, -1] - xb[0, 0]) * 180 / math.pi

        # Get origin (corner)
        x0, y0 = xb[0, 0], yb[0, 0]

        # Get distance vectors (x and y are the cross-shore and alongshore axis here)
        xvec = np.array([distance.euclidean((xp, yp), (x0, y0)) for xp, yp in zip(xb[0, :], yb[0, :])])
        yvec = np.array([distance.euclidean((xp, yp), (x0, y0)) for xp, yp in zip(xb[:, 0], yb[:, 0])])

        # Get number of cells in each direction
        xcells, ycells = len(xvec)-1, len(yvec)-1

        # Get rotation parameters
        cosrot = math.cos(-alpha * math.pi / 180)
        sinrot = math.sin(-alpha * math.pi / 180)

        # Get transformers between crs
        transformer_a = Transformer.from_crs(CRS.from_epsg(4326),
                                             CRS.from_epsg(3857),
                                             always_xy=True)
        transformer_b = Transformer.from_crs(CRS.from_epsg(3857),
                                             self.crs,
                                             always_xy=True)
        # Loop through each zoom level
        for izoom in range(zoom_range[0], zoom_range[1] + 1):

            print("Processing zoom level " + str(izoom))

            zoom_path = os.path.join(path, str(izoom))  # Create path for zoom level

            dxy = (40075016.686 / npix) / 2 ** izoom  # Get png cell size based on zoom level
            xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)  # Get distance grid for each tile

            # Get range of tile indices to be used for the zoom level
            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)

            for i in range(ix0, ix1 + 1):  # loop through x tiles

                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))  # Create path for x index

                for j in range(iy0, iy1 + 1):

                    file_name = os.path.join(zoom_path_i, str(j) + ".dat")  # Create file for y index

                    # Compute lat/lon at ll corner of tile
                    lat, lon = num2deg(i, j, izoom)

                    # Convert to Global Mercator
                    xo, yo = transformer_a.transform(lon, lat)

                    # Tile grid on local mercator (this is at cell center)
                    x = xv[:] + xo + 0.5 * dxy
                    y = yv[:] + yo + 0.5 * dxy

                    # Convert tile grid to crs of xbeach model
                    x, y = transformer_b.transform(x, y)

                    # Now rotate around origin of XBeach model (with x0, y0 the cell corner of xbeach)
                    x00 = x - x0
                    y00 = y - y0
                    xg = x00 * cosrot - y00 * sinrot
                    yg = x00 * sinrot + y00 * cosrot

                    # find mask of cells falling in XBeach domain
                    imask = (xg > 0) & (xg <= xvec[-1])
                    jmask = (yg > 0) & (yg <= yvec[-1])
                    mask = imask & jmask
                    ind = np.full(mask.shape, -999)  # Prepare index matrix

                    for ii in range(len(mask[:, 0])):  # loop through x
                        for jj in range(len(mask[0, :])):  # loop through y
                            if mask[ii, jj]:  #  only calculate if in the XBeach domain
                                iind = grid_ind(xg[ii, jj], xvec)
                                jind = grid_ind(yg[ii, jj], yvec)
                                ind[ii, jj] = iind * ycells + jind

                    if np.any(ind >= 0):  # Only continue if there is at least on png cell in the domain

                        if not path_okay:  # ensure path exists to save file
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True

                        # And write indices to file
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()


    def grid_coordinates(self, loc='cor'):
        """Get grid coordinates either at cell centers or corenrs"""
        # Get coordinates of cell centers
        xg = np.loadtxt(os.path.join(self.path, self.params['xfile']))
        yg = np.loadtxt(os.path.join(self.path, self.params['yfile']))

        if loc == 'cor': # if cells corners are used
            # Get cell grid size in x and y
            dxx = np.array([xg[0, 1] - xg[0, 0]] + [xg[0, i] - xg[0, i - 1] for i in range(1, len(xg[0, :]))])
            dyx = np.array([yg[0, 1] - yg[0, 0]] + [yg[0, i] - yg[0, i - 1] for i in range(1, len(yg[0, :]))])
            dxy = np.array([xg[1, 0] - xg[0, 0]] + [xg[i, 0] - xg[i - 1, 0] for i in range(1, len(xg[:, 0]))])
            dyy = np.array([yg[1, 0] - yg[0, 0]] + [yg[i, 0] - yg[i - 1, 0] for i in range(1, len(yg[:, 0]))])
            # Define grid at cell corners
            new_x, new_y = [], []
            for i in range(len(xg[:, 0])):
                new_x.append(xg[i, :] - (dxx / 2 + np.repeat(dxy[i], len(dxx)) / 2))
                new_y.append(yg[i, :] - (dyx / 2 + np.repeat(dyy[i], len(dyx)) / 2))
            new_x, new_y = np.array(new_x), np.array(new_y)
            # Add last grid line (new grid represent the extent)
            new_x = np.vstack((new_x, new_x[-1, :] + np.repeat(dxy[-1], len(dxx))))
            new_y = np.vstack((new_y, new_y[-1, :] + np.repeat(dyy[-1], len(dyx))))
            new_x = np.hstack((new_x, (new_x[:, -1] + np.repeat(dxx[-1], len(dxy)+1))[:, None]))
            new_y = np.hstack((new_y, (new_y[:, -1] + np.repeat(dyx[-1], len(dyy)+1))[:, None]))

        else:  # if cell centres are used
            new_x = xg
            new_y = yg

        return new_x, new_y


    def bounding_box(self, crs=None):
        """Determine bounding box of Xbeach grid"""
        xg, yg = self.grid_coordinates(loc='cor')

        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)

        x_range = [np.min(np.min(xg)), np.max(np.max(xg))]
        y_range = [np.min(np.min(yg)), np.max(np.max(yg))]

        return x_range, y_range


class BoundaryPoint():

    def __init__(self, x, y, name=None, crs=None, data=None):
        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data

class Point():
        
    def __init__(self, x, y, name = None, crs=None):
        
        self.x       = x
        self.y       = y
        self.crs     = crs
        self.name    = name
        self.data    = None

class Params(dict):
    """Read and write XBeach params.txt files."""
    @staticmethod
    def fromfile(filename='params.txt'):
        """read the params/txt file"""
        with open(filename,'U') as f:
            key = None
            values = Params()
            for line in f:
                if line.strip().startswith('#') or line.strip().startswith('-') or line.strip().startswith('%') or line.strip().endswith(':') or not line.strip():
                    continue
                else:
                    if '=' in line:
                        key, value = map(str.strip, line.split('='))
                        values[key] = tonumber(value)
                    else:
                        if key is None:
                            raise ValueError("invalid line:\n" + line)
                        # We have some array type of value
                        values[key[1:]] = values.get(key[1:], []) + [line.strip()]
        return values

    def tofile(self, filename="params.txt"):
        general = ['wavemodel','wbctype', 'deltahmin', 'fixedavaltime', 'oldTsmin', 'oldhmin', 'snells', 'droot', 'dstem', 'dynamicroughness', 'alfaD50']
        phys_process = ['swave', 'nonh', 'single_dir', 'sedtrans', 'morphology', 'avalanching']
        grid_params = ['xori', 'yori', 'alfa', 'nx', 'ny', 'posdwn', 'depfile', 'vardx', 'xfile', 'yfile',
                       'thetamin', 'thetamax', 'thetanaut', 'dtheta_s', 'dtheta', 'gridform']
        model_time = ['tstop']
        wave_bc = ['break', 'gamma', 'gamma2', 'alpha', 'bcfile', 'wavint']
        tide_bc = ['tideloc', 'zs0file', 'front', 'back', 'tidetype']
        flow_params = ['bedfriction', 'bedfricfile', 'nuhfac']
        bed_params = ['D50']
        morph_params = ['morfac', 'morstart','wetslp', 'struct', 'ne_layer']
        roller_params = ['beta']
        sed_params = ['waveform','facAs', 'facSk']
        output_vars = ['tintg', 'tintm', 'outputformat', 'tintp', 'tstart']

        with open(filename, 'w') as f:
            f.writelines('### XBeach parameter settings input file\n')
            f.writelines('### Created on: ' + datetime.datetime.now().strftime(fmt) + '\n')

            f.writelines('\n### General\n')
            for key, value in self.items():
                if key in general:
                        f.writelines('%-15s= %s' % (key, value) + '\n')
                        
            f.writelines('\n### Bed composition parameters\n')
            for key, value in self.items():
                if key in bed_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Physical processes\n')
            for key, value in self.items():
                if key in phys_process:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Grid parameters\n')
            for key, value in self.items():
                if key in grid_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Model time parameters\n')
            for key, value in self.items():
                if key in model_time:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Wave breaking parameters\n')
            for key, value in self.items():
                if key in wave_bc:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Tide boundary condition\n')
            for key, value in self.items():
                if key in tide_bc:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Flow parameters\n')
            for key, value in self.items():
                if key in flow_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Morphology parameters\n')
            for key, value in self.items():
                if key in morph_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Sediment transport parameters\n')
            for key, value in self.items():
                if key in sed_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')
                        
            f.writelines('\n### Roller parameters\n')
            for key, value in self.items():
                if key in roller_params:
                        f.writelines('%-15s= %s' % (key, value) + '\n')

            f.writelines('\n### Output variables\n')
            for key, value in self.items():
                if key in output_vars:
                    f.writelines('%-15s= %s' % (key, value) + '\n')

            for key, value in self.items():
                if key == 'nglobalvar':
                    f.writelines("\n%s = %s" % (key, value) + '\n')
                    for val in self['globalvar']:
                        f.writelines(val + '\n')
                    f.writelines('\n')
                elif key == 'nmeanvar':
                    f.writelines("%s = %s" % (key, value) + '\n')
                    for val in self['meanvar']:
                        f.writelines(val + '\n')


                # elif key in grid_params:
                #     for line in open(filename):
                #         match = re.match(r'###\s+Grid\s+Parameters.*$', line)
                #         print(match)
                #         if match:
                #             with open(filename, 'a') as f:
                #                 f.writelines("%s=%s" % (key, value) + '\n')
                #                 f.close()
                #


            # for line in open(filename):
            #     print(line)
            #     print(match)
            #     match = heading.match(line)
            #     if match: # found ID, print with comma
            #         f.write(match.group(1) + ",")
            #         continue
            #     match = pat_name.match(line)
            #     if match: # found name, print and end line
            #         f.write(match.group(1) + "\n")
            #
            #


# functions
def grid_ind(value, grid):
    """ get index of cell based on distance grid"""
    dif = value - grid
    i0 = np.argmin(abs(dif))
    i = i0 if dif[i0] > 0 else i0 - 1
    return i

def listdat(path):
    """find all .dat-files in directory"""
    import glob
    return glob.glob(os.path.join(path,'*.dat'))

def listfiles(path):
    import glob
    return glob.glob(os.path.join(path,'*.*'))

def copyfiles(src_path, dest_path):
    import shutil
    for filename in listfiles(src_path):
        shutil.copy(filename, dest_path)

def readdims(fullfile, verbose=True):
    """read dimensions from dims.dat"""

    if verbose:
        print('reading file: ' + fullfile)

    fileobj = open(fullfile, mode='rb')
    
    binvalues = array.array('d')
    binvalues.read(fileobj, 1 * 14)
    dims = numpy.array(binvalues, dtype=int);
    
    nt,nx,ny = tuple(1 + dims[0:3])
    
    fileobj.close()
    
    return nt,nx,ny

def readxy(fullfile, nx, ny, verbose=True):
    """read x and y from xy.dat"""
    
    if verbose:
        print('reading file: ' + fullfile)

    fileobj = open(fullfile, mode='rb')
    
    binvalues = array.array('d')
    binvalues.read(fileobj, nx * ny * 2)
    
    fileobj.close()
    
    xy = numpy.array(binvalues)
    
    x = numpy.reshape(xy[0:nx*ny], (ny, nx)).T
    y = numpy.reshape(xy[-nx*ny:], (ny, nx)).T
    
    return x,y

def readdata(fullfile, nx, nt, verbose=True):
    """read <variable> from <variable>.dat"""
    
    if verbose:
        print('reading file: ' + fullfile)
        
    fileobj = open(fullfile, mode='rb')
    
    binvalues = array.array('d')
    binvalues.read(fileobj, nx * nt)
    
    fileobj.close()
    
    data = numpy.array(binvalues)

    data = numpy.reshape(data, (nt, nx))
    
    return data

def tonumber(text):
    """cast a text to a number"""
    try:
        return int(text)
    except ValueError as e:
        pass
    try:
        return float(text)
    except ValueError as e:
        pass
    return text

def readbathy(filename='bed.dep'):
    """read the bathymetry file"""
    bathy = np.loadtxt(filename)
    return bathy

def writenc(ncfile, XB, nt, nx, ny, verbose=True):
    """write variable to netcdf file"""
    
    if verbose:
        print('writing file "' + ncfile + '"')
    # open file
    f = nc(ncfile, 'w')
    # global attributes
    f.title = 'XBeach calculation result'
    f.source = os.path.join(datadir, '*.dat')
    f.history = 'created on: ' + datetime.datetime.now().strftime(fmt)
    # dimensions
    f.createDimension('time', nt)
    f.createDimension('cross_shore', nx)
    f.createDimension('alongshore', ny)
    
    for variable in XB:
        if variable in ('x', 'y'):
            dims = ('cross_shore', 'alongshore')
        else:
            dims = ('time', 'cross_shore')
        tmp = f.createVariable(variable, 'f', dims)
        tmp[:] = XB[variable]
    
    # close file
    f.close()

if __name__ == '__main__':
    if not os.path.basename(sys.argv[0]) == 'spyder.pyw':
        # if not running in spyder
        parser = OptionParser()
        parser.add_option("-d", "--directory", dest="datadir", default='.', help="read .dat-files from <directory>*.dat")
        parser.add_option("-f", "--file", dest="filename", default='result.nc', help="write data to netcdf file FILENAME (extension='.nc')")
        parser.add_option("-q", "--quiet", action="store_false", dest="verbose", default=True, help="don't print status messages")
        (options, args) = parser.parse_args()
        
        datadir = os.path.abspath(options.datadir)
        verbose = options.verbose
        ncpath,ncfile = os.path.split(options.filename)
        if ncpath == '':
            ncpath = datadir
    else:
        datadir = os.path.abspath('.')
        ncpath = datadir
        ncfile = 'result.nc'
        verbose = True
    ncfname,ncext = os.path.splitext(ncfile)
    
    if not ncext.lower() == '.nc':
        ncfile = os.path.join(ncfname, '.nc')

    var = dict(listdat(datadir))
    if var.keys() == []:
        print('No .dat-files found in "' + datadir + '"')
    else:
        XB = dict()
        nt,nx,ny = readdims(var['dims'], verbose=verbose)
        XB['x'],XB['y'] = readxy(var['xy'], nx, ny, verbose=verbose)
        for variable in var:
            if not variable in ('dims', 'xy'):
                XB[variable] = readdata(var[variable], nx, nt, verbose=verbose)
        writenc(os.path.join(ncpath, ncfile), XB, nt, nx, ny, verbose=verbose)
