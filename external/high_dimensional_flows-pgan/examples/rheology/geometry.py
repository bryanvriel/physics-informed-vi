#-*- coding: utf-8 -*-

# Get numpy
import numpy as np

# Other packages
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys


class Profile:

    def __init__(self, x, h, u, A, rho_ice=917.0, rho_water=1024.0, t=0.0):
        """
        Stores profiles of DOWNSTREAM distance:

            s = 0 -> glacier start
            s = L -> glacier terminus

        and ice thickness and bed elevation below sea level.

        Parameters
        ----------
        x: (N,) ndarray
            Array of downstream coordinates in meters.
        h: (N,) ndarray
            Array of ice thickness in meters.
        b: (N,) ndarray
            Array of bed elevation in meters.
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        rho_ice: float, optional
            Ice density in kg/m^3. Default: 917.
        rho_water: float, optional
            Ocean water density in kg/m^3. Default: 1024.
        t: float, optional
            Time associated with profile data.

        Returns
        -------
        None
        """
        # Store the data
        self.x = x
        self.h = h
        self.u = u
        self.A = A
        self.rho_water, self.rho_ice = rho_water, rho_ice
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.t = t

        # Upstream distance
        self.l = x.max() - x

        # Create finite difference matrix
        self.D = construct_finite_diff_matrix(self.x, edge_order=2)

        # Initialize other thickness-dependent quantities
        self._init_derived_quantities()

    def _init_derived_quantities(self):
        """
        Private function for computing quantities depending on the thickness profile.
        """
        # Ice surface
        self.s = self.h * (1.0 - self.rho_ice / self.rho_water)
        # Surface slope
        self.alpha = -1.0 * np.dot(self.D, self.s)

    def set_profile_ydata(self, h=None, u=None):
        """
        Update thickness profiles for height, bed, and velocity.

        Parameters
        ----------
        h: (N,) ndarray
            Array of ice thickness in meters.
        u: (N,) ndarray
            Array of ice velocity in m/yr.
        
        Returns
        -------
        None
        """
        # Height
        if h is not None:
            self.h = h
            self._init_derived_quantities()
        
        # Velocity
        if u is not None:
            self.u = u

    def update_coordinates(self, x, interp_kind='cubic', extrapolate=False):
        """
        Creates a NEW Profile object with a new set of coordinates.

        Parameters
        ----------
        x: (N,) ndarray
            Array of downstream coordinates in meters to interpolate to.
        interp_kind: str, optional
            scipy.interp1d kwarg for interpolation kind. Default: 'cubic'.
        extrapolate: bool, optional
            Flag for extrapolation beyond x bounds. Default: False.

        Returns
        -------
        profile: Profile
            New Profile instance.
        """
        # Handle extrapolation arguments
        if extrapolate:
            fill_value = 'extrapolate'
            bounds_error = False
        else:
            fill_value = np.nan
            bounds_error = True

        # Create new object with interpolated profiles
        return Profile(
            x,
            interp1d(self.x, self.h, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
            interp1d(self.x, self.u, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
            interp1d(self.x, self.A, kind=interp_kind, fill_value=fill_value,
                     bounds_error=bounds_error)(x),
        )

    def smoothe(self, s=1.0, k=3, scale=1.0e-2, plot=False):
        """
        Use UnivariateSpline to smoothe velocity and thickness profiles.
        """
        x = 1.0e-3 * self.x
        if isinstance(s, int):
            s = {'u': s, 'h': s}
        assert isinstance(s, dict)
        for attr in ('u', 'h'):
            data = getattr(self, attr)
            spl = UnivariateSpline(x, scale*data, s=s[attr], k=k)
            data_sm = spl(x) / scale
            if plot:
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(self.x, data)
                ax.plot(self.x, data_sm)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            setattr(self, attr, data_sm)

        return
    
    def plot(self, axes=None, items=['geo', 'vel']):
        """
        Construct subplots for viewing various profile data.

        Parameters
        ----------
        axes: pyplot.axes.Axes, optional
            Pre-constructed axes to plot into.
        items: list, optional
            List of items to plot. Choose from ['geo', 'vel', 'flux'].
        
        Returns
        -------
        None
        """
        # Import default numpy
        import numpy

        if axes is None:
            fig, axes = plt.subplots(nrows=len(items), figsize=(10,6))
        else:
            assert len(axes) == len(items), 'Incompatible number of subplots'
        for count, item in enumerate(items):
            ax = axes[count]
            if item == 'geo':
                ax.plot(self.x, self.s, label='S')
            elif item == 'vel':
                ax.plot(self.x, self.u, label='U')
            elif item == 'flux':
                flux = self.h * self.u
                dflux = numpy.gradient(flux, self.x)
                ax.plot(self.x, flux, label='Flux')
                ax_twin = ax.twinx()
                ax_twin.plot(self.x, dflux, 'r')

        for ax in axes:
            ax.legend(loc='best')
            ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------

def load_profile_from_h5(h5file):
    """
    Creates a Profile object using data from an HDF5 output run

    Parameters
    ----------
    h5file: str
        Name of HDF5 file to load data from.

    Returns
    -------
    profile: Profile
        New Profile instance.
    """
    import h5py

    with h5py.File(h5file, 'r') as fid:
        u, h, A, x = [fid[key][()] for key in ('u', 'h', 'A', 'x')]
        profile = Profile(x, h, u, A)
        if 't' in fid.keys():
            profile.t = fid['t'][()]
        else:
            profile.t = 0.0
    return profile

def save_profile_to_h5(profile, h5file, aux_data={}):
    """
    Saves a Profile object to HDF5.

    Parameters
    ----------
    profile: Profile
        Profile instance to save.
    h5file: str
        HDF5 file to write to.
    aux_data: dict, optional
        Dict of any additional data to write to HDF5.

    Returns
    -------
    None
    """
    import h5py
    with h5py.File(h5file, 'w') as fid:

        # Save standard profile data
        for key in ('u', 'h', 'A', 'x'):
            fid[key] = getattr(profile, key)
        if hasattr(profile, 't'):
            fid['t'] = profile.t

        # If any extra data has been provided, save those as Datasets
        for key, value in aux_data.items():
            fid[key] = value

    return

def construct_finite_diff_matrix(x, edge_order=2):
    """     
    Construct finite difference matrix operator using central differences.
    
    Parameters
    ----------
    x: (N,) ndarray
        Array of coordinates.
    edge_order: {1, 2}, optional
        Gradient is calculated using N-th order accurate differences at boundaries.
        Default: 1.

    Returns
    -------
    D: (N, N) ndarray
        2D array for finite difference operator.
    """
    # Need standard numpy
    import numpy
      
    # Non-uniform grid spacing
    N = x.size
    dx = numpy.diff(x)
    D = numpy.zeros((N, N))
                
    # Off-diagonals for central difference
    dx1 = dx[:-1]
    dx2 = dx[1:]
    a = - (dx2) / (dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    D[range(1, N-1), range(0, N-2)] = a
    D[range(1, N-1), range(1, N-1)] = b
    D[range(1, N-1), range(2, N)] = c

    # First-order edges
    if edge_order == 1:

        # Left
        a = -1.0 / dx[0]
        b = 1.0 / dx[0]
        D[0,:2] = [a, b]

        # Right
        a = -1.0 / dx[-1]
        b = 1.0 / dx[-1]
        D[-1,-2:] = [a, b]

    # Second-order edges
    elif edge_order == 2:

        # Left
        dx1 = dx[0]
        dx2 = dx[1]
        a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
        b = (dx1 + dx2) / (dx1 * dx2)
        c = - dx1 / (dx2 * (dx1 + dx2))
        D[0,:3] = [a, b, c]

        # Right
        dx1 = dx[-2]
        dx2 = dx[-1]
        a = (dx2) / (dx1 * (dx1 + dx2))
        b = - (dx2 + dx1) / (dx1 * dx2)
        c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
        D[-1,-3:] = [a, b, c]

    else:
        raise ValueError('Invalid order number.')
    
    # Done
    return D


# end of file
