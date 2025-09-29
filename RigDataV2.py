"""
RigDataV2.py - Data Loading and Analysis Module

Author: Shane Nichols
LinkedIn: https://www.linkedin.com/in/shane-nichols/

This module provides classes and utilities for loading and analyzing experimental
data from instrumentation systems, including waveform data and camera acquisitions.
"""

import os
from typing import overload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import xarray as xr


def readParamFile(filepath:str):
    type_converters = {
        'str': str,
        'float': float,
        'int': int,
        'bool': bool,
        'float[]': lambda s: np.fromstring(s, dtype=float, sep=' '),
        'int[]': lambda s: np.fromstring(s, dtype=int, sep=' '),
        'str[]': lambda s: s.split(' ')
    }
    params = pd.read_csv(filepath, header=None, names=['name', 'dtype', 'value'], dtype=str, keep_default_na=False)
    params = dict(zip(params['name'], params.apply(lambda s: type_converters[s['dtype']](s['value']), axis=1)))
    return params


class InstrumentWaveform:
    '''A struct for a single waveform'''
    def __init__(self, waveform: np.ndarray, dt: float, name: str):
        # waveform data
        self.waveform = waveform
        # sample interval, in seconds
        self.dt = dt
        # name of the waveform
        self.name = name

    def __str__(self):
        return pprint.pformat(vars(self))

    def times(self):
        return np.linspace(0, self.waveform.size * self.dt, self.waveform.size, endpoint=False)

    def plot(self, ax=None, linespec='-', **kwargs):
        if not ax:
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel('Time [s]')
            ax.set_title(self.name)
        ax.plot(self.times(), self.waveform, linespec, **kwargs)


class InstrumentWaveforms(dict):
    def __init__(self, name='', *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    def plot(self, ax=None):
        if not ax:
            f,ax = plt.subplots()
            ax.set_xlabel('Time [s]')
            ax.set_title(self.name)
        for key in self:
            self[key].plot(ax=ax, label=key)
        ax.legend()


class RigData:
    def __init__(self, data_dir=os.getcwd()):
        self.datapath = data_dir
        self.name = os.path.split(self.datapath)[1]
        self.version = '0'  # default version
        # load and parse parameters file
        params = readParamFile(os.path.join(self.datapath, 'Parameters.txt'))
        # fix repetition variables if there were no reps
        if params['repetitionsType'] == 'Single Shot':
            params['repetitionsNumber'] = 1
            params['repetitionsInterval'] = 0
        # set attributes from params
        for key in params:
            if key not in ['names', 'types','sampleRates', 'waveformSizes']:
                setattr(self, key, params[key])
        # initialize waveform containers
        self.ao = InstrumentWaveforms()
        self.ai = InstrumentWaveforms()
        self.do = InstrumentWaveforms()
        self.di = InstrumentWaveforms()
        # load waveform data
        wfms = np.fromfile(os.path.join(self.datapath, 'waveforms.bin'), dtype=np.dtype('>f'))
        idx = 0
        n = np.sum(params['types'] < 2)
        for i,typ in enumerate(params['types'][0:n]):
            sz = params['waveformSizes'][i]
            channel = 'ao' if typ == 0 else 'ai'
            getattr(self, channel)[params['names'][i]] = InstrumentWaveform(
                wfms[idx:(idx+sz)],
                1/params['sampleRates'][i],
                params['names'][i]
            )
            idx += sz
        nwf = params['types'].size - n
        if nwf > 0:
            wfms = wfms[idx::].astype(int)
            binAr = np.zeros((np.max(params['waveformSizes'][n::]), nwf), dtype=bool)
            for j in np.arange(nwf):
                binAr[:,j] = wfms % 2
                wfms = wfms // 2
            for i,typ in enumerate(params['types'][n::]):
                sz = params['waveformSizes'][i+n]
                channel = 'do' if typ == 2 else 'di'
                getattr(self, channel)[params['names'][i+n]] = InstrumentWaveform(
                    binAr[1:sz, i],
                    1/params['sampleRates'][i+n],
                    params['names'][i+n]
                )

        if self.acquisitionType == 'galvo Path':
            self.timeShiftGalvoDriveWaveforms()

        # load camera data
        if self.acquisitionType.lower().find("camera") >= 0:
            self.cameras = {}
            if self.version == '0':
                self.cameras['flash'] = CameraData(name="flash", rig_data=self)
            elif self.version == '1':
                camera_param_files = [entry.name for entry in os.scandir(self.datapath) 
                        if entry.name.startswith('camera-parameters') and entry.is_file()]
                if camera_param_files:
                    for file in camera_param_files:
                        name = file.split('-')[-1].split('.')[0]
                        self.cameras[name.lower()] = CameraData(
                                name=name,
                                params_file=file, 
                                bin_file='frames-{0}.bin'.format(name), 
                                rig_data=self, 
                                data_dir=self.datapath)
            else:
                raise NotImplementedError('Unknown version number: {0}'.format(self.version))
    
    def __str__(self):
        return pprint.pformat(vars(self))

    def timeShiftGalvoDriveWaveforms(self):
        ''' Called for path scanning. The galvos
        lag behind the drive signals. But as the drive signal is
        periodic, this just amounts a phase shift.'''
        k = -np.round(self.offsetTime / self.ao['galvoX'].dt)
        self.ao['galvoX'].waveform = np.roll(self.ao['galvoX'].waveform, k)
        self.ao['galvoY'].waveform = np.roll(self.ao['galvoY'].waveform, k)
            

class CameraData():
    def __init__(
            self, 
            name:str, 
            params_file:str=None, 
            bin_file:str='frames-flash.bin', 
            data_dir:str=os.getcwd(), 
            rig_data:RigData=None, 
            dtype = np.uint16 ):
        self.name = name
        if not params_file:    # if no params file, params are assumed to reside in rig_data
            if not isinstance(rig_data, RigData):
                raise ValueError("Must supply either a valid parameter file or a RigData instance")
            data_dir = rig_data.datapath
            params = ['frameRate', 'binning', 'requestedFrames', 'exposure', 'roi', 'readoutSpeed', 'triggerMode']
            for p in params:
                if hasattr(rig_data, p):
                    setattr(self, p, getattr(rig_data, p))
        else:    # parameters are loaded from a file
            params_path = os.path.join(data_dir, params_file)
            if not os.path.exists(params_path):
                raise FileNotFoundError('Parameters could not be loaded. No such file: ' + params_path)
            params = readParamFile(params_path)
            for key in params:
                setattr(self, key, params[key])
        self.datapath = data_dir
        # check for required roi parameter
        if not hasattr(self, 'roi'):
            raise AttributeError('Missing required parameter "roi". Cannot load frames.')
        bin_path = os.path.join(data_dir, bin_file)
        # load frames
        self.frames = np.fromfile(bin_path, dtype=dtype).reshape((-1,
            params['roi'][3].astype(int),
            params['roi'][2].astype(int)))
        # set frame times
        self.setFrameTimes()

    def __str__(self):
        return pprint.pformat(vars(self))

    def setFrameTimes(self):
        n_frames = self.frames.shape[2]
        self.frametimes = np.linspace(0, n_frames/self.frameRate, n_frames, endpoint=False)

    @overload
    def setFrameTimesFromExposureOut(self, exposure_out: InstrumentWaveform):
        d = np.nonzero(np.diff(exposure_out.waveform) == True)[0]
        self.frametimes = exposure_out.dt * (d-1) - self.exposure
        # detect if the camera was driven by a square wave, and if so set frameRate
        if (d.mean() * 1E-5) > d.std():
            self.frameRate = 1/d.mean()
        
    @overload
    def setFrameTimesFromExposureOut(self, rig_data: RigData):
        if 'cameraExposureOut' in rig_data.di:
            self.setFrameTimesFromExposureOut(rig_data.di['cameraExposureOut'])
        else:
            raise ValueError("No such digital input 'cameraExposureOut'. Pass the appropiate digital input instead.")

    def toxarray(self):
        da = xr.DataArray(self.frames, dims=('x', 'y', 'time'), coords={'time': self.frametimes})
        da['x'].attrs['units'] = 'pixels'
        da['y'].attrs['units'] = 'pixels'
        da['time'].attrs['units'] = 'sec'
        da['time'].attrs['long_name'] = 'Time'
        return da
