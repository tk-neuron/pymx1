import h5py
import pandas as pd


def get_df(hdfpath: str, index_datastore: int=0, offset: bool=True, legacy: bool=False):
    with h5py.File(hdfpath, 'r') as file:
        if not legacy:
            data = MaxOneHDF(file)
            datastores = data.list_datastore()  # list of datastores
            datastore = datastores[index_datastore]
            params = data.read_params(datastore)
            mapping = data.read_mapping(datastore)
            spikes = data.read_spikes(datastore)
        else:
            data = MaxOneHDFLegacy(file)
            params = data.read_params()
            mapping = data.read_mapping()
            spikes = data.read_spikes()

    df_map = pd.DataFrame(mapping)
    df_map['index_'] = df_map.index  # for reading signal
    df_sp = pd.DataFrame(spikes)
    df_sp.amplitude = - df_sp.amplitude * params['lsb'] * 1e6
    
    if offset:
        df_sp = offset_spiketime(df_sp=df_sp, fs=params['sampling'])

    return df_sp, df_map


def offset_spiketime(df_sp: pd.DataFrame, fs: int | float):
    """
    offset spike dataframe and convert the time scale to sec
    """
    df = df_sp.copy()
    df['spiketime'] = (df.frameno - df.frameno.min()) / fs
    return df


class __MaxOneHDF:
    def __init__(self, file):
        """
        file: opened file

        example: 
        > with h5py.File(hdfpath, 'r') as f:
        >     data = MaxOneHDF(f)
        """
        self.file = file

    def tree(self) -> None:
        """
        print the h5 file structure
        """
        self.file.visititems(self.__print_dataset)

    @staticmethod
    def __print_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)


class MaxOneHDF(__MaxOneHDF):
    # TODO: better design with idx_datastore, not datastore str itself
    def list_datastore(self) -> list[str]:
        """
        list available datastores
        """
        return list(self.file['data_store'])

    def dir(self, datastore):
        return f'data_store/{datastore}/'

    def dir_setting(self, datastore):
        return self.dir(datastore) + 'settings/'

    def dir_raw(self, datastore):
        return self.dir(datastore) + 'groups/routed/'

    def read_mapping(self, datastore):
        """
        read mapping data 
        """
        mapping = self.file[self.dir_setting(datastore) + 'mapping'][()]
        return mapping

    def read_spikes(self, datastore):
        """
        read online detected spikes data
        """
        spikes = self.file[self.dir(datastore) + 'spikes'][()]
        return spikes

    # TODO: multiply sig with lsb here?  --> amplitude
    def read_raw(self, datastore, idx_channels=None, start=None, end=None):
        """
        read raw signal data
        """
        dir = self.dir_raw(datastore)
        frames = self.file[dir + 'frame_nos'][start:end]
        channels = self.file[dir + 'channels'][idx_channels]
        sig = self.file[dir + 'raw'][idx_channels, start:end]
        return sig, frames, channels

    def read_params(self, datastore):
        """
        read parameters for recording: lsb, samping, spike_threshold
        """
        params = {}
        dir = self.dir_setting(datastore)
        params['lsb'] = self.file[dir + 'lsb'][0]
        params['sampling'] = self.file[dir + 'sampling'][0]
        params['spike_threshold'] = self.file[dir + 'spike_threshold'][0]
        return params


class MaxOneHDFLegacy(__MaxOneHDF):
    @staticmethod
    def _frame2frameno(frames):
        return frames[1] + frames[2] * 2 ** 16

    def read_mapping(self):
        mapping = self.file['mapping'][()]
        return mapping

    def read_spikes(self):
        spikes = self.file['proc0/spikeTimes'][()]
        return spikes

    def read_stim(self, start, end):
        frames = self.file['sig'][-3:, int(start):int(end)]
        idx = (frames[0] == 1)
        stim = self._frame2frameno(frames[:, idx])
        return stim

    def read_raw(self, channels, start, end):
        sig = self.file['sig'][channels, int(start):int(end)]
        frames = self.file['sig'][-3:, int(start):int(end)]
        frameno = self._frame2frameno(frames)
        return sig, frameno

    def read_params(self, sampling=20000):
        params = {}
        lsb = self.file['settings/lsb'][()][0]
        params['lsb'] = lsb
        params['sampling'] = sampling
        return params
