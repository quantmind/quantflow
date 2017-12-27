import os
from datetime import date

import tables as tb

from ccy.dates import date2yyyymmdd

from .models import Trade


stores = {}


def store(security, exchange, model,
          dt=None, type=None, path=None, version=None,
          filters=None, mode='r'):
    '''High level function to create or retrieve an Order Book

    :param security: The security or filename associated with the Data file
    :param dt: Optional date (as a datetime.date object)
    :param type: Data file type
    :param path: Optional path
    :param version: Order book version. Not used at the moment.
    :param mode: File opening mode, one of 'r', 'w', 'a'
    :param download: Optional callback for download
    :return: a :class:`.DataFile` object
    '''
    security = str(security)
    dt = dt or date.today()
    Store = stores[exchange]
    if isinstance(dt, date):
        dt = date2yyyymmdd(dt)
    filename = '%s_%s_%d.h5' % (security, exchange, dt)
    if path:
        filename = os.path.join(path, filename)

    if hasattr(type, 'data_type'):
        type = type.data_type

    if os.path.isfile(filename):
        return Store(tb.open_file(filename, mode))

    else:
        # A new book
        if mode == 'r':
            raise ValueError('Cannot open book in read mode, "%s" '
                             'is not available' % filename)
        ds = Store(tb.open_file(filename, 'w'))
        ds.initialise(security, dt, filters)
        return ds


class DataFileMeta(type):
    '''Metaclass for a :class:`.DataFile`

    It simply registers a new :class:`.DataFile` class with the
    ``order_books`` dictionary.
    '''
    def __new__(cls, name, bases, attrs):
        c = super().__new__(cls, name, bases, attrs)
        c.data_type = name.lower()
        stores[c.data_type] = c
        return c


class Store(metaclass=DataFileMeta):
    """
    Base class for Quote files. It is subclassed by QuoteFile in lob.py.
    """
    # Field to be used for sorted iteration of the Quote data
    table_description = None
    sort_by_field = ''
    description = ''
    version = None

    def __init__(self, h5):
        self.__dict__['_h5'] = h5
        self.__dict__['_rows_written'] = 0

    def __repr__(self):
        return self.filename
    __str__ = __repr__

    @property
    def filename(self):
        return self._h5.filename

    @property
    def opened(self):
        '''Check if the file is open or not
        '''
        return bool(self._h5 and self._h5.isopen)

    def initialise(self, filters):
        """
        Initialises the Quote h5 file. This is called only when a new h5 is
        created to store the data from the raw files, for a specific contract.
        """
        if filters is None:
            filters = self.filters()
        self._h5.create_table('/', 'trades', Trade, filters=filters)

    def add(self, table, data):
        table = self.get_table(table)
        entry = table.row
        for key, value in data.items():
            entry[key] = value
        entry.append()

    def flush(self):
        if self.opened:
            self._h5.flush()

    def close(self):
        if self.opened:
            self._rows_written = 0
            self._h5.flush()
            self._h5.close()
