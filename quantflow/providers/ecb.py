import os
import tempfile
import csv
import zipfile
from io import StringIO
from datetime import date

from ccy import currency
from pq import api


usd = currency('USD')


@api.job('ecb.forex.history')
def download(self):
    url = 'http://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip'
    response = self.http.get(url)
    fp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(response.content)
        with zipfile.ZipFile(fp.name, 'r') as zfp:
            for m in zfp.infolist():
                # read each file in zip file
                with zfp.open(m.filename, 'r') as f:
                    text = StringIO(f.read().decode('utf-8'))
    finally:
        if fp:
            try:
                os.unlink(fp.name)
            except FileNotFoundError:
                pass

    reader = csv.DictReader(text)
    write_ecb_data(self, reader)


def write_ecb_data(self, reader, handler):
    for d in reader:
        dt = ecb_date(d['Date'])
        dollar = float(d[usd.code])
        handler('EUR', dt, dollar)
        for ccy, v in d.items():
            if ccy == usd.code or len(ccy) != 3:
                continue
            try:
                cobj = currency(ccy)
                cu = float(v)/usd
                if cobj.order < usd.order:
                    cu = 1./cu
            except Exception:
                continue
            handler(ccy, dt, cu)


def ecb_date(dstr):
    '''
    convert ecb string date into python date
    '''
    bits = dstr.split('-')
    year = int(bits[0])
    month = int(bits[1])
    day = int(bits[2])
    return date(year, month, day)
