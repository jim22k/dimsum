import numpy as np
import pandas as pd
from collections.abc import Mapping
from grblas import Vector
from .schema import Dimension, Schema
from .container import CodedArray, Flat


class DuplicateFrequencyError(Exception):
    pass


class CalendarDimension(Dimension):
    def __init__(self, name, periods, *, format=None, ordered=True):
        """
        Creates a new calendar dimension using pd.PeriodIndex to fully specify the frequency and range.
        The easiest way to create a pandas PeriodIndex is using pd.period_range.

        If format is specified, it will be used for the output string formatting of periods.

        :param name: str
        :param periods: pd.PeriodIndex
        :param format: str (optional)
        :return: Dimension
        """
        if not isinstance(periods, pd.PeriodIndex):
            raise TypeError(f'periods must be a pd.PeriodIndex, not {type(periods)}')

        if format is None:
            values = periods.to_native_types()
        else:
            values = periods.strftime(format)
        self.freq = periods.freqstr
        self._index = periods

        super().__init__(name, values, ordered=ordered)


class Calendar(Mapping):
    def __init__(self, schema: Schema):
        self._mappings = {}
        self._dimensions = {}
        self._schema = schema

        # Find all dimensions which are CalendarDimensions
        freqs = {}
        for dim in schema._dimensions:
            if isinstance(dim, CalendarDimension):
                if dim.freq in freqs:
                    raise DuplicateFrequencyError(f'Found more than one CalendarDimension with frequency={dim.freq}')
                self._dimensions[dim.name] = dim
                freqs[dim.freq] = dim.name

        self._build_mappings(schema, freqs)

    def __getitem__(self, name):
        return self._mappings[name]

    def __iter__(self):
        return iter(self._mappings)

    def __len__(self):
        return len(self._mappings)

    def _build_mappings(self, schema, freqs):
        from .methods import where

        map = self._mappings

        # Data object holds (name, dims, codes, values, dtype)
        data = []

        # Everything in _dimensions is a CalendarDimension
        for name, dim in self._dimensions.items():
            freq = dim.freq
            index = dim._index

            # Indices are ordered 1..n with 0 reserved for NULL
            indices = np.arange(1, len(dim.values))
            offset = schema.offset[name]
            codes = indices << offset

            # Build useful calendar information objects
            if freq == 'D':
                data.append((f'{name}.day', [name], codes, index.day, 'int8'))
                data.append((f'{name}.month', [name], codes, index.month, 'int8'))
                data.append((f'{name}.quarter', [name], codes, index.quarter, 'int8'))
                data.append((f'{name}.year', [name], codes, index.year, 'int32'))
                data.append((f'{name}.days_in_month', [name], codes, index.daysinmonth, 'int8'))
                data.append((f'{name}.days_in_year', [name], codes, index.asfreq('Y').dayofyear, 'int16'))
                iq = index.asfreq('Q')
                days_in_quarter = np.where(iq.quarter == 1, iq.dayofyear, iq.dayofyear - (iq - 1).dayofyear)
                data.append((f'{name}.days_in_quarter', [name], codes, days_in_quarter, 'int8'))
            elif freq == 'M':
                data.append((f'{name}.month', [name], codes, index.month, 'int8'))
                data.append((f'{name}.quarter', [name], codes, index.quarter, 'int8'))
                data.append((f'{name}.year', [name], codes, index.year, 'int32'))
                data.append((f'{name}.days_in_month', [name], codes, index.daysinmonth, 'int8'))
                data.append((f'{name}.days_in_year', [name], codes, index.asfreq('Y').dayofyear, 'int16'))
                iq = index.asfreq('Q')
                days_in_quarter = np.where(iq.quarter == 1, iq.dayofyear, iq.dayofyear - (iq - 1).dayofyear)
                data.append((f'{name}.days_in_quarter', [name], codes, days_in_quarter, 'int8'))
            elif freq == 'Q-DEC':
                data.append((f'{name}.quarter', [name], codes, index.quarter, 'int8'))
                data.append((f'{name}.year', [name], codes, index.year, 'int32'))
                data.append((f'{name}.days_in_year', [name], codes, index.asfreq('Y').dayofyear, 'int16'))
                days_in_quarter = np.where(index.quarter == 1, index.dayofyear, index.dayofyear - (index - 1).dayofyear)
                data.append((f'{name}.days_in_quarter', [name], codes, days_in_quarter, 'int8'))
            elif freq == 'A-DEC':
                data.append((f'{name}.year', [name], codes, index.year, 'int32'))
                data.append((f'{name}.days_in_year', [name], codes, index.dayofyear, 'int16'))

        # Wrap objects as CodedArrays
        for name, dims, codes, values, dtype in data:
            bitmask = schema.build_bitmask(dims)
            vec = Vector.from_values(codes, values, size=bitmask + 1, dtype=dtype)
            map[name] = CodedArray(Flat(vec, schema, dims))

        # Build all possible mappings
        d = freqs.get('D')
        m = freqs.get('M')
        q = freqs.get('Q-DEC')
        y = freqs.get('A-DEC')

        if d is not None and m is not None:
            criteria = (map[f'{d}.month'] == map[f'{m}.month']) & (map[f'{d}.year'] == map[f'{m}.year'])
            map[f'{d}->{m}'] = where(criteria, 1)
        if d is not None and q is not None:
            criteria = (map[f'{d}.quarter'] == map[f'{q}.quarter']) & (map[f'{d}.year'] == map[f'{q}.year'])
            map[f'{d}->{q}'] = where(criteria, 1)
        if d is not None and y is not None:
            criteria = map[f'{d}.year'] == map[f'{y}.year']
            map[f'{d}->{y}'] = where(criteria, 1)
        if m is not None and q is not None:
            criteria = (map[f'{m}.quarter'] == map[f'{q}.quarter']) & (map[f'{m}.year'] == map[f'{q}.year'])
            map[f'{m}->{q}'] = where(criteria, 1)
        if m is not None and y is not None:
            criteria = map[f'{m}.year'] == map[f'{y}.year']
            map[f'{m}->{y}'] = where(criteria, 1)
        if q is not None and y is not None:
            criteria = map[f'{q}.year'] == map[f'{y}.year']
            map[f'{q}->{y}'] = where(criteria, 1)
