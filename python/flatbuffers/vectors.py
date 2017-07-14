# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Indexer classes that provide a Pythonic interface to vectors """

from . import compat
from . import number_types as N

import numpy as np


class VectorIndexer(object):
    """ Base class that type-specific vector Indexers inherit from """

    __slots__ = ["_table", "_offset", "_length", "_stride"]

    def __init__(self, table, offset, stride):
        """ Creates the base vector

        Parameters:
            table  - The table this vector is part of (a flatbuffers.Table object)
            offset - The offset in the table to the vector we're accessing
            stride - The distance between each element of the vector in bytes
        """

        self._table = table
        self._offset = offset
        self._length = table.VectorLen(offset)
        self._stride = stride

    def __iter__(self):
        """ Return an iterator for this vector """
        return VectorIterator(self)

    def __getitem__(self, index):
        """ Return either one value or a slice of this vector """
        if isinstance(index, int):
            if index < 0 or index >= self._length:
                raise IndexError("vector index %d out of range" % index)
            return self._unpack_index(index)
        elif isinstance(index, slice):
            return self._unpack_slice_index(index)
        else:
            raise TypeError("vector indices must be integers or slices, not %s" %
                            type(index).__name__)

    def __len__(self):
        """ Returns the length of this vector """
        return self._length

    def _unpack_index(self, index):
        """ To be implemented by child classes """
        raise NotImplementedError

    def _unpack_slice_index(self, slice_index):
        """
        A naive implementation is provided, but should be subclassed
        for speed where possible.
        """
        return [self._unpack_index(i)
                for i in compat.range_func(slice_index.indices(self._length))]


class StructVectorIndexer(VectorIndexer):
    """ Indexer for vectors of structs """

    __slots__ = ["_struct_class"]

    def __init__(self, table, offset, stride, struct_class):
        super(StructVectorIndexer, self).__init__(table, offset, stride)
        self._struct_class = struct_class

    def _unpack_index(self, index):
        a = self._table.Vector(self._offset) + index * self._stride
        item = self._struct_class()
        item.Init(self._table.Bytes, a)
        return item


class SimpleVectorIndexer(VectorIndexer):
    """ Handles vectors of base (number and boolean) types """

    __slots__ = ["_flags"]

    def __init__(self, table, offset, flags):
        super(SimpleVectorIndexer, self).__init__(table, offset, flags.bytewidth)
        self._flags = flags

    def _unpack_index(self, index):
        a = self._table.Vector(self._offset) + index * self._stride
        return self._table.Get(self._flags, offset)

    def _unpack_slice_index(self, index):
        pass


class StringVectorIndexer(VectorIndexer):
    """ Handles vectors of strings """

    def __init__(self, table, offset, stride):
        super(SimpleVectorIndexer, self).__init__(table, offset, stride)

    def _unpack_index(self, index):
        a = self._table.Vector(self._offset) + index * self._stride
        return self._table.String(a)


class VectorIterator(object):
    """ Iterator used to iterator over a VectorIndexer """

    __slots__ = ["_vec", "_next_index", "_length"]

    def __init__(self, vector_Indexer):
        self._vec = vector_Indexer
        self._next_index = 0
        self._length = len(vector)

    def __iter__(self):
        return self

    def next(self):
        """ Returns the next item in the vector """
        if self._next_index >= self._length:
            raise StopIteration

        item = self._vec[self._next_index]
        self._next_index += 1
        return item
