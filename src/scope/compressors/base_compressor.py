# -*- coding: utf-8 -*-
"""
compressor.py

This module provides the `Compressor` class, which implements a lightweight abstraction
for compression using supported compression libraries such as `gzip` and `bz2`.

Features:
- Dynamically imports the specified compression module (`gzip` or `bz2`).
- Allows compression of strings, numpy arrays, lists, or tuples into byte sequences.
- Provides callable usage to compress data in a simplified way.

Classes:
- Compressor: Handles the compression workflow, ensures only supported compressors
  are used, and provides detailed information about the original and compressed data.

Example Usage:
    # Instantiate the compressor with a supported method
    compressor = Compressor('gzip')
    print(compressor)

    # Compress a string
    result = compressor(sequence='Hola')
    print(result)

Supported Compression Libraries:
- gzip
- bz2

Encoding: UTF-8
Author: Jesus Alan Hernandez Galvan
Date: 01/09/2024
"""

import numpy as np
from typing import Union, Tuple


class BaseCompressor:

    def __init__(self, compressor_module, compressor_name: str):
        """
        Initialize the compressor base class with the given module and name.

        Parameters:
            compressor_module: Module (e.g., gzip or bz2) that provides compression functionality.
            compressor_name: str
                The name of the compression method being used (e.g., 'gzip' or 'bz2').
        """
        self._compressor = compressor_module
        self._compressor_name: str = compressor_name

    @property
    def get_compressor(self) -> str:
        """
             Specifies the name of the compressor associated with this property.

             Returns the value of the private attribute `_compressor_name`.

             Return:
                 str: The name of the compressor used in the implementation.
        """
        return self._compressor_name

    def compress(self, sequence: Union[str, np.ndarray, list, tuple]) -> Tuple:
        """
        Compresses a sequence of data into a compressed byte stream. The method supports
        sequences of type `str`, `np.ndarray`, `list`, and `tuple`. Strings will be
        encoded to UTF-8, whereas arrays, lists, and tuples will be converted into
        a byte representation before compression. Returns the original sequence,
        compressed byte stream, and the byte length of the compressed data.

        Parameters
        ----------
        sequence : Union[str, np.ndarray, list, tuple]
            The input sequence to be compressed. Valid types are str, np.ndarray, list,
            or tuple.

        Returns
        -------
        Tuple
            A tuple containing the original sequence, compressed byte data, and the
            length of the compressed data in bytes.

        Raises
        ------
        ValueError
            If the input sequence type is not one of str, np.ndarray, list, or tuple.
        """

        # if not isinstance(sequence, bytes) or (type(sequence) not in ["str", "np.ndarray", "list", "tuple"]):
        #     raise ValueError(f'Unsupported sequence type: {type(sequence)}')

        if isinstance(sequence, str):
            sequence: bytes = sequence.encode('utf-8')
        elif type(sequence) in [np.ndarray, list, tuple]:
            sequence: np.ndarray = np.array(sequence) if isinstance(sequence, list) else sequence
            sequence: bytes = sequence.tobytes()

        sequence_compressed: bytes = self._compressor.compress(sequence)

        return sequence, sequence_compressed, len(sequence_compressed)

    def __repr__(self) -> str:
        """
        __repr__

        Provides a string representation of the Compressor object.

        Returns
        -------
        str
            A string that includes the name of the compressor.
        """
        return f'Compressor({self._compressor_name})'

    def __call__(self, sequence: Union[str, np.ndarray, list, tuple]) -> Tuple:
        """
        Calls the object to compress the given sequence.

        This method allows the object to be called as a function to perform compression
        on a given sequence. The sequence can be provided as a string, numpy array,
        list, or tuple, and the method will return the compressed form.

        Parameters:
        sequence: Union[str, np.ndarray, list, tuple]
            The sequence to be compressed. It should be of type string, numpy array,
            list, or tuple.

        Returns
        -------
        Tuple
            A tuple containing the original sequence, compressed byte data, and the
            length of the compressed data in bytes.
        """
        return self.compress(sequence)

