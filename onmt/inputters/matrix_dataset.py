# -*- coding: utf-8 -*-

import os

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase

# domain specific dependencies
try:
    import numpy as np
except ImportError:
    np = None


class MatrixDataReader(DataReaderBase):
    """Read matrix data from disk."""

    def __init__(self):
        self._check_deps()

    @classmethod
    def from_opt(cls, opt):
        return cls()

    @classmethod
    def _check_deps(cls):
        if any([np is None]):
            cls._raise_missing_dep(
                "numpy")

    def read(self, matrix_files, side, data_dir=None):
        """Read data into dicts.

        Args:
            side: "src"

        Yields:
            a dictionary containing matrix data, path and index for each line.
        """
        if isinstance(matrix_files, str):
            matrix_files = DataReaderBase._read_file(matrix_files)

        for i, filename in enumerate(matrix_files):
            filename = filename.decode("utf-8").strip()
            mat_path = os.path.join(data_dir, filename)
            if not os.path.exists(mat_path):
                mat_path = filename

            assert os.path.exists(mat_path), \
                'matrix path %s not found' % filename

            mat = torch.from_numpy(np.loadtxt(mat_path))

            yield {side: mat, side + '_path': filename, 'indices': i}


def mat_sort_key(ex):
    """Sort using the number of columns in the sequence."""
    return ex.src.size(0)


class MatrixField(Field):
    """Defines an matrix datatype and instructions for converting to Tensor.

    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=True, batch_first=False, pad_index=0,
                 dtype=torch.float, is_target=False):
        super(MatrixField, self).__init__(
            sequential=False, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=dtype,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=None, unk_token=None,
            pad_first=False, truncate_first=False, stop_words=None,
            is_target=is_target
        )
        self.pad_index = pad_index

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.

        Args:
            minibatch (List[torch.FloatTensor]): A list of matrix data, each
                having shape len x n_feats where len is variable.

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape ``(batch_size, max_len, n_feats)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length
        minibatch = list(minibatch)
        # minibatch[0] with shape len x n_feats
        lengths = [x.size(0) for x in minibatch]
        max_len = max(lengths)
        nfft = minibatch[0].size(1)
        matrixs = torch.full((len(minibatch), max_len, nfft), self.pad_index)
        for i, (mat, len_) in enumerate(zip(minibatch, lengths)):
            matrixs[i, 0:len_, :] = mat
        if self.include_lengths:
            return (matrixs, lengths)
        return matrixs

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.

        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True. Examples have shape
                ``(batch_size, n_feats, max_len)`` if `self.batch_first`
                else ``(max_len, batch_size, 1, n_feats)``.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if not self.batch_first:
            arr = arr.permute(1, 0, 2)
        arr = arr.contiguous()
        arr = arr.to(device)
        if self.include_lengths:
            return arr, lengths
        return arr


def matrix_fields(**kwargs):
    # mat = MatrixField(
    #     include_lengths=kwargs['include_lengths'], batch_first=False, 
    #     pad_index=0, dtype=torch.float)
    mat = MatrixField(
        include_lengths=True, batch_first=False, 
        pad_index=0, dtype=torch.float)
    return mat
