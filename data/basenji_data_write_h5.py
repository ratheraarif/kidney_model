#!/usr/bin/env python
# Copyright 2024
#
# Write HDF5 files for batches of model sequences for PyTorch.

from optparse import OptionParser
import os
import sys
import h5py
import numpy as np
import pdb
import pysam

from basenji_data_h5 import ModelSeq
from dna_io import dna_1hot_index

def main():
    usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <h5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='decimals',
        default=None, type='int',
        help='Round values to given decimals [Default: %default]')
    parser.add_option('-s', dest='start_i',
        default=0, type='int',
        help='Sequence start index [Default: %default]')
    parser.add_option('-e', dest='end_i',
        default=None, type='int',
        help='Sequence end index [Default: %default]')
    parser.add_option('-u', dest='umap_npy',
        help='Unmappable array numpy file')
    parser.add_option('--umap_clip', dest='umap_clip',
        default=1, type='float',
        help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
    parser.add_option('-x', dest='extend_bp',
        default=0, type='int',
        help='Extend sequences on each side [Default: %default]')
    (options, args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide input arguments.')
    else:
        fasta_file = args[0]
        seqs_bed_file = args[1]
        seqs_cov_dir = args[2]
        h5_file = args[3]

    # Read model sequences
    model_seqs = []
    for line in open(seqs_bed_file):
        a = line.split()
        model_seqs.append(ModelSeq(a[0], int(a[1]), int(a[2]), None))

    if options.end_i is None:
        options.end_i = len(model_seqs)
    num_seqs = options.end_i - options.start_i

    # Determine sequence coverage files
    seqs_cov_files = []
    ti = 0
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
    while os.path.isfile(seqs_cov_file):
        seqs_cov_files.append(seqs_cov_file)
        ti += 1
        seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)

    if len(seqs_cov_files) == 0:
        print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
        exit(1)

    seq_pool_len = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
    num_targets = len(seqs_cov_files)

    # Read targets
    targets = np.zeros((num_seqs, seq_pool_len, num_targets), dtype='float16')
    for ti in range(num_targets):
        seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
        targets[:, :, ti] = seqs_cov_open['targets'][options.start_i:options.end_i, :]
        seqs_cov_open.close()

    # Modify unmappable
    if options.umap_npy is not None and options.umap_clip < 1:
        unmap_mask = np.load(options.umap_npy)
        for si in range(num_seqs):
            msi = options.start_i + si
            seq_target_null = np.percentile(targets[si], q=[100 * options.umap_clip], axis=0)[0]
            targets[si, unmap_mask[msi, :], :] = np.minimum(targets[si, unmap_mask[msi, :], :], seq_target_null)
    elif options.umap_npy is not None:
        unmap_mask = np.load(options.umap_npy)
    else:
        unmap_mask = None

    # Open FASTA
    fasta_open = pysam.Fastafile(fasta_file)

    # Prepare arrays for HDF5
    seq_length = model_seqs[0].end - model_seqs[0].start + 2 * options.extend_bp
    sequences = np.zeros((num_seqs, seq_length), dtype='uint8')
    coords = []

    for si in range(num_seqs):
        msi = options.start_i + si
        mseq = model_seqs[msi]
        mseq_start = mseq.start - options.extend_bp
        mseq_end = mseq.end + options.extend_bp
        seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
        seq_1hot = dna_1hot_index(seq_dna)
        sequences[si, :] = seq_1hot
        coords.append((mseq.chr, mseq.start, mseq.end))

    fasta_open.close()

    # Optionally round targets
    if options.decimals is not None:
        targets = np.around(targets.astype('float32'), decimals=options.decimals).astype('float16')

    # Write HDF5
    with h5py.File(h5_file, 'w') as h5_out:
        h5_out.create_dataset('sequence', data=sequences, compression='gzip')
        h5_out.create_dataset('target', data=targets, compression='gzip')
        if unmap_mask is not None:
            h5_out.create_dataset('umap', data=unmap_mask[options.start_i:options.end_i, :], compression='gzip')
        # Store coordinates as strings
        dt = h5py.string_dtype(encoding='utf-8')
        coord_arr = np.array(['%s:%d-%d' % (c[0], c[1], c[2]) for c in coords], dtype=dt)
        h5_out.create_dataset('coords', data=coord_arr)


def fetch_dna(fasta_open, chrm, start, end):
    seq_len = end - start
    seq_dna = ''
    if start < 0:
        seq_dna = 'N' * (-start)
        start = 0
    seq_dna += fasta_open.fetch(chrm, start, end)
    if len(seq_dna) < seq_len:
        seq_dna += 'N' * (seq_len - len(seq_dna))
    return seq_dna

if __name__ == '__main__':
    main() 