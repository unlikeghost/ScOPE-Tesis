import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA

from scope.compressors import LZ77Compressor, GZIPCompressor, BZ2Compressor
from scope.matrix import MatrixFactoryV2 as MatrixFactory
from scope.models.ScopeOT import ScOPEOT
from scope.samples.sample_generator import generate_samples
from scope.utils.report_generation import make_report

np.random.seed(42)

FILE_NAME: str = 'clintox'
FILE_PATH: str = os.path.join('data', 'datasets', f'{FILE_NAME}.csv')
RESULTS_PATH: str = os.path.join('data', 'results')


STR_SEPARATOR: str = '\t'

MIN_SAMPLES: int = 3
MAX_SAMPLES: int = 30

COMPRESSION_DISTANCES_TO_EVALUATE: list = [
    'ncd',
    'cdm',
    'clm'
]

COMPRESSORS_FUNCTIONS_TO_EVALUATE: list = [
    LZ77Compressor(),
    GZIPCompressor(),
    BZ2Compressor()
]

BEST_SIGMA: List[bool] = [True, False]
USE_MATCHING_METHOD: List[bool] = [True, False]