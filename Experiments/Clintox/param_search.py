import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import deepchem as dc
from sklearn.model_selection import train_test_split

from scope.utils import ScOPEOptimizerBayesian
from scope.utils.sample_generation import SampleGenerator


seed: int = 42
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['figure.max_open_warning'] = 0
np.random.seed(seed)

tasks, datasets, _ = dc.molnet.load_clintox(featurizer='Raw')
train, valid, test = datasets

full_dataset = dc.data.DiskDataset.merge([train, valid, test])

smiles_list = full_dataset.ids
labels = full_dataset.y.astype(int)

print("Número de muestras:", len(smiles_list))
print("Forma de etiquetas:", labels.shape)
print("Tareas:", tasks)


df = pd.DataFrame({
    "smiles": smiles_list,
    "fda_approved": labels[:, 0],
    "ct_tox": labels[:, 1]
})


print(df.head())
print("\nDistribución de clases - FDA_APPROVED:")
print(df["fda_approved"].value_counts())
print("\nDistribución de clases - CT_TOX:")
print(df["ct_tox"].value_counts())


STUDY_NAME: str = 'Clintox'
RESULTS_PATH: str = os.path.join('results')
ANALYSYS_RESULTS_PATH: str = os.path.join(RESULTS_PATH, 'Optimization')
SMILES_COLUMN: str = 'smiles'
LABEL_COLUMN: str = 'fda_approved'

TEST_SAMPLES:list = [
    3, 5, 10, 15, 20, 30, 50
]

TIMEOUT: int = 350 -> 690
TRIALS: int = 3000

x_test, x_search, y_test, y_search =  train_test_split(
    df['smiles'].values,
    df[LABEL_COLUMN].values,
    train_size=.70,
    random_state=seed,
    stratify=df[LABEL_COLUMN].values
)

print(x_test.shape, y_test.shape, x_search.shape, y_search.shape)

for sample_size in TEST_SAMPLES:
    
    generator = SampleGenerator(
        data=x_search,
        labels=y_search,
        seed=seed,
    )
    
    output_path: str = f'{RESULTS_PATH}/{sample_size}'

    optimizer = ScOPEOptimizerBayesian(
        n_trials=TRIALS,
        timeout=TIMEOUT,
        target_metric='combined',
        study_name=f'{STUDY_NAME}_Samples_{sample_size}',
        output_path=output_path
    )
    
    all_x = []
    all_y = []
    all_kw = []
    
    for test_x, test_y, kw_samples in generator.generate(num_samples=sample_size):
        all_x.append(test_x)
        all_y.append(test_y)
        all_kw.append(kw_samples)
    
    
    study = optimizer.optimize(all_x, all_y, all_kw)
        

    optimizer.save_complete_analysis(top_n=200)
