"""mimic dataset."""

from dataclasses import dataclass
import re

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

@dataclass
class LocalMimicConfig(tfds.core.BuilderConfig):
    icd9_prefix_code: str = None
    hadm_ids_split_csv: str = None


class LocalMimic(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mimic dataset."""

    VERSION = tfds.core.Version('3.1.4')

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Make sure you have an account at https://doi.org/10.13026/C2XW26.
        Then download files with:

        wget -N -c -np --user $USER --ask-password https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz -O $PERSISTENT_DIR/mimic/noteevents.csv.gz
        wget -N -c -np --user $USER --ask-password https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz -O $PERSISTENT_DIR/mimic/diagnoses_icd.csv.gz
    """

    BUILDER_CONFIGS = [
        LocalMimicConfig(
            icd9_prefix_code='250.00',
            hadm_ids_split_csv='https://gist.githubusercontent.com/AndreasMadsen/529d3705c2836baebaa5c76512206162'
                               '/raw/9f6ebfe628ca69d614f3bb07784f9a4734c5be6e/diabetes.csv',
            name='diabetes',
            description=f"The diabetes task of the MIMIC-III dataset",
        ),
        LocalMimicConfig(
            icd9_prefix_code='285.1',
            hadm_ids_split_csv='https://gist.githubusercontent.com/AndreasMadsen/529d3705c2836baebaa5c76512206162'
                               '/raw/9f6ebfe628ca69d614f3bb07784f9a4734c5be6e/anemia.csv',
            name='anemia',
            description=f"The diabetes task of the MIMIC-III dataset",
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text(),
                'diagnosis': tfds.features.ClassLabel(
                    names=['negative', 'positive']
                ),
            }),
            supervised_keys=('text', 'diagnosis'),
            homepage='https://doi.org/10.13026/C2XW26',
            license="PhysioNet Credentialed Health Data License 1.5.0"
    )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Get relevant cases
        hadm_ids = pd.read_csv(
            dl_manager.download_and_extract(self.builder_config.hadm_ids_split_csv),
            header=None,
            engine='c',
            names=['HADM_ID', 'SPLIT'],
            index_col=[0])

        # Get labels
        df_icd9_codes = pd.read_csv(dl_manager.manual_dir / 'mimic' / 'diagnoses_icd.csv.gz',
                                    compression='gzip',
                                    usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
                                    engine='c') \
            .dropna() \
            .join(hadm_ids, on=['HADM_ID'], how='inner') \
            .assign(**{
                'ICD9_MATCH': lambda x: (
                    x['ICD9_CODE'].str.slice(0, 3) + '.' + x['ICD9_CODE'].str.slice(3)
                ).str.startswith(self.builder_config.icd9_prefix_code),
            }) \
            .groupby(['SUBJECT_ID', 'HADM_ID', 'SPLIT']) \
            .agg({
                'ICD9_MATCH': lambda col: np.logical_xor.reduce(col),
            }) \
            .reset_index(['SPLIT'])

        # Get discharge summary
        df_notes = pd.read_csv(dl_manager.manual_dir / 'mimic' / 'NOTEEVENTS.csv.gz',
                            compression='gzip',
                            usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'CHARTDATE', 'DESCRIPTION', 'TEXT'],
                            engine='c') \
            .dropna() \
            .query('`CATEGORY` == "Discharge summary"') \
            .join(df_icd9_codes, on=['SUBJECT_ID', 'HADM_ID'], how='inner') \
            .replace({'DESCRIPTION': {'Report' : 0, 'Addendum' : 1}}) \
            .sort_values(by=['DESCRIPTION', 'CHARTDATE']) \
            .groupby(['SUBJECT_ID', 'HADM_ID', 'SPLIT', 'ICD9_MATCH'], as_index=False) \
            .agg({
                'TEXT': lambda col: col.str.cat(sep=' ').strip(),
            })

        # Generate splits
        return {
            split: self._generate_examples(df_notes.query('`SPLIT` == @split'))
            for split in ['train', 'validation', 'test']
        }

    def _generate_examples(self, df):
        for _, row in df.iterrows():
            text = row['TEXT'].lower()
            text = re.sub(r'\[\s*\*\s*\*(.*?)\*\s*\*\s*\]', ' <DE> ', text)
            text = re.sub(r'([^a-zA-Z0-9])(\s*\1\s*)+', r'\1 ', text)
            text = re.sub(r'[^\s.,?]*\d[^\s.,?]*', ' [DIGITS] ', text)
            text = re.sub(r'\s+', ' ', text.strip())

            yield f"{row['SUBJECT_ID']}-{row['HADM_ID']}", {
                'text': text,
                'diagnosis': row['ICD9_MATCH'],
            }
