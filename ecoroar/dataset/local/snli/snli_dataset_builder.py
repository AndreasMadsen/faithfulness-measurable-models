"""snli dataset."""

import json

import tensorflow_datasets as tfds


class LocalSNLI(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SNLI dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'premise': tfds.features.Text(),
                'hypothesis': tfds.features.Text(),
                'label': tfds.features.ClassLabel(
                    names=['entailment', 'neutral', 'contradiction']
                ),
            }),
            # No supervised key, as both question and answer has to be passed as input
            supervised_keys=None,
            homepage="https://nlp.stanford.edu/projects/snli/"
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        container_path = dl_manager.download_and_extract('https://nlp.stanford.edu/projects/snli/snli_1.0.zip')

        return {
            'test': self._generate_examples(container_path / 'snli_1.0' / 'snli_1.0_test.jsonl'),
            'validation': self._generate_examples(container_path / 'snli_1.0' / 'snli_1.0_dev.jsonl'),
            'train': self._generate_examples(container_path / 'snli_1.0' / 'snli_1.0_train.jsonl')
        }

    def _generate_examples(self, path):
        """This function returns the examples in the raw (text) form."""
        with path.open('r') as fp:
            for idx, line in enumerate(fp):
                observation = json.loads(line)
                if observation['gold_label'] == '-':
                    continue

                yield idx, {
                    'premise': observation['sentence1'],
                    'hypothesis': observation['sentence2'],
                    'label': observation['gold_label'],
                }
