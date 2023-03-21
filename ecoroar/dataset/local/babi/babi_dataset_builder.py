"""babi dataset."""

from dataclasses import dataclass

import tensorflow_datasets as tfds

_PATHS = {
    "qa1": {
        "train": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt",
    },
    "qa2": {
        "train": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_test.txt",
    },
    "qa3": {
        "test": "tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt",
    }
}

@dataclass
class LocalBabiConfig(tfds.core.BuilderConfig):
    task_no: int = None

class LocalBabi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for babi dataset."""

    VERSION = tfds.core.Version("1.2.0")

    BUILDER_CONFIGS = [
        LocalBabiConfig(
            task_no=task_no,
            name=f'en-10k/qa{task_no}',
            description=f"The 'qa{task_no}' task of the bAbI 'en-10k' dataset",
        )
        for task_no in range(1, 4)
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'paragraph': tfds.features.Text(),
                'question': tfds.features.Text(),
                'answer': tfds.features.ClassLabel(
                    names=['garden', 'hallway', 'kitchen', 'office', 'bedroom', 'bathroom']
                ),
            }),
            # No supervised key, as both question and answer has to be passed as input
            supervised_keys=None,
            homepage="https://research.fb.com/downloads/babi/",
            license="Creative Commons Attribution 3.0 License"
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        container_path = dl_manager.download_and_extract('http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')

        return {
            split: self._generate_examples(container_path / filepath)
            for split, filepath in _PATHS[f'qa{self.builder_config.task_no}'].items()
        }

    def _generate_examples(self, path):
        story = []
        story_id = 0
        paragraph_id = 0
        with path.open('r') as fp:
            for line in fp:
                tid, line_tid_striped = line.rstrip('\n').split(' ', 1)
                line_data = line_tid_striped.split('\t')

                # Start of a new paragraph construction
                if tid == '1':
                    story = []
                    story_id += 0

                # paragraph component
                if len(line_data) == 1:
                    story.append(line_data[0].strip())
                # question
                else:
                    yield f'{story_id}-{paragraph_id}', {
                        "paragraph": ' '.join(story),
                        "question": line_data[0].strip(),
                        "answer": line_data[1].strip()
                    }
                    paragraph_id += 1
