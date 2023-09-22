
import numpy as np
import tensorflow as tf

from ecoroar.transform import BucketedPaddedBatch


def _covert_to_dataset(sequence_lengths):
    return tf.data.Dataset.from_tensor_slices(sequence_lengths) \
        .map(lambda length: (
            {'input_ids': tf.range(1, length + 1),
             'attention_mask': tf.range(1, length + 1)},
            2
        ))


def _expected_batch(sequence_lengths, max_length, input_ids_fill=-1, attention_mask_fill=0):
    input_ids = [
        tf.concat([tf.range(1, length + 1), tf.fill([max_length - length], input_ids_fill)], axis=0)
        for length in sequence_lengths
    ]
    attention_mask = [
        tf.concat([tf.range(1, length + 1), tf.fill([max_length - length], attention_mask_fill)], axis=0)
        for length in sequence_lengths
    ]
    target = [2 for length in sequence_lengths]

    return (
        {'input_ids': tf.stack(input_ids),
         'attention_mask': tf.stack(attention_mask)},
        tf.stack(target)
    )


def _assert_identical_batch(actual, desired):
    np.testing.assert_array_equal(actual[0]['input_ids'], desired[0]['input_ids'])
    np.testing.assert_array_equal(actual[0]['attention_mask'], desired[0]['attention_mask'])
    np.testing.assert_array_equal(actual[1], desired[1])


def test_bounderies_detection_quantiles():
    dataset = _covert_to_dataset(tf.range(1, 101))
    batcher = BucketedPaddedBatch([dataset], batch_size=1)
    np.testing.assert_array_equal(batcher.bounderies, [25, 50, 75, 90, 100])

    batcher = BucketedPaddedBatch([dataset], batch_size=1, quantiles=[0.25, 0.5, 0.75, 0.9])
    np.testing.assert_array_equal(batcher.bounderies, [25, 50, 75, 90, 100])

    batcher = BucketedPaddedBatch([dataset], batch_size=1, quantiles=[0.4, 0.8])
    np.testing.assert_array_equal(batcher.bounderies, [40, 80, 100])


def test_bounderies_detection_batch_size():
    dataset = _covert_to_dataset([1, 2, 2, 2, 4, 4, 4, 4, 10, 10, 10, 10, 10, 100])
    batcher = BucketedPaddedBatch([dataset], batch_size=8)
    np.testing.assert_array_equal(batcher.bounderies, [4, 100])


def test_bounderies_are_unique():
    dataset = _covert_to_dataset([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    batcher = BucketedPaddedBatch([dataset], batch_size=2)
    np.testing.assert_array_equal(batcher.bounderies, [1, 2, 3])


def test_batching_cardinality_preserved():
    dataset = _covert_to_dataset([1, 2, 2, 2, 4, 4, 4, 4, 10, 10, 10, 10, 10, 100])
    batcher = BucketedPaddedBatch([dataset], batch_size=2)

    batch_2 = dataset.apply(batcher(2, padding_values=({'input_ids': -1, 'attention_mask': 0}, None)))
    np.testing.assert_equal(tf.data.experimental.cardinality(batch_2).numpy(), 7)

    batch_4 = dataset.apply(batcher(4, padding_values=({'input_ids': -1, 'attention_mask': 0}, None)))
    np.testing.assert_equal(tf.data.experimental.cardinality(batch_4).numpy(), 4)

    batch_8 = dataset.apply(batcher(8, padding_values=({'input_ids': -1, 'attention_mask': 0}, None)))
    np.testing.assert_equal(tf.data.experimental.cardinality(batch_8).numpy(), 2)


def test_batching_output():
    dataset = _covert_to_dataset([1, 2, 1, 3, 2, 2, 2, 4, 4, 4, 2, 6, 4, 8, 4, 7, 8, 8, 10])
    batcher = BucketedPaddedBatch([dataset], batch_size=2, quantiles=[0.2, 0.4, 0.8])
    np.testing.assert_array_equal(batcher.bounderies, [2, 4, 8, 10])

    batches = list(dataset
                   .apply(batcher(2, padding_values=({'input_ids': -1, 'attention_mask': 0}, None)))
                   )
    np.testing.assert_equal(len(batches), 10)

    # assert content and shape of each batch
    _assert_identical_batch(batches[0], _expected_batch([1, 2], 2))
    _assert_identical_batch(batches[1], _expected_batch([1, 3], 4))
    _assert_identical_batch(batches[2], _expected_batch([2, 2], 2))
    _assert_identical_batch(batches[3], _expected_batch([2, 4], 4))
    _assert_identical_batch(batches[4], _expected_batch([4, 4], 4))
    _assert_identical_batch(batches[5], _expected_batch([2, 6], 8))
    _assert_identical_batch(batches[6], _expected_batch([4, 8], 8))
    _assert_identical_batch(batches[7], _expected_batch([4, 7], 8))
    _assert_identical_batch(batches[8], _expected_batch([8, 8], 8))
    _assert_identical_batch(batches[9], _expected_batch([10], 10))
