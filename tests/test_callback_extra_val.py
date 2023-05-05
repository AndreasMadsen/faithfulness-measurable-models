
from ecoroar.callback import ExtraValidationDataset

import tensorflow as tf

def _create_dataset(rng, n_obs, w):
    w = tf.convert_to_tensor(w, dtype=tf.dtypes.float32)
    x = rng.normal([n_obs, tf.size(w)])
    y = x @ tf.expand_dims(w, axis=-1)
    return tf.data.Dataset.from_tensor_slices((x, y))

def test_callback_extra_validation_dataset():
    rng = tf.random.Generator.from_seed(0)

    model = tf.keras.Sequential([ tf.keras.layers.Dense(units=1) ])
    model.compile(
        optimizer='rmsprop',
        loss='mean_squared_logarithmic_error',
        metrics=['mean_squared_error', 'mean_absolute_error'],
        run_eagerly=False
    )

    dataset_train = _create_dataset(rng, 128, [0.5, 2, 0.3]).batch(32)
    dataset_valid_primary = _create_dataset(rng, 128, [0.5, 2, 0.3]).batch(32)
    dataset_valid_extra_1 = _create_dataset(rng, 128, [1, 1, 1]).batch(32)
    dataset_valid_extra_2 = _create_dataset(rng, 128, [2, 2, 2]).batch(32)

    history = model.fit(dataset_train, validation_data=dataset_valid_primary, epochs=2, callbacks=[
        ExtraValidationDataset(dataset_valid_extra_1, name='val_extra_1'),
        ExtraValidationDataset(dataset_valid_extra_2, name='val_extra_2')
    ], verbose=0)

    tf.debugging.assert_shapes([
        (history.history['loss'], ('E', )),
        (history.history['mean_squared_error'], ('E', )),
        (history.history['mean_absolute_error'], ('E', )),

        (history.history['val_loss'], ('E', )),
        (history.history['val_mean_squared_error'], ('E', )),
        (history.history['val_mean_absolute_error'], ('E', )),

        (history.history['val_extra_1_loss'], ('E', )),
        (history.history['val_extra_1_mean_squared_error'], ('E', )),
        (history.history['val_extra_1_mean_absolute_error'], ('E', )),

        (history.history['val_extra_2_loss'], ('E', )),
        (history.history['val_extra_2_mean_squared_error'], ('E', )),
        (history.history['val_extra_2_mean_absolute_error'], ('E', )),
    ])
