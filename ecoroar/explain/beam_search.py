
from typing import Tuple, Union

import tensorflow as tf

from ..types import TokenizedDict
from ..transform import SequenceIndentifier
from ._importance_measure import ImportanceMeasureBatch
from ._util_evaluate import BatchEvaluator
from ._util_batch_parallel import batch_parallel

BeamType = Tuple[Union[tf.RaggedTensor, tf.Tensor],
                 Union[tf.RaggedTensor, tf.Tensor],
                 Union[tf.RaggedTensor, tf.Tensor]]

@tf.function(reduce_retracing=True)
def _candiates_added(existing_candidates, maybe_candidates, aux_data):
    is_new_candidate = tf.vectorized_map(
        lambda maybe_candidate: tf.math.reduce_all(
            tf.math.reduce_any(
                existing_candidates != tf.expand_dims(maybe_candidate, 0),
                axis=1),
            axis=0),
        maybe_candidates)

    selector = tf.squeeze(tf.where(is_new_candidate), axis=1)

    return (
        tf.gather(maybe_candidates, selector),
        tf.gather(aux_data, selector)
    )

@tf.function(reduce_retracing=True)
def _candiates_expand(beam):
    root_candidates, removal_order, score = beam
    root_candidates_n, max_sequence_length = tf.unstack(tf.shape(root_candidates), num=2)

    # This function needs to support empty arrays
    if root_candidates_n == 0:
        return (
            tf.zeros_like(root_candidates),
            tf.zeros((0, tf.shape(removal_order)[1] + 1), dtype=removal_order.dtype),
            tf.zeros_like(score)
        )

    # Encode root candiates as sparse tensors. This is used as a set data structure.
    # The T-dimentional indice will encode the mask vector.
    total_candidates = tf.zeros((0, max_sequence_length), dtype=tf.dtypes.bool)
    total_removal_order_ta = tf.TensorArray(removal_order.dtype, size=root_candidates_n, element_shape=(None, None))
    total_score_ta = tf.TensorArray(score.dtype, size=root_candidates_n, element_shape=(None, ))

    # For each candidate
    for root_candidate_i in tf.range(tf.shape(root_candidates)[0]):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(total_candidates, tf.TensorShape([None, None]))]
        )

        # Identify which tokens to mask
        root_candidate = root_candidates[root_candidate_i, :]
        token_idx_to_mask = tf.squeeze(tf.cast(tf.where(root_candidate), dtype=removal_order.dtype), axis=1)

        # Expand the root_candidate with masking options
        maybe_candidates_n = tf.size(token_idx_to_mask)
        maybe_candidates = tf.tensor_scatter_nd_update(
            tf.repeat(tf.expand_dims(root_candidate, 0), maybe_candidates_n, axis=0),
            tf.stack([tf.range(maybe_candidates_n, dtype=token_idx_to_mask.dtype), token_idx_to_mask], axis=1),
            tf.zeros((maybe_candidates_n, ), dtype=tf.dtypes.bool)
        )

        # Identify the candidates that should be added
        added_candidates, added_token_idx_to_mask = _candiates_added(
            total_candidates, maybe_candidates, token_idx_to_mask)
        added_candidates_n = tf.shape(added_candidates)[0]

        # convert added_token_idx_to_mask to added_removal_order
        # I.e. removal_order[i] = [1,2] and added_token_idx_to_mask = [3,4]
        # added_removal_order = [[1,2,3], [1,2,4]]
        added_removal_order = tf.concat((
            tf.repeat(tf.expand_dims(removal_order[root_candidate_i, :], axis=0),
                      added_candidates_n,
                      axis=0),
            tf.expand_dims(added_token_idx_to_mask, axis=1)
        ), axis=1)

        # copy score, such it can be incremeted later
        added_score = tf.repeat(score[root_candidate_i], added_candidates_n, axis=0)

        # collect candidates
        total_candidates = tf.concat((total_candidates, added_candidates), axis=0)
        total_removal_order_ta = total_removal_order_ta.write(root_candidate_i, added_removal_order)
        total_score_ta = total_score_ta.write(root_candidate_i, added_score)

    # Return new beam, with old scores.
    return (
        total_candidates,
        total_removal_order_ta.concat(),
        total_score_ta.concat()
    )

@tf.function(reduce_retracing=True)
def _beam_select(beam_score, beam_size):
    return tf.argsort(beam_score, stable=True, direction='DESCENDING')[:beam_size]

class BeamSearch(ImportanceMeasureBatch):
    _name = 'beam-sign'
    _defer_jit = True
    _default_beam_size = None

    def __init__(self, *args, beam_size: int=None, debugging = False, validate = True,
                 run_eagerly: bool = False, jit_compile: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, run_eagerly=run_eagerly, jit_compile=jit_compile, **kwargs)

        if beam_size is None:
            beam_size = self._default_beam_size
        if beam_size is None:
            raise ValueError('beam_size should be set')

        self._sequence_identifier = SequenceIndentifier(self._tokenizer)
        self._evaluate = BatchEvaluator(self._model, batch_size=self._inference_batch_size, run_eagerly=run_eagerly, jit_compile=jit_compile)

        self._beam_size = tf.convert_to_tensor(beam_size, dtype=tf.dtypes.int32)
        self._debugging = debugging
        self._validate = validate

        self._wrap_candiates_expand = batch_parallel(self._dataset_batch_size, validate=self._debugging)(_candiates_expand)
        self._wrap_beam_select = batch_parallel(self._dataset_batch_size, validate=self._debugging)(_beam_select)

    def _debug(self, interation_i, beam, x_beam_flatten, new_score):
        if self._debugging:
            input_ids = tf.RaggedTensor.from_row_lengths(x_beam_flatten['input_ids'], row_lengths=beam[0].row_lengths())

            print(f'iteration: {interation_i}')
            for obs_i in tf.range(new_score.nrows()):
                print(f'  overvation: {obs_i}')
                for input_ids_p, removal_order_p, score_p in zip(input_ids[obs_i], beam[1][obs_i], new_score[obs_i]):
                    print(
                        '    ' +
                        ' '.join(map(str, input_ids_p.numpy().tolist())) + ' | ' +
                        ''.join(map(str, removal_order_p.numpy().tolist())) + ':' +
                        str(score_p.numpy().tolist())
                    )

    def _create_masked_inputs(self, x: TokenizedDict, beam: BeamType, maskable_tokens: tf.Tensor) -> TokenizedDict:
        mask_pattern, _, _ = beam

        x_repeat = tf.nest.map_structure(
            lambda item: tf.gather(item, mask_pattern.value_rowids()),
            x)

        x_repeat['input_ids'] = tf.where(
            tf.logical_and(tf.expand_dims(maskable_tokens, 1), tf.logical_not(mask_pattern)).merge_dims(0, 1),
            self._tokenizer.mask_token_id,
            x_repeat['input_ids'])

        return x_repeat

    # STEP: 0 - intialize
    # maskable       : order, AUC score
    # [0, 1, 1, 1, 0]: ([], 0)

    # STEP: 1.a - expand
    # maskable       : order, AUC score
    # [0, 1, 1, 0, 0]: ([3],  0.3)
    # [0, 1, 0, 1, 0]: ([2],  0.2)
    # [0, 0, 1, 1, 0]: ([1],  0.1)

    # Implement this as a progressive expansion, and reduce simultaniusely.
    # STEP: 2.a - expand
    # maskable       : order, AUC score
    # [0, 0, 1, 0, 0]: ([3, 1],  ?)
    # [0, 1, 0, 0, 0]: ([3, 2],  ?)
    # [0, 0, 0, 1, 0]: ([2, 1],  ?)
    # [0, 1, 0, 0, 0]: ([2, 3],  ?)
    # [0, 0, 0, 1, 0]: ([1, 2],  ?)
    # [0, 0, 1, 0, 0]: ([1, 3],  ?)

    # STEP: 2.b - reduce
    # maskable       : order, AUC score
    # [0, 0, 1, 0, 0]: ([3, 1],  ?)
    # [0, 1, 0, 0, 0]: ([3, 2],  ?)
    # [0, 0, 0, 1, 0]: ([2, 1],  ?)

    # This is not beamed in the traditional sense, so the compute is O(B*N*N).
    # STEP: 2.c - evaluate
    # maskable       : order, AUC score
    # [0, 0, 1, 0, 0]: ([3, 1],  0.5)
    # [0, 1, 0, 0, 0]: ([3, 2],  0.4)
    # [0, 0, 0, 1, 0]: ([2, 1],  0.6)

    # STEP: 2.d - beam crop
    # maskable       : order, AUC score
    # [0, 0, 0, 1, 0]: ([2, 1],  0.6)
    # [0, 0, 1, 0, 0]: ([3, 1],  0.5)
    # [0, 1, 0, 0, 0]: ([3, 2],  0.4)

    # STEP: 3.a - expand
    # maskable       : order, AUC score
    # [0, 0, 0, 0, 0]: ([2, 1, 3], ?)
    # [0, 0, 0, 0, 0]: ([3, 1, 2], ?)
    # [0, 0, 0, 0, 0]: ([3, 2, 1], ?)

    # STEP: 3.b - reduce
    # maskable       : order, AUC score
    # [0, 0, 0, 0, 0]: ([2, 1, 3], ?)

    # STEP: 3.c - evaluate
    # maskable       : order, AUC score
    # [0, 0, 0, 0, 0]: ([2, 1, 3], 0.8)

    # STEP: 3.d - beam crop
    # maskable       : order, AUC score
    # [0, 0, 0, 0, 0]: ([2, 1, 3], 0.8)

    def _explain_batch(self, x, y):
        input_ids = x['input_ids']
        batch_size, max_sequence_length = tf.unstack(tf.shape(input_ids), num=2)

        # Since only explanation w.r.t. the first sequence are considred, only attempt
        # measures on the first sequence.
        maskable_tokens = self._sequence_identifier(input_ids) == 1
        # For masked inputs, [MASK] would be replaced with [MASK].
        # This enforces zero attribution score. Therefore this can be optimized by
        #   skipping the model evaluation.
        maskable_tokens = tf.logical_and(maskable_tokens, input_ids != self._tokenizer.mask_token_id)

        # Compute baseline prediction
        y_baseline = self._evaluate.single(x, y)

        # 0. Intialize beam.
        # Assumed to be sorted highest score to lowest score.
        # Note, the beam is not a tuble here, because tuples do not interact
        #   with set_loop_options(shape_invariants=[])
        beam_candidates = tf.RaggedTensor.from_row_lengths(
            maskable_tokens,
            row_lengths=tf.ones(batch_size, dtype=tf.dtypes.int64),
            validate=self._validate
        )
        beam_removal_order = tf.RaggedTensor.from_row_lengths(
            tf.zeros((batch_size, 0), dtype=tf.dtypes.int32),
            row_lengths=tf.ones(batch_size, dtype=tf.dtypes.int64),
            validate=self._validate
        )
        beam_score = tf.RaggedTensor.from_row_lengths(
            tf.zeros((batch_size, ), dtype=y_baseline.dtype),
            row_lengths=tf.ones(batch_size, dtype=tf.dtypes.int64),
            validate=self._validate
        )

        explanations = tf.zeros((batch_size, max_sequence_length), dtype=tf.dtypes.float32)
        max_iterations = tf.math.reduce_sum(tf.cast(maskable_tokens, tf.dtypes.int32), axis=1)

        for iteration_i in tf.range(tf.math.reduce_max(max_iterations)):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(beam_candidates,          tf.TensorShape([None, None, None])),
                                  (beam_removal_order,       tf.TensorShape([None, None, None])),
                                  (beam_score,               tf.TensorShape([None, None]))]
            )
            beam = (beam_candidates, beam_removal_order, beam_score)

            # a. & b. expand beam
            beam = self._wrap_candiates_expand(beam)

            # c. evaluate
            x_beam_flatten = self._create_masked_inputs(x, beam, maskable_tokens)
            y_pred = self._evaluate(x_beam_flatten, tf.gather(y, beam[2].value_rowids()))
            y_pred = beam[2].with_values(y_pred) # reshape to the expected RaggedTensor
            score_inc = tf.expand_dims(y_baseline, 1) - y_pred
            new_score = beam[2] + score_inc
            self._debug(iteration_i, beam, x_beam_flatten, new_score)

            # d. crop beam
            crop_selector = self._wrap_beam_select(new_score, extra_args=(self._beam_size, ))
            beam = (
                tf.gather(beam[0], crop_selector, batch_dims=1),
                tf.gather(beam[1], crop_selector, batch_dims=1),
                tf.gather(new_score, crop_selector, batch_dims=1)
            )

            # Transfer the explanation for observations that are done.
            #   For following iterations, _candiates_expand(beam) will make those iterations have
            #   zero candidates. So the optimal_removal_order can not be recovered afterwards.
            observation_done = max_iterations == (iteration_i + 1)
            if tf.math.reduce_any(observation_done):
                # grap observations
                done_obs_idx = tf.squeeze(tf.cast(tf.where(observation_done), dtype=tf.dtypes.int32), axis=1)
                optimal_removal_order = tf.gather(beam[1], done_obs_idx).merge_dims(0, 1)

                # convert removal_order to explanations and save
                explanations = tf.tensor_scatter_nd_update(
                    explanations,
                    tf.stack([
                        tf.repeat(done_obs_idx, iteration_i + 1),
                        tf.reshape(optimal_removal_order, [-1])
                    ], axis=1),
                    tf.repeat(tf.range(iteration_i + 1, 0, -1, dtype=tf.dtypes.float32), tf.size(done_obs_idx))
                )

            # update loop invariants
            beam_candidates, beam_removal_order, beam_score = beam

        return explanations

# TODO: I will admit. This is kinda a hack to control this hyperparameter. This should
#  go in the experiment_id generator, but then I would have to integrate that parameter everywhere.

class BeamSearch10(BeamSearch):
    _name = 'beam-sign-10'
    _default_beam_size = 10

class BeamSearch20(BeamSearch):
    _name = 'beam-sign-20'
    _default_beam_size = 20

class BeamSearch50(BeamSearch):
    _name = 'beam-sign-50'
    _default_beam_size = 50
