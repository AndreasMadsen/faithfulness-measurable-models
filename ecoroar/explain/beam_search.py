
from typing import Tuple

import tensorflow as tf

from ..types import Tokenizer, TokenizedDict
from ..transform import SequenceIndentifier
from ..util import get_compiler
from ._importance_measure import ImportanceMeasureObservation
from ._util_evaluate import BatchEvaluator

BeamType = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]

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

def _candiates_expand(sequence_length: tf.Tensor, beam: BeamType) -> BeamType:
    root_candidates, removal_order, score = beam
    root_candidates_n = tf.shape(root_candidates)[0]

    # Encode root candiates as sparse tensors. This is used as a set data structure.
    # The T-dimentional indice will encode the mask vector.
    total_candidates = tf.zeros((0, sequence_length), dtype=tf.dtypes.bool)
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
    return (total_candidates, total_removal_order_ta.concat(), total_score_ta.concat())

class BeamSearch(ImportanceMeasureObservation):
    _name = 'beam-sign'
    _defer_jit = True
    _default_beam_size = None

    def __init__(self, *args, beam_size: int=None, debugging=False,
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

    def _debug(self, interation_i, beam, x_beam, new_score):
        if self._debugging:
            print(f'iteration: {interation_i}')
            for x_beam_p, removal_order_p, score_p in zip(x_beam['input_ids'], beam[1], new_score):
                print(
                    '  ' +
                    ' '.join(map(str, x_beam_p.numpy().tolist())) + ' | ' +
                    ''.join(map(str, removal_order_p.numpy().tolist())) + ':' +
                    str(score_p.numpy().tolist())
                )

    def _create_masked_inputs(self, x: TokenizedDict, beam: BeamType, maskable_tokens: tf.Tensor) -> TokenizedDict:
        unmasked, _, _ = beam
        n_samples = tf.shape(unmasked)[0]

        x_repeat = tf.nest.map_structure(
            lambda item: tf.repeat(tf.expand_dims(item, axis=0), n_samples, axis=0),
            x)
        x_repeat['input_ids']  = tf.where(
            tf.logical_and(maskable_tokens, tf.logical_not(unmasked)),
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

    # STEP: 1.d - beam crop
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

    @tf.function(reduce_retracing=True)
    def _explain_observation(self, x, y):
        input_ids = x['input_ids']
        sequence_length = tf.shape(input_ids)[0]

        # Since only explanation w.r.t. the first sequence are considred, only attempt
        # measures on the first sequence.
        maskable_tokens = tf.squeeze(self._sequence_identifier(tf.expand_dims(input_ids, 0)), 0) == 1
        # For masked inputs, [MASK] would be replaced with [MASK].
        # This enforces zero attribution score. Therefore this can be optimized by
        #   skipping the model evaluation.
        maskable_tokens = tf.logical_and(maskable_tokens, input_ids != self._tokenizer.mask_token_id)

        # Compute baseline prediction
        y_baseline = tf.squeeze(
            self._evaluate.single(tf.nest.map_structure(lambda item: tf.expand_dims(item, axis=0), x), tf.expand_dims(y, 0)),
            axis=0)

        # 0. Intialize beam.
        # Assumed to be sorted highest score to lowest score.
        # Note, the beam is not a tuble here, because tuples do not interact
        #   with set_loop_options(shape_invariants=[])
        beam_candidates = tf.expand_dims(maskable_tokens, 0)
        beam_removal_order = tf.zeros((1, 0), dtype=tf.dtypes.int32)
        beam_score = tf.zeros((1, ), dtype=y_baseline.dtype)

        for iteration_i in tf.range(tf.math.reduce_sum(tf.cast(maskable_tokens, tf.dtypes.int32))):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(beam_candidates, tf.TensorShape([None, None])),
                                  (beam_removal_order, tf.TensorShape([None, None])),
                                  (beam_score, tf.TensorShape([None, ]))]
            )
            beam = (beam_candidates, beam_removal_order, beam_score)

            # a. & b. expand beam
            beam = _candiates_expand(sequence_length, beam)

            # c. evaluate
            x_beam = self._create_masked_inputs(x, beam, maskable_tokens)
            y_pred = self._evaluate(x_beam, tf.repeat(y, tf.shape(x_beam['input_ids'])[0]))
            score_inc = y_baseline - y_pred
            new_score = beam[2] + score_inc
            self._debug(iteration_i, beam, x_beam, new_score)

            # d. crop beam
            crop_selector = tf.argsort(new_score, stable=True, direction='DESCENDING')[:self._beam_size]
            beam_candidates = tf.gather(beam[0], crop_selector)
            beam_removal_order = tf.gather(beam[1], crop_selector)
            beam_score = tf.gather(new_score, crop_selector)

        # Convert optimal_removal_order to importance measure
        # i.e. [4, 2, 3, 1, 0] -> [1, 2, 4, 3, 5]
        optimal_removal_order = tf.squeeze(beam_removal_order, axis=0)

        return tf.tensor_scatter_nd_update(
            tf.zeros((sequence_length, ), dtype=tf.dtypes.float32),
            tf.expand_dims(optimal_removal_order, axis=1),
            tf.range(tf.size(optimal_removal_order), 0, -1, dtype=tf.dtypes.float32)
        )

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
