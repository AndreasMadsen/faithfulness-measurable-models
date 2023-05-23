import pathlib

from ecoroar.dataset import \
    SNLIDataset, \
    Babi1Dataset, Babi2Dataset, Babi3Dataset, \
    MimicDiabetesDataset, MimicAnemiaDataset


def test_local_snli():
    dataset = SNLIDataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for (premise, hypothesis), answer in dataset.train().take(1):
        assert premise.numpy() == b'A man washes or dies clothes in a primitive setting.'
        assert hypothesis.numpy() == b'Washing clothes on a camping trip.'
        assert answer.numpy() == 1

def test_local_babi_1():
    dataset = Babi1Dataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for (paragraph, question), answer in dataset.train().take(1):
        assert paragraph.numpy() == (
            b'Daniel moved to the garden.'
            b' John went to the garden.'
            b' John travelled to the bathroom.'
            b' Mary journeyed to the bedroom.'
            b' Mary went to the hallway.'
            b' Sandra went to the kitchen.'
        )
        assert question.numpy() == b'Where is Sandra?'
        assert answer.numpy() == 2

def test_local_babi_2():
    dataset = Babi2Dataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for (paragraph, question), answer in dataset.train().take(1):
        assert paragraph.numpy() == (
            b'John went back to the garden.'
            b' Daniel picked up the apple there.'
            b' Sandra travelled to the bedroom.'
            b' John travelled to the bathroom.'
            b' Sandra took the football there.'
            b' Daniel discarded the apple.'
            b' Mary grabbed the apple there.'
            b' Sandra travelled to the hallway.'
            b' Mary discarded the apple there.'
            b' Sandra left the football.'
            b' Daniel got the football there.'
            b' Daniel journeyed to the bedroom.'
        )
        assert question.numpy() == b'Where is the football?'
        assert answer.numpy() == 4

def test_local_babi_3():
    dataset = Babi3Dataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    for (paragraph, question), answer in dataset.train().take(1):
        assert paragraph.numpy() == (
            b'Mary moved to the office.'
            b' John went to the kitchen.'
            b' Mary went to the bathroom.'
            b' John grabbed the apple.'
            b' Mary travelled to the office.'
            b' Sandra went to the kitchen.'
            b' John put down the apple there.'
            b' Daniel went back to the bathroom.'
            b' Daniel went to the garden.'
            b' Sandra picked up the apple.'
            b' Sandra moved to the office.'
            b' Sandra put down the apple.'
            b' Sandra went back to the hallway.'
            b' Sandra got the football there.'
            b' Mary journeyed to the bedroom.'
            b' Mary travelled to the garden.'
            b' Sandra dropped the football.'
            b' Sandra journeyed to the office.'
            b' John went back to the office.'
            b' Sandra journeyed to the kitchen.'
            b' Daniel journeyed to the hallway.'
            b' John picked up the apple.'
            b' Daniel took the football.'
            b' Mary travelled to the office.'
            b' Mary travelled to the bathroom.'
            b' Sandra went to the office.'
            b' John dropped the apple.'
            b' John took the apple.'
            b' Daniel journeyed to the bathroom.'
            b' Daniel grabbed the milk.'
            b' Daniel went to the kitchen.'
            b' Daniel went back to the bathroom.'
            b' Daniel put down the football.'
            b' Mary grabbed the football.'
            b' Sandra journeyed to the bedroom.'
            b' Mary dropped the football.'
            b' Daniel picked up the football.'
            b' Daniel left the football.'
            b' Sandra went back to the garden.'
            b' Daniel travelled to the bedroom.'
            b' John put down the apple.'
            b' Daniel moved to the garden.'
            b' Mary took the football.'
            b' Daniel put down the milk there.'
        )
        assert question.numpy() == b'Where was the milk before the bathroom?'
        assert answer.numpy() == 2

def test_local_mimic_diabetes():
    dataset = MimicDiabetesDataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    # HIPPA prevents testing this accuately
    for (text, ), answer in dataset.train().take(1):
        assert len(text.numpy().decode('utf-8').split(' ')) == 1546
        assert answer.numpy() == 0

def test_local_mimic_anemia():
    dataset = MimicAnemiaDataset(persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)

    # HIPPA prevents testing this accuately
    for (text, ), answer in dataset.train().take(1):
        assert len(text.numpy().decode('utf-8').split(' ')) == 1264
        assert answer.numpy() == 1
