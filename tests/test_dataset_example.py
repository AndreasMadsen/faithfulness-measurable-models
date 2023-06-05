
import pathlib
from dataclasses import dataclass
from typing import Tuple

import pytest

from ecoroar.dataset import datasets

@dataclass
class DatasetExample:
    name: str
    expected: Tuple[str]

# NOTE: This does not include MIMIC datasets, as that would be a HIPPA violation
expected_sizes = [
    DatasetExample('bAbI-1', (
        "Daniel went back to the bathroom. Mary went to the kitchen. Mary went to the hallway. Sandra moved to the garden. Daniel went back to the hallway. John moved to the office. Sandra journeyed to the kitchen. Mary went to the kitchen.",
        "Where is Mary?"
    )),
    DatasetExample('bAbI-2', (
        "Daniel moved to the hallway. Sandra journeyed to the garden. Sandra journeyed to the bedroom. Daniel went back to the kitchen. Daniel got the milk there. Mary travelled to the bathroom. Daniel discarded the milk there. Sandra travelled to the hallway. Mary went back to the kitchen. Sandra travelled to the bedroom. Mary moved to the office. John moved to the bathroom. Daniel got the milk there. Daniel travelled to the bathroom. Daniel left the milk. Daniel picked up the milk there. Mary moved to the hallway. Sandra went back to the office. John journeyed to the kitchen. Daniel put down the milk.",
        "Where is the milk?"
    )),
    DatasetExample('bAbI-3', (
        "Mary went back to the hallway. Sandra moved to the kitchen. Sandra grabbed the milk. Daniel travelled to the bathroom. Daniel picked up the apple there. John moved to the hallway. Daniel put down the apple. John moved to the kitchen. Mary travelled to the bathroom. Mary grabbed the apple. Sandra discarded the milk. Mary put down the apple. John travelled to the office. Mary moved to the bedroom. John journeyed to the bedroom. Mary grabbed the football. Sandra journeyed to the bedroom. Mary discarded the football. John picked up the football. John left the football there. Daniel took the apple. Sandra took the football. Sandra journeyed to the bathroom. John went to the hallway. Mary went back to the office. Daniel travelled to the kitchen. Mary travelled to the bedroom. Sandra put down the football. Sandra grabbed the football. Daniel left the apple. Sandra discarded the football. Daniel got the apple. Sandra grabbed the football. Sandra travelled to the hallway. Daniel moved to the bedroom. Daniel journeyed to the bathroom. Sandra went back to the kitchen. Sandra put down the football. Daniel journeyed to the office. Mary went back to the kitchen. Daniel went to the kitchen. Sandra went to the garden.",
        "Where was the football before the kitchen?"
    )),
    DatasetExample('BoolQ', (
        "Evil Queen (Disney) -- This version of the fairy tale character has been very well received by film critics and the public, and is considered one of Disney's most iconic and menacing villains. Besides in the film, the Evil Queen has made numerous appearances in Disney attractions and productions, including not only these directly related to the tale of Snow White, such as Fantasmic!, The Kingdom Keepers and Kingdom Hearts Birth by Sleep, sometimes appearing in them alongside Maleficent from Sleeping Beauty. The film's version of the Queen has also become a popular archetype that influenced a number of artists and non-Disney works.",
        "are maleficent and the evil queen the same"
    )),
    DatasetExample('CB', (
        "Obeying his instruction, I proffered my hand, open palm upwards, towards the animal. The ratbird climbed on and began to preen its fur unconcernedly. Nobody will blame me if I say that in the circumstances I became very uneasy.",
        "in the circumstances she became very uneasy"
    )),
    DatasetExample('CoLA', (
        "That picture of Susan offended her.",
    )),
    DatasetExample('IMDB', (
        "There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come.",
    )),
    DatasetExample('MNLI', (
        "uh-huh oh yeah all the people for right uh life or something",
        "yeah lots of people for the right life "
    )),
    DatasetExample('MRPC', (
    "The show 's closure affected third-quarter earnings per share by a penny .",
        "The company said this impacted earnings by a penny a share ."
    )),
    DatasetExample('QNLI', (
        "The South Florida/Miami area has previously hosted the event 10 times (tied for most with New Orleans), with the most recent one being Super Bowl XLIV in 2010.",
        "When were the finalists announced?"
    )),
    DatasetExample('QQP', (
        "Who is going to be a better president - Hillary Clinton or Donald Trump?",
        "In what aspects is Hillary Clinton better than Trump?"
    )),
    DatasetExample('RTE', (
        "Tropical Storm Irene on August 11, 2005 at 16:15 UTC. Tropical Storm Irene will increase in strength over the next several days, possibly developing into   a hurricane that will hit the east coast of the United States, said the National Hurricane Center of Miami, Florida in a report today.  Irene was located approximately 975 kilometers south-southeast of Bermuda at 16:00 UTC today. Forecasters say that the storm is now moving in a west-  northwest direction with top sustained winds of 40 miles per hour.",
        "A storm called Irene is going to approach the east coast of the US."
    )),
    DatasetExample('SNLI', (
        "A girl in a blue leotard hula hoops on a stage with balloon shapes in the background.",
        "A girl is entertaining on stage"
    )),
    DatasetExample('SST2', (
        "a valueless kiddie paean to pro basketball underwritten by the nba . ",
    )),
    DatasetExample('WNLI', (
        "The older students were bullying the younger ones, so we rescued them.",
        "We rescued the older students."
    ))
]

@pytest.mark.parametrize("info", expected_sizes, ids=lambda info: info.name)
def test_dataset_output(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'), use_snapshot=False, use_cache=False)
    x, _ = dataset.test().take(1).get_single_element()
    assert len(x) == len(info.expected)
    for output, expected in zip(x, info.expected):
        assert output.numpy().decode('utf-8') == expected
