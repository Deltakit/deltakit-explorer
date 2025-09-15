import pytest
import numpy as np
from deltakit_explorer._utils._decorators import validate_and_split_decoding, validate_generation
from deltakit_explorer.types._types import DetectionEvents, ObservableFlips, Decoder
from deltakit_explorer.enums._api_enums import DecoderType, QECECodeType
from deltakit_explorer.types._experiment_types import QECExperimentDefinition, CircuitParameters


@pytest.mark.parametrize("decoder_type", [DecoderType.MWPM, DecoderType.LCD, DecoderType.BP_OSD])
def test_validate_and_split_decoding_batches(decoder_type):
    @validate_and_split_decoding
    def dummy(obj, dets, obs, decoder, circuit, leakage=None):
        return type("Result", (), {"fails": 1, "shots": 2, "times": [1.0], "counts": [2], "predictions": None})()
    dets = DetectionEvents(np.ones((1000000, 2)), "B8", 2)
    obs = ObservableFlips(np.ones((1000000, 2)), "B8", 2)
    decoder = Decoder(decoder_type)
    result = dummy(None, dets, obs, decoder, "OBSERVABLE_INCLUDE\n" * 1000 + "DETECTOR(\n" * 10000)
    assert result.fails > 0

def test_validate_generation_warns(mocker):
    @validate_generation
    def dummy(obj, exp_def):
        return "ok"
    params = CircuitParameters.from_sizes([22, 22])
    exp_def = QECExperimentDefinition(
        experiment_type=1, code_type=QECECodeType.ROTATED_PLANAR,
        observable_basis=1, num_rounds=1, basis_gates=["CZ"], parameters=params
    )
    assert dummy(None, exp_def) == "ok"

def test_validate_and_split_decoding_negative(mocker):
    @validate_and_split_decoding
    def dummy(obj, dets, obs, decoder, circuit, leakage=None):
        raise Exception("fail")
    dets = DetectionEvents(np.ones((10, 2)), "B8", 2)
    obs = ObservableFlips(np.ones((10, 2)), "B8", 2)
    decoder = Decoder(DecoderType.MWPM)
    with pytest.raises(Exception):
        dummy(None, dets, obs, decoder, "CIRCUIT")
