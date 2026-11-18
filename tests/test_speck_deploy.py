import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from neurofusion.speck_deploy import apply_verified, load_config


class Config:
    def __init__(self): self.value = {'network': 1}
    def from_json(self, text): self.value = json.loads(text)
    def to_json(self): return json.dumps(self.value)


def sdk(valid=True):
    return SimpleNamespace(speck2f=SimpleNamespace(
        configuration=SimpleNamespace(SpeckConfiguration=Config),
        validate_configuration=lambda c:(valid,'invalid network')))


class SpeckDeployTests(unittest.TestCase):
    def monitored_configs(self):
        requested, actual = Config(), Config()
        for config in (requested, actual):
            config.value = {'value0':{'factoryConfig':{'ioSel':0},'weight':7}}
            config.dvs_layer = SimpleNamespace(raw_monitor_enable=False,monitor_enable=False)
            config.cnn_layers = [SimpleNamespace(monitor_enable=True)]
        actual.value['value0']['factoryConfig']['ioSel'] = 32
        runtime = sdk()
        runtime.__version__ = '0.48.6.0'
        return requested,actual,runtime

    def test_known_monitor_normalization_is_accepted(self):
        requested,actual,runtime = self.monitored_configs()
        model=SimpleNamespace(apply_configuration=lambda c:None,get_configuration=lambda:actual)
        self.assertIs(apply_verified(model,requested,runtime),actual)
        self.assertEqual(requested.value['value0']['factoryConfig']['ioSel'],0)

    def test_monitor_normalization_does_not_hide_weight_changes(self):
        requested,actual,runtime = self.monitored_configs()
        actual.value['value0']['weight'] = 8
        model=SimpleNamespace(apply_configuration=lambda c:None,get_configuration=lambda:actual)
        with self.assertRaisesRegex(RuntimeError,'weight'):
            apply_verified(model,requested,runtime)

    def test_invalid_file_rejected_before_device_access(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Invalid Speck2f'):
                load_config(p,sdk(False))

    def test_changed_readback_is_rejected(self):
        requested,actual=Config(),Config()
        actual.value={'network':2}
        model=SimpleNamespace(apply_configuration=lambda c:None,get_configuration=lambda:actual)
        with self.assertRaisesRegex(RuntimeError,'differs'):
            apply_verified(model,requested,sdk())

    def test_invalid_readback_is_rejected(self):
        requested=Config()
        model=SimpleNamespace(apply_configuration=lambda c:None,get_configuration=lambda:requested)
        with self.assertRaisesRegex(RuntimeError,'invalid'):
            apply_verified(model,requested,sdk(False))

    def test_verified_apply_returns_readback(self):
        requested=Config()
        calls=[]
        model=SimpleNamespace(apply_configuration=calls.append,get_configuration=lambda:requested)
        self.assertIs(apply_verified(model,requested,sdk()),requested)
        self.assertEqual(calls,[requested])
