from unittest import TestCase

from propinfer import SubsamplingGenerator
from numpy import stack, sum
from numpy.random import randint
from pandas import DataFrame

class TestExperiment(TestCase):
    def test_subsampling_generator(self):
        attr1 = randint(0, 2, 32768)
        attr2 = randint(0, 3, 32768)
        attr3 = randint(0, 4, 32768)

        data = DataFrame(data=stack((attr1, attr2, attr3), axis=1), columns=['Bin', 'Tri', 'Quad'])

        gen = SubsamplingGenerator(data, 'Quad', 'Bin', proportion=0.1)

        sample = gen.sample(False)
        assert 0.4 < sum(sample['Bin'] == 1) / len(sample) < 0.6
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        sample = gen.sample(True)
        assert 0.09 < sum(sample['Bin'] == 1) / len(sample) < 0.11
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        self.assertRaises(AssertionError, SubsamplingGenerator, data, 'Tri', 'Quad', proportion=0.1)

        gen = SubsamplingGenerator(data, 'Tri', 'Quad', target_category=1, proportion=0.1)
        sample = gen.sample(False)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        sample = gen.sample(True)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.09 < sum(sample['Quad'] == 1) / len(sample) < 0.11
        assert 0.25 < sum(sample['Quad'] == 0) / len(sample) < 0.35
        assert 0.25 < sum(sample['Quad'] == 2) / len(sample) < 0.35
        assert 0.25 < sum(sample['Quad'] == 3) / len(sample) < 0.35

        gen = SubsamplingGenerator(data, 'Tri', 'Quad', target_category=1, proportion=0.1, split=True)
        sample = gen.sample(False)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        sample = gen.sample(True)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.09 < sum(sample['Quad'] == 1) / len(sample) < 0.11