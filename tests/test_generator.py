from unittest import TestCase

from propinfer import SubsamplingGenerator
from numpy import stack, sum, int32
from numpy.random import randint
from pandas import DataFrame

class TestExperiment(TestCase):
    def test_subsampling_generator(self):
        attr1 = randint(0, 2, 32768)
        attr2 = randint(0, 3, 32768)
        attr3 = randint(0, 4, 32768)

        data = DataFrame(data=stack((attr1, attr2, attr3), axis=1), columns=['Bin', 'Tri', 'Quad'], dtype=int32)
        data.loc[:, 'Cat'] = data.Quad.astype('category')

        gen = SubsamplingGenerator(data, 'Quad', 'Bin', proportion=0.1)

        sample = gen.sample(False)
        assert 0.49 < sum(sample['Bin_1']) / len(sample) < 0.51
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        sample = gen.sample(True)
        assert 0.09 < sum(sample['Bin_1']) / len(sample) < 0.11
        assert 0.2 < sum(sample['Quad'] == 1) / len(sample) < 0.3

        self.assertRaises(AssertionError, SubsamplingGenerator, data, 'Tri', 'Quad', proportion=0.1)

        gen = SubsamplingGenerator(data, 'Tri', 'Quad', target_category=1, proportion=0.1)
        sample = gen.sample(False)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.49 < sum(sample['Quad_1']) / len(sample) < 0.51

        sample = gen.sample(True)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.09 < sum(sample['Quad_1']) / len(sample) < 0.11
        assert 0.25 < sum(sample['Quad_0']) / len(sample) < 0.35
        assert 0.25 < sum(sample['Quad_2']) / len(sample) < 0.35
        assert 0.25 < sum(sample['Quad_3']) / len(sample) < 0.35

        gen = SubsamplingGenerator(data, 'Tri', 'Quad', target_category=1, proportion=0.1, split=True)
        sample = gen.sample(False)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.49 < sum(sample['Quad_1']) / len(sample) < 0.51

        sample = gen.sample(True)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.09 < sum(sample['Quad_1']) / len(sample) < 0.11

        gen = SubsamplingGenerator(data, 'Tri', 'Cat', target_category=1, proportion=0.1)
        gen.sample(False)

        gen = SubsamplingGenerator(data, 'Cat', 'Bin', proportion=0.1)
        gen.sample(False)

        gen = SubsamplingGenerator(data, 'Tri', 'Quad', target_category=1, regression=True)
        sample = gen.sample(0.5)
        assert 0.25 < sum(sample['Tri'] == 1) / len(sample) < 0.4
        assert 0.49 < sum(sample['Quad_1']) / len(sample) < 0.51

        sample = gen.sample(0.25)
        assert 0.24 < sum(sample['Quad_1']) / len(sample) < 0.26

        sample = gen.sample(0.75)
        assert 0.74 < sum(sample['Quad_1']) / len(sample) < 0.76

        sample = gen.sample(0.)
        assert sum(sample['Quad_1']) / len(sample) < 0.01

        sample = gen.sample(1.)
        assert 0.99 < sum(sample['Quad_1']) / len(sample)