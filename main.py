"""App entry point"""
import logging

from sources.models import Neuron
from sources.generators import boolean_generator, boolean_str_to_seq, DataGenerator
from sources.activations import Threshold
from sources.initializers import zeros


logging.basicConfig(level=logging.INFO)


BF = '0001000100010000'


def main():
    data = boolean_str_to_seq(BF)
    data = boolean_generator(data)
    data = DataGenerator(data, len(BF), 100)

    model = Neuron(
        inputs=4,
        use_bias=True,
        activation=Threshold(),
        weights_initializer=zeros(),
        bias_initializer=zeros()
    )

    batch, epoch = model.fit(data, norm=0.3)

    batch.save('batch.txt')
    epoch.save('epoch.txt')


if __name__ == '__main__':
    main()
