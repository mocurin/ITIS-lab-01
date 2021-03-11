"""App entry point"""
import logging

from sources.models import Neuron, FitState
from sources.generators import (boolean_generator,
                                boolean_str_to_seq,
                                boolean_mask_generator,
                                subset,
                                DataGenerator)
from sources.activations import activations
from sources.initializers import zeros


logging.basicConfig(level=logging.INFO)


BF = '1101111100001111'
MAX_EPOCH = 100
ACTIVATIONS = ['Threshold', 'Logistic']


def main():
    data = boolean_str_to_seq(BF)
    raw_data = boolean_generator(data)

    # Since `train_data` & `validation_data` both work sequential
    # and consume exactly `len(BF)` samples from `raw_data`, `raw_data`
    # is never invalidated -> can use single `raw_data` generator
    validation_data = DataGenerator(raw_data, len(BF))

    for activation_name in ACTIVATIONS:
        # Walk over masks from [1, ..., 1] to [0, ... 0, 1]
        bmg = boolean_mask_generator(len(BF))

        activation = activations[activation_name]

        for idx, masks in enumerate(bmg):
            for mask in masks:
                # Take samples subset
                logging.info(f"Mask: {''.join(str(value) for value in mask)}")

                train_data = subset(raw_data, mask)
                train_data = DataGenerator(train_data, len(BF), MAX_EPOCH)

                model = Neuron(
                    inputs=4,
                    use_bias=True,
                    activation=activation(),
                    weights_initializer=zeros(),
                    bias_initializer=zeros()
                )

                (_, epoch_report), state = model.fit(
                    train_data,
                    validation_data,
                    norm=0.3,
                    write_sample_history=False
                )

                mask = ''.join(str(value) for value in mask)

                if state is FitState.EARLY_STOP:
                    logging.info("Early stop!")
                    epoch_report.storage.append({'mask': mask})
                    epoch_report.save(f"histories/{activation_name}.txt")
                    break

                if state is FitState.STALE_STOP:
                    logging.info("Stale stop!")
                    continue

                logging.info("Eternity end!")


if __name__ == '__main__':
    main()
