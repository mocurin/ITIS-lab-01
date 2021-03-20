"""App entry point"""
import logging
import sys


from sources.models import Neuron, FitStopEvents
from sources.generators import (boolean_generator,
                                boolean_str_to_seq,
                                boolean_mask_generator,
                                subset,
                                DataGenerator)
from sources.activations import activations
from sources.initializers import zeros
from sources.historian import LoggingVerbosity


logging.basicConfig(level=logging.INFO, stream=sys.stdout)


BF = '1110111011101111'
MAX_EPOCH = 100
ACTIVATIONS = ['Threshold', 'Logistic']


# Set parameters
def create_and_fit_model(raw_data, validation_data, mask, name: str, activation):
    # Take samples subset
    train_data = subset(raw_data, mask)
    train_data = DataGenerator(train_data, len(BF), MAX_EPOCH)

    model = Neuron(
        inputs=4,
        use_bias=True,
        activation=activation(),
        weights_initializer=zeros(),
        bias_initializer=zeros()
    )

    return model.fit(
        train_data,
        validation_data,
        norm=0.3,
        delta=0.00001,
        plot_prefix=name,
        logging_verbosity=LoggingVerbosity.EPOCH
    )


def find_tiniest_subset(raw_data, validation_data, mask_generator, activation_name: str):
    masks = (
        mask
        for masks
        in mask_generator
        for mask
        in masks
    )

    activation = activations[activation_name]

    for mask in masks:
        # Reverse mask as distinct_permutations gives mask in reverse order
        # mask = list(reversed(mask))

        # Log created set mask
        str_mask = ''.join(str(value) for value in mask)
        logging.info(f"Mask: {str_mask}, activation: {activation_name}")

        # Create and fit model
        _, event = create_and_fit_model(
            raw_data,
            validation_data,
            mask,
            f"{str_mask}_{activation_name}",
            activation
        )

        if event is FitStopEvents.METRIC_STOP:
            logging.info("Metric stop!")
            break

        if event is FitStopEvents.STALE_STOP:
            continue


def fit_complete_subset(raw_data, validation_data, activation_name: str):
    # Get required activation
    activation = activations[activation_name]

    logging.info(f"Complete set, activation: {activation_name}")

    # Create and fit model
    _, event = create_and_fit_model(
        raw_data,
        validation_data,
        [1] * 16,
        f"full_{activation_name}",
        activation
    )

    logging.info(f"Fitting completed with: {repr(event)}")


def main():
    data = boolean_str_to_seq(BF)
    raw_data = boolean_generator(data)

    # Since `train_data` & `validation_data` both work sequential
    # and consume exactly `len(BF)` samples from `raw_data`, `raw_data`
    # is never invalidated -> can use single `raw_data` generator
    validation_data = DataGenerator(raw_data, len(BF))

    # Find tiniest subset
    for activation_name in ACTIVATIONS:
        find_tiniest_subset(
            raw_data,
            validation_data,
            boolean_mask_generator(len(BF), start_from=3),
            activation_name
        )

    # Fit on complete subsets
    for activation_name in ACTIVATIONS:
        fit_complete_subset(
            raw_data,
            validation_data,
            activation_name
        )


if __name__ == '__main__':
    main()
