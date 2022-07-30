#-*- coding: utf-8 -*-

import math

class LearningRateSchedule:
    """
    Mimicks the behavior of tf.keras.optimizers.schedules.LearningRateSchedule. Essentially
    returns a learning rate value for a given epoch number in order to model an
    evolving learning rate schedule, e.g. exponential decay.
    """

    def __init__(self, *args, **kwargs):
        self.schedule_function = None

    def __call__(self, iteration):
        """
        Evaluate schedule function at a given iteration number.
        """
        return self.schedule_function(iteration)


class ConstantSchedule(LearningRateSchedule):
    """
    Constant learning rate.
    """
    
    def __init__(self, lr_value, **kwargs):

        # Initialize parent class
        super().__init__([], **kwargs)

        # Save the learning rate value
        self.lr_value = lr_value

        # Construct the schedule function
        self.schedule_function = lambda n: self.lr_value

        return


class ExponentialDecay(LearningRateSchedule):
    """
    Exponentially-decaying learning rate schedule.
    """

    def __init__(self, lr_init, lr_final, n_epoch, **kwargs):
        """
        Initialize the exponential decaying schedule.
        """
        # Initialize parent class
        super().__init__([], **kwargs)

        # Compute the decay parameter
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.tau = -n_epoch / math.log(lr_final / lr_init)

        # Construct the schedule function
        self.schedule_function = lambda n: self.lr_init * math.exp(-n / self.tau)

        return

# end of file
