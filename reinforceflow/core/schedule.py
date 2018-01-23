from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Schedule(object):
    @staticmethod
    def create(schedule, initial_value, total_steps):
        if isinstance(schedule, Schedule):
            return schedule

        if schedule is None or schedule in ['const', 'constant']:
            return ConstantSchedule(initial_value)

        if schedule in ['linear']:
            return LinearSchedule(total_steps, initial_value, 0)

    def value(self, step):
        """Returns scheduled value at given timestep."""
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def value(self, step):
        return self.initial_value


class LinearSchedule(Schedule):
    def __init__(self, total_steps, initial_value, final_value):
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.final_value = final_value

    def value(self, step):
        """See `Schedule.value`."""
        fraction = min(step / self.total_steps, 1.0)
        return self.initial_value + fraction*(self.final_value - self.initial_value)
