"""Schedulers."""

import datetime
import os
from abc import ABC, abstractmethod

import numpy as np

import env

class Action(object):
    """Schedule Action."""

    def __init__(self, task, node):
        self.task = task
        self.node = node

    def __repr__(self):
        return 'Action(task={0} -> node={1})'.format(self.task.label, self.node.label)


class Scheduler(ABC):
    """Scheduler Interface."""

    @abstractmethod
    def schedule(self):
        pass


class CompactScheduler(Scheduler):
    """Compact scheduler."""

    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        """Higher priority for higher utilization."""
        actions = []
        indices = []

        # sort nodes according to reversed utilization, schedule tasks from queue to nodes
        for i_task in range(len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        # proceed to the next timestep
        self.environment.timestep()

        return actions


class SpreadScheduler(Scheduler):
    """Spread scheduler."""

    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        """Higher priority for lower utilization."""
        actions = []
        indices = []

        # sort nodes according to utilization, schedule tasks from queue to nodes
        for i_task in range(len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1])
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        # proceed to the next timestep
        self.environment.timestep()

        return actions