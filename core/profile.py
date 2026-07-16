import numpy as np

class Profile:
      def __init__(self, name, descriptors):
            self.name = name
            self.descriptors = descriptors

      def add_descriptor(self, descriptor):
            self.descriptors.append(descriptor)

      @property
      def average_descriptor(self):
            return np.mean(self.descriptors, axis=0)

# descriptors is numpy array with shape (N, 512)
# each descriptor is (512,)