"""
extractor extracts data from json dump and seperates test inputs from target outputs for use in (semi-supervised) learning
also separates attributes that are used for identification of samples and are not to be used in training
might consume all of your system memory
"""
import json
from time import time
import logging
from matplotlib import pyplot as plt
import numpy as np
import os

"""
available keys as of now are:
diameter
binary_object
ascending_node
b_minus_v
object_type
number
v_minus_rprime
spin_min_amplitude
v_minus_zprime
v_minus_uprime
aphelion_distance
albedo
epoch_jd
diameter_neowise_4
diameter_neowise_3
diameter_neowise_2
albedo_neowise
absolute_magnitude
mean_anomaly
diameter_4
diameter_2
diameter_3
phase_slope
panstarrs_v_minus_iprime
v_minus_gprime
argument_of_perihelion
inclination
diameter_neowise
panstarrs_v_minus_yprime
delta_v
perihelion_date_jd
rc_minus_ic
u_minus_b
period
albedo_2
albedo_3
panstarrs_v_minus_rprime
observations
albedo_4
eccentricity
lightcurve_quality
panstarrs_v_minus_zprime
panstarrs_v_minus_uprime
name
albedo_neowise_2
v_minus_wprime
v_minus_yprime
spin_period
panstarrs_v_minus_gprime
albedo_neowise_4
v_minus_rc
v_minus_iprime
semimajor_axis
panstarrs_v_minus_wprime
albedo_neowise_3
spin_max_amplitude
taxonomy_class
residual_rms
"""

"""unusedKeys = [binary_object, delta_v, lightcurve_quality, residual_rms]"""

if __name__ == "__main__":
  logging.basicConfig(level = logging.DEBUG)

log = logging.getLogger(__name__)


class Sample(object):
  def __init__(self, inputData, identification, outputData):
    self.x = inputData
    self.id = identification
    self.y = outputData
    
  def __repr__(self):
    return("x: %s,\ny: %s\n%s" % (self.x, self.y, self.id))
    
class SampleFactory(object):
  def __init__(self, listOfFiles = ["data_dumps/data_dump_%s.json" % i for i in range(45)]):
    script_dir = os.path.dirname(__file__)
    self.listOfFiles = map(lambda x : os.path.join(script_dir, x), listOfFiles)
    self.samples = []
    self.encodings = dict()

  """
  def prepareSampling(self):
    j, m, n = 0, 0, 0
    taxonomies = set()
    for source in self.listOfFiles:
      with open(source) as source:
        json_data = json.load(source)
        for sample in json_data:
          taxonomies.add(sample['taxonomy_class'])
          abcedkmloqpsrutvx
          for key in sample:
            if(sample['object_type'] == u'M'):
              m += 1
            elif(sample['object_type'] == u'J'):
              j += 1
            elif(sample['object_type'] == None):
              n += 1
            else:
              print(sample['object_type'])
               
    print(n,j,m)
    print("taxonomies %s" %taxonomies)
    """

  def prepareSampling(self):
    taxes = {u'A' : 0,
             u'C' : 0,
             u'B' : 0,
             u'D' : 0,
             u'E' : 0,
             u'K' : 0,
             u'L' : 0,
             u'O' : 0,
             u'M' : 0,
             u'M' : 0,
             u'P' : 0,
             u'Q' : 0,
             u'R' : 0,
             u'S' : 0,
             u'T' : 0,
             u'U' : 0,
             u'V' : 0,
             u'X' : 0,
             None : 0}
    for source in self.listOfFiles:
      with open(source) as source:
        json_data = json.load(source)
        for sample in json_data:
          taxes.update({sample['taxonomy_class'] : taxes[sample['taxonomy_class']]+1})
    print(taxes)

  def parseSample(self, sample):
    outputKeys = ["taxonomy_class"]
    """
    not enough values for the following stuff:
    "object_type", "diameter", "diameter_4", "diameter_2", "diameter_3",  "ascending_node", "b_minus_v", "diameter_neowise",
    """
    identificationKeys = ["number", "name", "taxonomy_class"]
    inputKeys = ["v_minus_rprime", "spin_min_amplitude", "v_minus_zprime",
                 "v_minus_uprime", "aphelion_distance", "albedo", "epoch_jd", "diameter_neowise_4",
                 "diameter_neowise_3", "diameter_neowise_2", "albedo_neowise", "absolute_magnitude",
                 "mean_anomaly", "phase_slope", "panstarrs_v_minus_iprime", "v_minus_gprime",
                 "argument_of_perihelion", "inclination", "panstarrs_v_minus_yprime",
                 "perihelion_date_jd", "rc_minus_ic", "u_minus_b", "period", "albedo_2",
                 "albedo_3", "panstarrs_v_minus_rprime", "observations", "albedo_4", "eccentricity",
                 "panstarrs_v_minus_zprime", "panstarrs_v_minus_uprime", "albedo_neowise_2",
                 "v_minus_wprime", "v_minus_yprime", "spin_period", "panstarrs_v_minus_gprime",
                 "albedo_neowise_4", "v_minus_rc", "v_minus_iprime", "semimajor_axis",
                 "panstarrs_v_minus_wprime", "albedo_neowise_3", "spin_max_amplitude"]

    def magic(x, key):
      if(key in self.encodings):
        return(self.encodings[key](x))
      if(x == None):
        return(0.0)
      return(x)
    
    columnsThatNeedEncoding = set()

    self.samples.append(Sample(
      [magic(sample[key], key) for key in inputKeys],
      [sample[key] for key in identificationKeys],
      [magic(sample[key], key) for key in outputKeys]))
                       

  def run(self):
    t0 = time()
    #self.prepareSampling()
    for source in self.listOfFiles:
      with open(source) as source:
        json_data = json.load(source)
        for sample in json_data:
          self.parseSample(sample)
    n = len(self.samples)
    t = time() - t0
    log.info("read %s samples in %ss\n%s samples per second" % (n , t, n/t))
    
f = SampleFactory()
f.run()

matrix = []
categories = []
"""taxes = {u'A' : 1,
         u'C' : 2,
         u'B' : 3,
         u'D' : 4,
         u'E' : 5,
         u'K' : 6,
         u'L' : 7,
         u'O' : 8,
         u'M' : 9,
         u'M' : 10,
         u'P' : 11,
         u'Q' : 12,
         u'R' : 13,
         u'S' : 14,
         u'T' : 15,
         u'U' : 16,
         u'V' : 17,
         u'X' : 18,
         None : 0}
         None """
for s in f.samples:
  matrix.append(s.x)
  categories.append(s.y[0])
matrix = np.array(matrix)
with file("matrix") as savedMatrix:
 np.save(savedMatrix, matrix)
categories = np.array(categories)
inputKeys = ["v_minus_rprime", "spin_min_amplitude", "v_minus_zprime",
                 "v_minus_uprime", "aphelion_distance", "albedo", "epoch_jd", "diameter_neowise_4",
                 "diameter_neowise_3", "diameter_neowise_2", "albedo_neowise", "absolute_magnitude",
                 "mean_anomaly", "phase_slope", "panstarrs_v_minus_iprime", "v_minus_gprime",
                 "argument_of_perihelion", "inclination", "panstarrs_v_minus_yprime",
                 "perihelion_date_jd", "rc_minus_ic", "u_minus_b", "period", "albedo_2",
                 "albedo_3", "panstarrs_v_minus_rprime", "observations", "albedo_4", "eccentricity",
                 "panstarrs_v_minus_zprime", "panstarrs_v_minus_uprime", "albedo_neowise_2",
                 "v_minus_wprime", "v_minus_yprime", "spin_period", "panstarrs_v_minus_gprime",
                 "albedo_neowise_4", "v_minus_rc", "v_minus_iprime", "semimajor_axis",
                 "panstarrs_v_minus_wprime", "albedo_neowise_3", "spin_max_amplitude"]
for i in range(len(matrix.T)):
  try:
    x = matrix.T[i]
    #x = np.take(x, x.nonzero())
    log.info("preparing %s for plotting" % inputKeys[i])
    xFlatNonZero = np.flatnonzero(x)
    x = x.ravel()[xFlatNonZero]
    c = categories.ravel()[xFlatNonZero]
    minValue, maxValue = int(np.amin(x))-1, int(np.amax(x))+1
    plt.subplot(211)
    plt.plot(np.sort(x))
    plt.subplot(212)
    for category in [u'A',u'C',u'B',u'D',u'E',u'K',u'L',u'O',u'M',u'M',u'P',u'Q',u'R',u'S',u'T',u'U',u'V',u'X',None]:
      try:
        index = np.argwhere(c == category)
        print(plt.hist(x.ravel()[index], range(minValue, maxValue), alpha = 0.2, label = category))
      except ValueError as e:
        print("inner loop error %s" %e)
    plt.hist(x, range=(minValue, maxValue), alpha = 0.2)
  except ValueError as e:
    print("outer loop error %s" %e)
  plt.title(inputKeys[i])
  plt.legend()
  plt.show()
