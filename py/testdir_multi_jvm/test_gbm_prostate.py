import sys
sys.path.insert(1, '../../h2o-py/src/main/py')

from h2o import H2OConnection
from h2o import H2OFrame
from h2o import H2OGBM
from tabulate import tabulate

######################################################
# Parse command-line args.
#
# usage:  python test_name.py --usecloud ipaddr:port
#

ip_port = sys.argv[2].split(":")
print ip_port
ip = ip_port[0]
port = int(ip_port[1])

######################################################
#
# Sample Running GBM on prostate.csv

# Connect to a pre-existing cluster
cluster = H2OConnection(ip = ip, port = port)

df = H2OFrame(remote_fname="../../../smalldata/logreg/prostate.csv")
print df.describe()

# Remove ID from training frame
del df['ID']

# For VOL & GLEASON, a zero really means "missing"
vol = df['VOL']
vol[vol==0] = None
gle = df['GLEASON']
gle[gle==0] = None

# Convert CAPSULE to a logical factor
df['CAPSULE'] = df['CAPSULE'].asfactor()

# Test/train split
r = vol.runif()
train = df[r< 0.8]
test  = df[r>=0.8]

# See that the data is ready
print train.describe()
print test .describe()

# Run GBM
gbm = H2OGBM(dataset=train,x="CAPSULE",validation_dataset=test,ntrees=50,max_depth=5,learn_rate=0.1)
#print gbm._model
mm = gbm.metrics()
mm0 = mm[0]
cm = mm0['cm']
conf = cm['confusion_matrix']
print tabulate(conf)


