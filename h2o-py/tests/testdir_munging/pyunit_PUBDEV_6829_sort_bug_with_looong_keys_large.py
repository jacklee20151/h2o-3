from __future__ import print_function
import sys
sys.path.insert(1,"../../")
from tests import pyunit_utils
import time
import h2o

def sort():
    h2o.remove_all()
    df = h2o.import_file(pyunit_utils.locate("bigdata/laptop/jira/PUBDEV_6829_srot_bug_bigKey_part.csv.zip"))
    t1 = time.time()
    df1 = df.sort([1])
    assert df1[0,1] < df1[1,1], "Bug in sorting!"
    print("Time taken to perform sort is {0}".format(time.time()-t1))

if __name__ == "__main__":
    pyunit_utils.standalone_test(sort)
else:
    sort()

