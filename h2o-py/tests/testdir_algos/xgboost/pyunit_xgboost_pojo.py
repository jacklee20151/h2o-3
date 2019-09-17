import sys, os
sys.path.insert(1, "../../../")
import h2o
from h2o.estimators.xgboost import *
from tests import pyunit_utils

def xgboost_mojo_pojo():
    assert H2OXGBoostEstimator.available()

    # Import big dataset to ensure run across multiple nodes
    training_frame = h2o.import_file(pyunit_utils.locate("smalldata/testng/insurance_train1.csv"))
    test_frame = h2o.import_file(pyunit_utils.locate("smalldata/testng/insurance_validation1.csv"))
    x = ['Age', 'District']
    y = 'Claims'
    
    model = H2OXGBoostEstimator(
        training_frame=training_frame, learn_rate=0.7,
        booster='gbtree', seed=1, ntrees=10, distribution='gaussian'
    )
    model.train(x=x, y=y, training_frame=training_frame)

    mojo_name = pyunit_utils.getMojoName(model._id)
    tmp_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "results", mojo_name))
    os.makedirs(tmp_dir)
    model.download_mojo(path=tmp_dir)

    h2o.download_csv(test_frame[x], os.path.join(tmp_dir, 'in.csv'))  # save test file, h2o predict/mojo use same file
    pred_h2o, pred_mojo = pyunit_utils.mojo_predict(model, tmp_dir, mojo_name)  # load model and perform predict

    # h2o.download_csv(pred_h2o, os.path.join(tmp_dir, "h2oPred.csv"))
    pred_pojo = pyunit_utils.pojo_predict(model, tmp_dir, mojo_name, get_xgboost_jar=True)
    print("Comparing mojo predict and h2o predict...")
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)
    print("Comparing pojo predict and h2o predict...")
    pyunit_utils.compare_frames_local(pred_mojo, pred_pojo, 0.1, tol=1e-10)

if __name__ == "__main__":
    pyunit_utils.standalone_test(xgboost_mojo_pojo)
else:
    xgboost_mojo_pojo()
