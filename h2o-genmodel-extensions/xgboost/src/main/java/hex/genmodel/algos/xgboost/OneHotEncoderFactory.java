package hex.genmodel.algos.xgboost;

import biz.k11i.xgboost.util.FVec;
import hex.genmodel.GenModel;

public class OneHotEncoderFactory {

  private final boolean _sparse;
  private final int[] _catMap;
  private final float _notHot;
  private final int _cats;
  private final int _nums;
  private final int[] _catOffsets;
  private final boolean _useAllFactorLevels;

  public OneHotEncoderFactory(boolean sparse, int cats, int nums, int[] catOffsets, boolean useAllFactorLevels) {
    _sparse = sparse;
    _cats = cats;
    _nums = nums;
    _catOffsets = catOffsets;
    _notHot = sparse ? Float.NaN : 0;
    _useAllFactorLevels = useAllFactorLevels;
    if (catOffsets == null) {
      _catMap = new int[0];
    } else {
      _catMap = new int[catOffsets[cats]];
      for (int c = 0; c < cats; c++) {
        for (int j = catOffsets[c]; j < catOffsets[c+1]; j++)
          _catMap[j] = c;
      }
    }
  }

  public OneHotEncoderFVec fromArray(double[] input) {
    float[] numValues = new float[_nums];
    int[] catValues = new int[_cats];
    GenModel.setCats(input, catValues, _cats, _catOffsets, _useAllFactorLevels);
    for (int i = 0; i < numValues.length; i++) {
      float val = (float) input[_cats + i];
      numValues[i] = _sparse && (val == 0) ? Float.NaN : val;
    }

    return new OneHotEncoderFVec(_catMap, catValues, numValues, _notHot);
  }

  private static class OneHotEncoderFVec implements FVec {
    private final int[] _catMap;
    private final int[] _catValues;
    private final float[] _numValues;
    private final float _notHot;

    private  OneHotEncoderFVec(int[] catMap, int[] catValues, float[] numValues, float notHot) {
      _catMap = catMap;
      _catValues = catValues;
      _numValues = numValues;
      _notHot = notHot;
    }

    @Override
    public final float fvalue(int index) {
      if (index >= _catMap.length)
        return _numValues[index - _catMap.length];

      final boolean isHot = _catValues[_catMap[index]] == index;
      return isHot ? 1 : _notHot;
    }
  }

}
