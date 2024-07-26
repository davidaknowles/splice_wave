To get the data run `get_data.sh`, which will download from public URLs. 

Optional: compile Cython `one_hot` encoding
```
python setup.py build_ext --inplace
```
Will speed things up a bit: if not done a pure Python version will be used. 

`train.py` runs the actual model training. 