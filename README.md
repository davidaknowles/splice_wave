Dependencies should be pretty minimal: `numpy`, `torch`, `torch_xla` and `libtpu`. 

To get the data run `get_data.sh`, which will download from public URLs. 

Optional: compile Cython `one_hot` encoding (requires `Cython` to be installed)
```
python setup.py build_ext --inplace
```
Will speed things up a bit: if not done a pure Python version will be used. 

`train.py` runs the actual model training. 