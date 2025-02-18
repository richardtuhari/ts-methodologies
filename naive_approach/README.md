# Notes

This framework implements a naive walk forwarder, to test and evaluate various preprocessing transformations.
Figures 3 and 4 are produced using code provided here.

## Files

* `model_mlp.py` implements _MLP_
* `model_lst.py` implements _LST_
* `transformer.py` implements various transformers: _Z_, _S_, _ZoS_, _ZoFoS_ with coupled and decoupled inverse, and a future biased _Z_
* `wrapper.py` glues transformer and model to have an operation chain.
* `simulate.py` teach and evaluate a system with given parametrization.


