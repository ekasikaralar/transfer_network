# Pooled output data

Empty by design.  The reference linear program does not depend on how its
instances are later grouped for training, so it is solved once, per grid point,
by

    ../../separate/output_data/d_term_offline.jl

The pooled network reads those per-interval drift vectors and concatenates
them, adding the grid point as an input feature.  Keeping a single
implementation of the program means the two branches are trained on identical
targets and any difference between them is attributable to the network alone.
