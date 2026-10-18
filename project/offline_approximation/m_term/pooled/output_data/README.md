# Pooled output data

Empty by design.  The linear program does not depend on how its instances are
later grouped for training, so it is solved once, per grid point, by

    ../../separate/output_data/m_term_offline.jl

The pooled network reads those per-interval objective values and concatenates
them, adding the grid point as an input feature.  