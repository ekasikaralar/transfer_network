# Pooled input data

Empty by design.  The pooled branch consumes the same observations as the
separate branch, so the sampling lives in one place:

    ../../separate/input_data/m_net_input_data.py

That script writes one state file and one gradient file per grid point.  The
pooled network reads all of them and concatenates.
