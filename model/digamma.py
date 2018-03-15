import tensorflow as tf
# @MISC {1446110,
# TITLE = {Approximating the Digamma function},
# AUTHOR = {njuffa (https://math.stackexchange.com/users/114200/njuffa)},
# HOWPUBLISHED = {Mathematics Stack Exchange},
# NOTE = {URL:https://math.stackexchange.com/q/1446110 (version: 2015-09-22)},
# EPRINT = {https://math.stackexchange.com/q/1446110},
# URL = {https://math.stackexchange.com/q/1446110}}

def digamma_approx(x):
    def digamma_over_one(x):
        return tf.log(x + 0.4849142940227510) \
                - 1/(1.0271785180163817*x)
    return digamma_over_one(x+1) - 1./x
