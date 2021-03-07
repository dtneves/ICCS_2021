########################################################################################################################
# Research Centers:
# -----------------
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
# Intelligent Analytics for Massive Data -- IAM
# German Research Center for Artificial Intelligence -- DFKI
# Deutsches Forschungszentrum für Künstliche Intelligenz -- DFKI
# https://www.dfki.de/web/
#
#
# Description:
# ------------
# This module allows to impute missing values on incomplete datasets, more details are provided in [1].
# The missingness can be due to several factors and, usually, it falls into one of the following categories [3]:
#  * Missing Completly at Random (MCAR).
#  * Missing at Random (MAR).
#  * Missing Not at Random (MNAR).
# As mentioned in [4], "Rubin’s distinction is important for understanding why some methods will work, and others not.
# His theory lays down the conditions under which a missing data method can provide valid statistical inferences.
# Most simple fixes only work under the restrictive and often unrealistic MCAR assumption.
# If MCAR is implausible, such methods can provide biased estimates.".
# The implementation of advanced techniques to impute missing values on an incomplete dataset is planned,
# those will be based on metadata and domain knowledge.
# One should be aware that exception handling to take care of incorrect data types, incorrect parameters' values, and
# so forth is, typically, NOT performed, the rule is: We are all grown up (Python) programmers!
#
#
# Moto:
# -----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Related Work:
# -------------
#   * https://github.com/epsilon-machine/missingpy
#   * https://github.com/eltonlaw/impyute
#   * https://github.com/iskandr/fancyimpute
#   * https://github.com/kearnz/autoimpute
#   * https://github.com/awslabs/datawig
#   * https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
#   * https://www.statsmodels.org/stable/api.html?#imputation
#   * https://github.com/jsyoon0823/GAIN
#
#
# References:
# -----------
#  [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
#      "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
#      International Conference on Computational Science (ICCS), 2021.
#  [2] Jinsung Yoon, James Jordon, Mihaela van der Schaar,
#      "GAIN: Missing Data Imputation using Generative Adversarial Nets,"
#      International Conference on Machine Learning (ICML), 2018.
#  [3] Rubin, Donald B. "Inference and missing data." Biometrika 63.3 (1976): 581-592.
#  [4] Van Buuren, Stef. Flexible imputation of missing data. Chapman and Hall/CRC, 2018.
#
#
# Authors:
# --------
# diogo telmo neves -- {dneves@di.uminho.de, diogo-telmo.neves@charite.de}
#
#
# Copyright:
# ----------
# Copyright (c) 2020 diogo telmo neves.
# All rights reserved.
#
#
# Conditions:
# -----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER:
# -----------
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Date:
# -----
# December 2020
########################################################################################################################

from purify.imputation.gain import SGAIN        # the class that implements the SGAIN algorithm
from purify.imputation.gain import WSGAIN_CP    # the class that implements the WSGAIN-CP algorithm
from purify.imputation.gain import WSGAIN_GP    # the class that implements the WSGAIN-GP algorithm

# __all__ = [
#     'SGAIN',      # the class that implements the SGAIN algorithm
#     'WSGAIN-CP',  # the class that implements the WSGAIN-CP algorithm
#     'WSGAIN-GP',  # the class that implements the WSGAIN-GP algorithm
# ]

