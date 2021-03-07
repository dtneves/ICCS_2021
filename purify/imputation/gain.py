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

import abc

import numpy as np

import tensorflow as tf

from tensorflow.python.framework.ops import Operation, Tensor
from tensorflow.python.ops.variables import RefVariable

from sklearn.preprocessing import MinMaxScaler

from GAIN.utils import sample_batch_index, rounding

from tqdm import tqdm

from typing import Any, Dict, List, Tuple


tf.compat.v1.disable_v2_behavior()


class SGAIN:
    """"This class implements the Slim GAIN (SGAIN) algorithm [1].

    References:
        [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
            "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
            International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any]):
        self.scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1.00, +1.00))
        self.data: np.ndarray = data.copy()
        self.data_miss: np.ndarray = self.scaler.fit_transform(X=self.data)
        self.data_mask: np.ndarray = 1 - np.isnan(self.data)
        self.n_obs: int = self.data.shape[0]
        self.m_dim: int = self.data.shape[1]
        # handling algorithm parameters, ensure that if one is absent then its default value is used
        self.batch_size: int = algo_parameters['batch_size'] if 'batch_size' in algo_parameters \
            else int(np.ceil(np.sqrt(self.n_obs)))
        self.alpha: float = algo_parameters['alpha'] if 'alpha' in algo_parameters else 100
        self.optimizer: str = algo_parameters['optimizer'] if 'optimizer' in algo_parameters else 'GDA'
        self.learn_rate: float = algo_parameters['learn_rate'] if 'learn_rate' in algo_parameters else 0.001
        self.beta_1: float = algo_parameters['beta_1'] if 'beta_1' in algo_parameters else 0.900
        self.beta_2: float = algo_parameters['beta_2'] if 'beta_2' in algo_parameters else 0.999
        self.decay: float = algo_parameters['decay'] if 'decay' in algo_parameters else 0.900
        self.momentum: float = algo_parameters['momentum'] if 'momentum' in algo_parameters else 0.000
        self.epsilon: float = algo_parameters['epsilon'] if 'epsilon' in algo_parameters else 1e-8
        self.n_iterations: int = algo_parameters['n_iterations'] if 'n_iterations' in algo_parameters else 1000
        self.verbose: bool = algo_parameters['verbose'] == 'True' if 'verbose' in algo_parameters else False
        # replace missing values by zero, later on these will be imputed see `impute()` method
        self.data_miss = np.nan_to_num(x=self.data_miss, nan=0.00)
        # build the Generative Adversarial Network (GAN) architecture
        self.gan_architecture()

    def gan_architecture(self) -> None:
        self.X: Tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.m_dim])  # data Tensor
        self.M: Tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.m_dim])  # mask Tensor
        # noise Tensor (data + noise in missing values)
        self.Z: Tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.m_dim])

        self.G_W1: RefVariable = tf.compat.v1.Variable(
            initial_value=tf.random.uniform(shape=[2 * self.m_dim, self.m_dim], minval=-0.01, maxval=+0.01))
        self.G_b1: RefVariable = tf.compat.v1.Variable(initial_value=tf.zeros(shape=[self.m_dim]))

        self.G_W2: RefVariable = tf.compat.v1.Variable(
            initial_value=tf.random.uniform(shape=[self.m_dim, self.m_dim], minval=-0.01, maxval=+0.01))
        self.G_b2: RefVariable = tf.compat.v1.Variable(initial_value=tf.zeros(shape=[self.m_dim]))

        self.theta_G: List[: RefVariable] = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.D_W1: RefVariable = tf.compat.v1.Variable(
            initial_value=tf.random.uniform(shape=[self.m_dim, self.m_dim], minval=-0.01, maxval=+0.01))
        self.D_b1: RefVariable = tf.compat.v1.Variable(initial_value=tf.zeros(shape=[self.m_dim]))

        self.D_W2: RefVariable = tf.compat.v1.Variable(
            initial_value=tf.random.uniform(shape=[self.m_dim, self.m_dim], minval=-0.01, maxval=+0.01))
        self.D_b2: RefVariable = tf.compat.v1.Variable(initial_value=tf.zeros(shape=[self.m_dim]))

        self.theta_D: List[: RefVariable] = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        self.G_sample: Tensor = self.generator(z=self.Z, m=self.M)
        self.D_real: Tensor = self.discriminator(x=self.X)
        self.D_fake: Tensor = self.discriminator(x=self.G_sample)

        self.MSE_loss: Tensor = tf.reduce_mean(
            input_tensor=(self.M * (self.X - self.G_sample)) ** 2) / tf.reduce_mean(input_tensor=self.M)
        self.G_loss: Tensor = -tf.reduce_mean(input_tensor=((1 - self.M) * self.D_fake)) + self.alpha * self.MSE_loss
        self.D_loss: Tensor = tf.reduce_mean(
            input_tensor=(self.M * self.D_real)) - tf.reduce_mean(input_tensor=((1 - self.M) * self.D_fake))

        # the optimizer plays the minimax two-player game:
        #  - minimize the loss function of the generator
        #  - maximize the loss function of the discriminator, which is the same as
        #    minimize the loss function of the discriminator and multiply it by minus one
        if self.optimizer == 'GDA':
            self.G_solver: Operation = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=self.learn_rate).minimize(loss=self.G_loss, var_list=self.theta_G)
            self.D_solver: Operation = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=self.learn_rate).minimize(loss=-self.D_loss, var_list=self.theta_D)
        elif self.optimizer == 'RMSProp':
            self.G_solver: Operation = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.learn_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon).minimize(
                loss=self.G_loss, var_list=self.theta_G)
            self.D_solver: Operation = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.learn_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon).minimize(
                loss=-self.D_loss, var_list=self.theta_D)
        else:  # self.optimizer == 'Adam':
            self.G_solver: Operation = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learn_rate, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.epsilon).minimize(
                loss=self.G_loss, var_list=self.theta_G)
            self.D_solver: Operation = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learn_rate, beta1=self.beta_1, beta2=self.beta_2, epsilon=self.epsilon).minimize(
                loss=-self.D_loss, var_list=self.theta_D)

    def generator(self, z: Tensor, m: Tensor) -> Tensor:
        G_h1: Tensor = tf.nn.relu(features=(tf.matmul(a=tf.concat(values=[z, m], axis=1), b=self.G_W1) + self.G_b1))

        return tf.nn.tanh(x=(tf.matmul(a=G_h1, b=self.G_W2) + self.G_b2))  # returns `G_prob`, which is a Tensor

    def discriminator(self, x: Tensor) -> Tensor:
        D_h1: Tensor = tf.nn.relu(features=(tf.matmul(a=x, b=self.D_W1) + self.D_b1))

        return tf.nn.tanh(x=(tf.matmul(a=D_h1, b=self.G_W2) + self.G_b2))  # returns `D_prob`, which is a Tensor

    @staticmethod
    def sample_z(n_rows: int, m_cols: int, feature_range: Tuple[float, float] = (-0.01, +0.01)) -> np.ndarray:
        return np.random.uniform(low=feature_range[0], high=feature_range[1], size=[n_rows, m_cols])

    def impute(self, sess: tf.compat.v1.Session) -> np.ndarray:
        Z_all: np.ndarray = self.data_mask * self.data_miss + (1 - self.data_mask) * SGAIN.sample_z(
            n_rows=self.n_obs, m_cols=self.m_dim)
        imputed_data: np.ndarray = sess.run(
            fetches=[self.G_sample], feed_dict={self.M: self.data_mask, self.Z: Z_all})[0]

        imputed_data = self.scaler.inverse_transform(
            X=(self.data_mask * self.data_miss + (1 - self.data_mask) * imputed_data))
        imputed_data = rounding(imputed_data=imputed_data, data_x=self.data)

        return imputed_data

    def execute(self) -> np.ndarray:
        """"This method implements the Slim GAIN (SGAIN) algorithm [1].

        References:
            [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
                "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
                International Conference on Computational Science (ICCS), 2021.
        """
        sess: tf.compat.v1.Session = tf.compat.v1.Session()

        sess.run(fetches=tf.compat.v1.global_variables_initializer())

        for iteration in tqdm(range(self.n_iterations)):
            indices_mb: List[int] = sample_batch_index(total=self.n_obs, batch_size=self.batch_size)
            X_mb: np.ndarray = self.data_miss[indices_mb, :]
            M_mb: np.ndarray = self.data_mask[indices_mb, :]
            Z_mb: np.ndarray = M_mb * X_mb + (1 - M_mb) * SGAIN.sample_z(n_rows=self.batch_size, m_cols=self.m_dim)
            D_loss_curr: float
            G_loss_curr: float
            MSE_loss_curr: float

            _, D_loss_curr = sess.run(
                fetches=[self.D_solver, self.D_loss],
                feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            _, G_loss_curr, MSE_loss_curr = sess.run(
                fetches=[self.G_solver, self.G_loss, self.MSE_loss],
                feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            if self.verbose and (iteration % (self.n_iterations / 10) == 0):
                tqdm.write(f"Iteration: {iteration}; "
                           f"D loss: {D_loss_curr:.4}; G_loss: {G_loss_curr:.4}; MSE_loss: {MSE_loss_curr:.4}")

        return self.impute(sess=sess)


class WSGAIN(SGAIN):
    """"This class is an abstract backbone skeleton that specifies the common attributes and methods that
    its subclasses share (see :class:`purify.imputation.gain.WSGAIN_CP` and :class:`purify.imputation.gain.WSGAIN_GP`).
    In other words, this class implements what the Wasserstein Slim GAIN with Clipping Penalty (WSGAIN-CP) and
    the Wasserstein Slim GAIN with Gradient Penalty (WSGAIN-GP) algorithms [1] have in common.

    References:
        [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
            "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
            International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any]):
        super().__init__(data=data, algo_parameters=algo_parameters)
        # NOTE: THIS DIVISION IS AN HACK TO PROMOTE FAIR COMPARISONS AS EXPLAINED IN THE ICCS 2021 PAPER,
        #       IT WILL ENSURE THAT THE SUM OF ITERATIONS TO TRAIN THE GENERATOR AND THE CRITIC OF THIS IMPLEMENTATION
        #       IS (APPROXIMATELY) THE SAME OT THE COUNTERPART ONES OF GAIN AND SGAIN
        self.n_iterations: int = int(np.ceil(self.n_iterations / 3))
        self.n_critic: int = algo_parameters['n_critic'] if 'n_critic' in algo_parameters else 5

    @abc.abstractmethod
    def refine_gan_architecture(self, algo_parameters: Dict[str, Any]) -> None:
        pass

    def discriminator(self, x: Tensor) -> Tensor:
        D_h1: Tensor = tf.nn.relu(features=(tf.matmul(a=x, b=self.D_W1) + self.D_b1))

        return tf.matmul(a=D_h1, b=self.D_W2) + self.D_b2  # returns `D_prob`, which is a Tensor

    @abc.abstractmethod
    def execute(self) -> np.ndarray:
        return None


class WSGAIN_CP(WSGAIN):
    """"This class implements the Wasserstein Slim GAIN with Clipping Penalty (WSGAIN-CP) algorithm [1].

    References:
        [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
            "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
            International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any]):
        super().__init__(data=data, algo_parameters=algo_parameters)
        # some refinement needs to be introduced into the GAN architecture due to the clipping penalty
        self.refine_gan_architecture(algo_parameters=algo_parameters)

    def refine_gan_architecture(self, algo_parameters: Dict[str, Any]) -> None:
        clip_value: float = algo_parameters['clip_value'] if 'clip_value' in algo_parameters else 0.01
        clip_value_min: float = min(-1 * clip_value, +1 * clip_value)
        clip_value_max: float = max(-1 * clip_value, +1 * clip_value)

        self.clip_D: List[Tensor] = [p.assign(value=tf.clip_by_value(
            t=p, clip_value_min=clip_value_min, clip_value_max=clip_value_max)) for p in self.theta_D]

    def execute(self) -> np.ndarray:
        """"This method implements the Wasserstein Slim GAIN with Clipping Penalty (WSGAIN-CP) algorithm [1].

        References:
            [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
                "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
                International Conference on Computational Science (ICCS), 2021.
        """
        sess: tf.compat.v1.Session = tf.compat.v1.Session()

        sess.run(fetches=tf.compat.v1.global_variables_initializer())

        for iteration in tqdm(range(self.n_iterations)):
            D_loss_curr: float
            G_loss_curr: float
            MSE_loss_curr: float

            for _ in range(self.n_critic):  # train the critic a few times more per each train of the generator
                indices_mb: List[int] = sample_batch_index(total=self.n_obs, batch_size=self.batch_size)
                X_mb: np.ndarray = self.data_miss[indices_mb, :]
                M_mb: np.ndarray = self.data_mask[indices_mb, :]
                Z_mb: np.ndarray = M_mb * X_mb + (1 - M_mb) * SGAIN.sample_z(n_rows=self.batch_size, m_cols=self.m_dim)

                _, D_loss_curr, _ = sess.run(
                    fetches=[self.D_solver, self.D_loss, self.clip_D],
                    feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            _, G_loss_curr, MSE_loss_curr = sess.run(
                fetches=[self.G_solver, self.G_loss, self.MSE_loss],
                feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            if self.verbose and (iteration % (self.n_iterations / 10) == 0):
                tqdm.write(f"Iteration: {iteration}; "
                           f"D loss: {D_loss_curr:.4}; G_loss: {G_loss_curr:.4}; MSE_loss: {MSE_loss_curr:.4}")

        return self.impute(sess=sess)


class WSGAIN_GP(WSGAIN):
    """"This class implements the Wasserstein Slim GAIN with Gradient Penalty (WSGAIN-GP) algorithm [1].

    References:
        [1] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença,
            "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation,"
            International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any]):
        super().__init__(data=data, algo_parameters=algo_parameters)
        self.lambd: float = algo_parameters['lambd'] if 'lambd' in algo_parameters else 10
        # some refinement needs to be introduced into the GAN architecture due to the gradient penalty
        self.refine_gan_architecture(algo_parameters=algo_parameters)

    def refine_gan_architecture(self, algo_parameters: Dict[str, Any]) -> None:
        eps: np.ndarray = SGAIN.sample_z(n_rows=self.batch_size, m_cols=self.m_dim)
        X_inter: Tensor = eps * (self.M * self.X) + (1 - eps) * ((1 - self.M) * self.G_sample)
        grad: Tensor = tf.gradients(ys=self.discriminator(x=X_inter), xs=[X_inter])[0]
        # note: `self.epsilon` is used as a workaround to the bug mentioned in
        #       https://github.com/pytorch/pytorch/issues/2534
        #       however, it is NOT enough and one has to ensure that
        #       the learning rate is kept smaller or equal to `1e-3`
        # grad_norm: Tensor = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1))
        grad_norm: Tensor = tf.sqrt(self.epsilon + tf.reduce_sum(input_tensor=(grad ** 2), axis=1))
        grad_pen: Tensor = self.lambd * tf.reduce_mean(input_tensor=((grad_norm - 1) ** 2))

        self.D_loss: Tensor = tf.reduce_mean(input_tensor=(self.M * self.D_real)) - tf.reduce_mean(
            input_tensor=((1 - self.M) * self.D_fake)) + grad_pen

    def execute(self) -> np.ndarray:
        sess: tf.compat.v1.Session = tf.compat.v1.Session()

        sess.run(fetches=tf.compat.v1.global_variables_initializer())

        for iteration in tqdm(range(self.n_iterations)):
            D_loss_curr: float
            G_loss_curr: float
            MSE_loss_curr: float

            for _ in range(self.n_critic):  # train the critic a few times more per each train of the generator
                indices_mb: List[int] = sample_batch_index(total=self.n_obs, batch_size=self.batch_size)
                X_mb: np.ndarray = self.data_miss[indices_mb, :]
                M_mb: np.ndarray = self.data_mask[indices_mb, :]
                Z_mb: np.ndarray = M_mb * X_mb + (1 - M_mb) * SGAIN.sample_z(n_rows=self.batch_size, m_cols=self.m_dim)

                _, D_loss_curr = sess.run(
                    fetches=[self.D_solver, self.D_loss],
                    feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            _, G_loss_curr, MSE_loss_curr = sess.run(
                fetches=[self.G_solver, self.G_loss, self.MSE_loss],
                feed_dict={self.X: X_mb, self.M: M_mb, self.Z: Z_mb})

            if self.verbose and (iteration % (self.n_iterations / 10) == 0):
                tqdm.write(f"Iteration: {iteration}; "
                           f"D loss: {D_loss_curr:.4}; G_loss: {G_loss_curr:.4}; MSE_loss: {MSE_loss_curr:.4}")

        return self.impute(sess=sess)

