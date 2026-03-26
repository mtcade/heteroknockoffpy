"""
    From:
    https://github.com/firmai/tsgan/blob/master/alg/knockoffgan/KnockoffGAN.py
    Jinsung Yoon (9/27/2018)
    
    Generates Knockoffs with "Generative Adversarial Networks"
"""

# 2024-10-24: Keras changes have broken calls of tf.{method} on keras objects
# See: "#&&

#%% Necessary Packages
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import logging
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#%% KnockoffGAN Function
'''
Inputs:
x_train: Training data
lamda: Power network parameter = 0.01
mu: WGAN parameter = 1
'''
# UPDATE: to make gradients work
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger()

#%% Necessary Functions
    
# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    # UPDATE: should be tf.random.normal
    # UPDATE: should be tf.random.normal
    if False:
        result = tf.random_normal(
            shape = size,
            stddev = xavier_stddev
        )
    else:
        result = tf.random.normal(
            shape = size,
            stddev = xavier_stddev
        )
    return result
#/def xavier_init

# 2. Sample from normal distribution: Random variable generation
def sample_Z(
    m: int,
    n: int,
    x_name: str
    ) -> np.ndarray:
    
    result: np.ndarray
    if ((x_name == 'Normal') | (x_name == 'AR_Normal')):
        result = np.random.normal(0., np.sqrt(1./3000), size = [m, n]).copy()

    elif ((x_name == 'Uniform') | (x_name == 'AR_Uniform')):
        result = np.random.uniform(-3*np.sqrt(1./3000),3*np.sqrt(1./3000),[m,n]).copy()
    #/switch x_name
    
    return result
# 3. Sample from the real data (Mini-batch index sampling)
def sample_X(m, n):
    return np.random.permutation(m)[:n].copy()
            
# 4. Permutation for MINE computation
def Permute (x):
    n = len(x[:,0])
    idx = np.random.permutation(n)
    out = x[idx,:].copy()
    return out
    
# 5. Bernoulli sampling for Swap and Hint variables
def sample_SH(m, n, p):
    return np.random.binomial(1, p, [m,n]).copy()

# TODO: is `x_train` type np.ndarray?
# TODO: x_name typehints?
def KnockoffGAN (
    x_train: np.ndarray,
    x_name: str,
    lamda = 1, # 2025-01-16: previously 10
    mu = 1,
    lam = 10,
    lr = 1e-4,
    mb_size=128,
    niter=2000
    ):
    
    tf_debug = False
    if tf_debug:
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        config = tf.ConfigProto()
        config.log_device_placement=True
        config.gpu_options.allow_growth = True
    #
    else:
        run_opts = None
        config = None
    #if tf_debug/else
    
    #%% Parameters
    # 1. # of samples
    n = len(x_train[:,0])
    
    # 2. # of features
    x_dim = len(x_train[0,:])
    
    # 3. # of random dimensions
    z_dim = int(x_dim)
        
    # 4. # of hidden dimensions
    h_dim = int(x_dim)
        
    # 5. # of minibatch
    # mb_size = 128
    
    # 6. WGAN parameters
    # 2024-04-28: Moved to inputs
    #lam =
    #lr =
        
    # 1. Feature
    ## UPDATE: placeholder deprecated in tf v2
    ## See: https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder
    #
    TF_COMPAT: bool = True
    
    if False:
        # 1. Feature
        X = tf.placeholder(tf.float32, shape = [None, x_dim])
        # 2. Feature (Permute)
        X_hat = tf.placeholder(tf.float32, shape = [None, x_dim])
        # 3. Random Variable
        Z = tf.placeholder(tf.float32, shape = [None, z_dim])
        # 4. Swap
        S = tf.placeholder(tf.float32, shape = [None, x_dim])
        # 5. Hint
        H = tf.placeholder(tf.float32, shape = [None, x_dim])
        #
    elif TF_COMPAT:
        X = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])
        # 2. Feature (Permute)
        X_hat = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])
        # 3. Random Variable
        Z = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
        # 4. Swap
        S = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])
        # 5. Hint
        H = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])
        #
    else:
        # UPDATES: remove the leading [None,...] from shape,
        #   since input does it automatically
        
        X = tf.keras.Input(
            dtype = tf.float32,
            shape = [x_dim]
        )
        # 2. Feature (Permute)
        X_hat = tf.keras.Input(
            dtype = tf.float32,
            shape = [x_dim]
        )
        # 3. Random Variable
        Z = tf.keras.Input(
            dtype = tf.float32,
            shape = [z_dim]
        )
        # 4. Swap
        S = tf.keras.Input(
            dtype = tf.float32, shape = [x_dim]
        )
        # 5. Hint
        H = tf.keras.Input(
            dtype = tf.float32,
            shape = [x_dim]
        )
    #
         
    #%% Network Building
    
    #%% 1. Discriminator
    # Input: Swap (X, tilde X) and Hint
    ## UPDATE: have to specify "variable_def"
    D_W1 = tf.Variable(
        initial_value = xavier_init([x_dim + x_dim + x_dim, h_dim])
    )
    D_b1 = tf.Variable(
        initial_value = tf.zeros(shape=[h_dim])
    )
    
    D_W2 = tf.Variable(
        initial_value = xavier_init([h_dim,x_dim])
    )
    D_b2 = tf.Variable(
        initial_value = tf.zeros(shape=[x_dim])
    )
    
    theta_D = [D_W1, D_W2, D_b1, D_b2]
        
    #%% 2. WGAN Discriminator
    # Input: tilde X
    WD_W1 = tf.Variable(
        initial_value = xavier_init([x_dim, h_dim])
    )
    WD_b1 = tf.Variable(
        initial_value = tf.zeros(shape=[h_dim])
    )
    
    WD_W2 = tf.Variable(
        initial_value = xavier_init([h_dim,1])
    )
    WD_b2 = tf.Variable(
        initial_value = tf.zeros(shape=[1])
    )
    
    theta_WD = [WD_W1, WD_W2, WD_b1, WD_b2]
    
    #%% 3. Generator
    # Input: X and Z
    G_W1 = tf.Variable(
        initial_value = xavier_init([x_dim + z_dim, h_dim])
    )
    G_b1 = tf.Variable(
        initial_value = tf.zeros(shape=[h_dim])
    )
    
    G_W2 = tf.Variable(
        initial_value = xavier_init([h_dim,x_dim])
    )
    G_b2 = tf.Variable(
        initial_value = tf.zeros(shape=[x_dim])
    )
        
    theta_G = [G_W1, G_W2, G_b1, G_b2]
    
    #%% 4. MINE
    # Input: X and tilde X
    # For X
    M_W1A = tf.Variable(
        initial_value = xavier_init([x_dim])
    )
    M_W1B = tf.Variable(
        initial_value = xavier_init([x_dim])
    )
    M_b1 = tf.Variable(
        initial_value = tf.zeros(shape=[x_dim])
    )
    
    # For tilde X
    M_W2A = tf.Variable(
        initial_value =xavier_init([x_dim])
    )
    M_W2B = tf.Variable(
        initial_value = xavier_init([x_dim])
    )
    M_b2 = tf.Variable(
        initial_value = tf.zeros(shape=[x_dim])
    )
    
    # Combine
    M_W3 = tf.Variable(
        initial_value = xavier_init([x_dim])
    )
    M_b3 = tf.Variable(
        initial_value = tf.zeros(shape=[x_dim])
    )
        
    theta_M = [M_W1A, M_W1B, M_W2A, M_W2B, M_W3, M_b1, M_b2, M_b3]
    
    #%% Functions
    # 1. Generator
    def generator(x, z):
        inputs = tf.concat(axis=1, values = [x, z])
        #inputs = tf.keras.layers.Concatenate(axis = 1)([x,z])
        
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
        G_out = (tf.matmul(G_h1, G_W2) + G_b2)
            
        return G_out
    
    # 2. Discriminator
    def discriminator(sA, sB, h):
    
        inputs = tf.concat(axis=1, values = [sA, sB, h])
        
        D_h1 = tf.nn.tanh(tf.matmul(inputs, D_W1) + D_b1)
        D_out = tf.nn.sigmoid(tf.matmul(D_h1, D_W2) + D_b2)
            
        return D_out
    
    # 3. WGAN Discriminator
    def WGAN_discriminator(x):
        WD_h1 = tf.nn.relu(tf.matmul(x, WD_W1) + WD_b1)
        WD_out = (tf.matmul(WD_h1, WD_W2) + WD_b2)
            
        return WD_out
        
     # 4. MINE
    def MINE(x, x_hat):
        M_h1 = tf.nn.tanh(M_W1A * x + M_W1B * x_hat + M_b1)
        M_h2 = tf.nn.tanh(M_W2A * x + M_W2B * x_hat + M_b2)
        M_out = (M_W3 * (M_h1 + M_h2) + M_b3)
        
        Exp_M_out = tf.exp(M_out)
        
        return M_out, Exp_M_out
        
    #%% Combination across the networks
    # 1. Generater Knockoffs
    G_sample = generator(X,Z)
    
    # 2. WGAN Outputs for real and fake
    WD_real = WGAN_discriminator(X)
    WD_fake = WGAN_discriminator(G_sample)
    
    # 3. Generate swapping (X, tilde X)
    SwapA = S * X + (1-S) * G_sample
    SwapB = (1-S) * X + S * G_sample
    
    # 4. Discriminator output
    # (X, tilde X) is SwapA, SwapB. Hint is generated by H * S
    D_out = discriminator(SwapA, SwapB, H*S)
    
    # 5. MINE Computation
    # Without permutation
    M_out, _ = MINE(X, G_sample)
    # With permutation
    _, Exp_M_out = MINE(X_hat, G_sample)
    
    # 6. WGAN Loss Replacement of Clipping algorithm to Penalty term
    # 1. Line 6 in Algorithm 1
    # UPDATE: get rid of random_uniform
    eps = tf.random.uniform([mb_size, 1], minval = 0., maxval = 1.)
    X_inter = eps*X + (1. - eps) * G_sample
    
    # 2. Line 7 in Algorithm 1
    # TODO: v2 requires GradientTape
    grad = tf.gradients(WGAN_discriminator(X_inter), [X_inter])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2 + 1e-8, axis = 1))
    grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)
    
    #%% Loss function
    # 1. WGAN Loss
    WD_loss = tf.reduce_mean(WD_fake) - tf.reduce_mean(WD_real) + grad_pen

    # 2. Discriminator loss
    # UPDATE: to math.log
    D_loss = -tf.reduce_mean(S * (1-H) * tf.math.log(D_out + 1e-8) + (1-S) * (1-H) * tf.math.log(1 - D_out + 1e-8))
    
    # 3. MINE Loss
    M_loss = tf.reduce_sum( tf.reduce_mean(M_out, axis = 0) - tf.math.log(tf.reduce_mean(Exp_M_out, axis = 0)) )
    
    # 4. Generator loss
    G_loss =  - D_loss + mu * -tf.reduce_mean(WD_fake) + lamda * M_loss
    
    # Solver
    # UPDATE: tf.train.AdamOptimizer is v1, switched to
    #   tf.compat.v1.train.AdamOptimizer
    WD_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(WD_loss, var_list = theta_WD))
    D_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(D_loss, var_list = theta_D))
    G_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(G_loss, var_list = theta_G))
    M_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(-M_loss, var_list = theta_M))
            
        
    #%% Sessions
    # UPDATE: to tf.compat.v1.Session
    if tf_debug:
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer(), options=run_opts)
    else:
        sess = tf.compat.v1.Session()
        # UPDATE: compat.v1
        sess.run(tf.compat.v1.global_variables_initializer())

    #%% Iterations
    for it in tqdm(range(niter)):

        for dummy_range in range(5):
            #%% WGAN, Discriminator and MINE Training
            
            # Random variable generation
            Z_mb = sample_Z(mb_size, z_dim, x_name)
                
            # Minibatch sampling
            X_idx = sample_X(n,mb_size)
            X_mb = x_train[X_idx,:].copy()
            X_perm_mb = Permute(X_mb)
            
            # Swap generation
            S_mb = sample_SH(mb_size, x_dim, 0.5)
            
            # Hint generation
            H_mb = sample_SH(mb_size, x_dim, 0.9)
                
            # 1. WGAN Training
            
            _, WD_loss_curr = sess.run([WD_solver, WD_loss], feed_dict = {X: X_mb, Z: Z_mb, X_hat: X_perm_mb, S: S_mb, H: H_mb}, options=run_opts)
            
            # 2. Discriminator Training
            # print('discriminator training')
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: Z_mb, X_hat: X_perm_mb, S: S_mb, H: H_mb}, options=run_opts)
            
            # 3. MINE Training
            # print('mine training')
            _, M_loss_curr = sess.run([M_solver, M_loss], feed_dict = {X: X_mb, Z: Z_mb, X_hat: X_perm_mb, S: S_mb, H: H_mb}, options=run_opts)
                        
        #%% Generator Training
        
        # Random variable generation
        Z_mb = sample_Z(mb_size, z_dim, x_name)
                
        # Minibatch sampling
        X_idx = sample_X(n,mb_size)
        X_mb = x_train[X_idx,:].copy()
        X_perm_mb = Permute(X_mb)
                        
        # Swap generation
        S_mb = sample_SH(mb_size, x_dim, 0.5)
            
        # Hint generation
        H_mb = sample_SH(mb_size, x_dim, 0.0)
        
        # Generator training
        # print('gen training')
        _, G_loss_curr, G_sample_curr = sess.run([G_solver, G_loss, G_sample], feed_dict = {X: X_mb, Z: Z_mb, X_hat: X_perm_mb, S: S_mb, H: H_mb}, options=run_opts)
      
    #%% Output
    #print('last session run')
    X_knockoff = sess.run([G_sample], feed_dict = {X: x_train, Z: sample_Z(n, z_dim, x_name)}, options=run_opts)[0]
    # X_knockoff = sess.run([G_sample], feed_dict = {X: x_train, Z: sample_Z(n, z_dim, x_name)})[0]
    #print('closing session')
    sess.close()
    # UPDATE: compat.v1
    tf.compat.v1.reset_default_graph()
    return X_knockoff
#/def knockoffGAN

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i')
    parser.add_argument(
        '-o')
    parser.add_argument(
        '--bs', default=128, type=int)
    parser.add_argument(
        '--it', default=2000, type=int)
    parser.add_argument(
        '--target')
    parser.add_argument(
        '--xname', default='Normal', help='Sample distribution [Normal, Uniform]')
    parser.add_argument(
        '--scale', default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()
    df = pd.read_csv(args.i)
    niter = args.it
    use_scale = args.scale
    x_name = args.xname
    lbl = args.target
    features = list(df.columns)
    features.remove(lbl)

    # scale/normalize dataset
    range_scaler = (0, 1)
    scaler = MinMaxScaler(feature_range=range_scaler)

    x = df[features]

    if use_scale:
        scaler.fit(x)
        x = scaler.transform(x)
    else:
        x = x.values

    x_k = KnockoffGAN(
        x,
        x_name,
        mb_size=args.bs,
        niter=niter)
    df_k = pd.DataFrame(x_k, columns=features)
    df_k[lbl] = df[lbl]
    df_k.to_csv(args.o, index=False)
