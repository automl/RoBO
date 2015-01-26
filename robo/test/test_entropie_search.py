import os
import random
import errno
import unittest
import matplotlib; #matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb 
import numpy as np
#pb.ion()
from robo.models import GPyModel
from robo.test_functions import branin2, branin
from robo.acquisition import PI, UCB, Entropy, EI
from robo.maximize import cma, DIRECT, grid_search
from robo.loss_functions import logLoss

here = os.path.abspath(os.path.dirname(__file__))

#
# Plotting Stuff.
# It's really ugly until now, because it only fitts to 1D and to the branin function
# where the second domension is 12
#

obj_samples = 700
plot_min = -8
plot_max = 19
plotting_range = np.linspace(plot_min, plot_max, num=obj_samples)
second_branin_arg = np.empty((obj_samples,))
second_branin_arg.fill(12)
branin_arg = np.append(np.reshape(plotting_range, (obj_samples, 1)), np.reshape(second_branin_arg, (obj_samples, 1)), axis=1)
branin_result = branin(branin_arg)
# branin_result = branin_result.reshape(branin_result.shape[0],)

def _plot_model(model, acquisition_fkt, objective_fkt, i, callback):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    model.m.plot(ax=ax, plot_limits=[plot_min, plot_max])
    xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
    ax.set_ylim(min(np.min(branin_result), ylim_min), max(np.max(branin_result), ylim_max))
    c1 = np.reshape(plotting_range, (obj_samples, 1))
    c2 = acquisition_fkt(c1)
    c2 = c2*50 / np.max(c2)
    c2 = np.reshape(c2,(obj_samples,))
    ax.plot(plotting_range,c2, 'r')
    ax.plot(plotting_range, branin_result, 'black')
    callback(ax)
    fig.savefig("%s/../tmp/pmin_%s.png"%(here, i), format='png')
    fig.clf()
    plt.close()
    
class EmptySampleTestCase(unittest.TestCase):
    def setUp(self):

        dims = 1
        self.num_initial_vals = 3
        self.X_lower = np.array([-5]);#, -8])
        self.X_upper = np.array([15]);#, 19])
        #initialize the samples
        X = np.empty((self.num_initial_vals, dims))
    
        Y = np.empty((self.num_initial_vals, 1))
        self.x_values = [12.0, 4.0, 8.0, 3.0, -8.0, -4.0, 19.0, 0.1, 16.0, 15.6]
        #draw a random sample from the objective function in the
        #dimension space 
        self.objective_fkt= branin2
        for j in range(self.num_initial_vals):
            for i in range(dims):
                X[j,i] = self.x_values[j]#random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
            Y[j:] = self.objective_fkt( np.array([X[j,:]]))
        
        #
        # Building up the model
        #    
        #old gpy version 
        try:
            self.kernel = GPy.kern.rbf(input_dim=dims, variance=28.5375**2, lengthscale=5.059)
        #gpy version >=0.6
        except AttributeError, e:
            self.kernel = GPy.kern.RBF(input_dim=dims, variance=28.5375**2, lengthscale=5.059)
            
        self.model = GPyModel(self.kernel, optimize=False, noise_variance =0.0015146**2)
        self.model.train(X,Y)
        #self.model.m.optimize()
        #
        # creating an acquisition function
        #
        self.acquisition_fkt = Entropy(self.model, self.X_lower, self.X_upper)
    @unittest.skip("skip it")
    def test_pmin(self):
        for i in xrange(self.num_initial_vals, len(self.x_values)):
            self.acquisition_fkt.update(model)
            new_x = np.array(self.x_values[i]).reshape((1,1,))#  grid_search(self.acquisition_fkt, self.X_lower, self.X_upper)
            new_y = self.objective_fkt(new_x)
            def plot_pmin(ax):
                ax.plot(self.acquisition_fkt.zb,np.exp(self.acquisition_fkt.logP)*100, marker="o", color="#ff00ff", linestyle="");
                ax.plot(self.acquisition_fkt.zb,self.acquisition_fkt.logP, marker="h", color="#00a0ff", linestyle="");
                ax.text(0.95, 0.01, str(self.acquisition_fkt.current_entropy),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='#222222', fontsize=10)
            _plot_model(self.model, self.acquisition_fkt, self.objective_fkt, i, plot_pmin)
            self.model.update(np.array(new_x), np.array(new_y))

    @unittest.skip("skip it")
    def test_innovation(self):
        self.acquisition_fkt = Entropy(self.model, self.X_lower, self.X_upper, Nb = 10)
        Var= np.array( [[  493.2476,  192.0936,  189.3891,  426.5045,  419.9972,  332.1155,  483.1472,  -30.6904,  437.6728,  475.2670], 
                        [  192.0936,  267.2359,  265.4794,   88.2276,  288.7455,   45.6296,  146.3654, -112.2569,  278.3833,  242.1045], 
                        [  189.3891,  265.4794,  263.7466,   86.6537,  285.6440,   44.7031,  144.0873, -112.2385,  275.2662,  239.0783], 
                        [  426.5045,   88.2276,   86.6537,  498.2159,  274.9279,  469.3545,  471.2575,   -9.0593,  297.7650,  360.6824], 
                        [  419.9972,  288.7455,  285.6440,  274.9279,  460.5593,  179.1422,  368.1959,  -69.6378,  463.5500,  455.9222], 
                        [  332.1155,   45.6296,   44.7031,  469.3545,  179.1422,  498.5377,  396.1277,   -3.5359,  198.9042,  258.4307], 
                        [  483.1472,  146.3654,  144.0873,  471.2575,  368.1959,  396.1277,  496.6195,  -19.5925,  389.6205,  441.9061], 
                        [  -30.6904, -112.2569, -112.2385,   -9.0593,  -69.6378,   -3.5359,  -19.5925,  132.7377,  -63.3793,  -46.8737], 
                        [  437.6728,  278.3833,  275.2662,  297.7650,  463.5500,  198.9042,  389.6205,  -63.3793,  468.7849,  467.3753], 
                        [  475.2670,  242.1045,  239.0783,  360.6824,  455.9222,  258.4307,  441.9061,  -46.8737,  467.3753,  484.3670]])
        zb  = np.array([ [0.1569],  [6.4306],  [6.4735], [-2.6102],  [2.7876], [-4.3457], [-0.9570], [13.6003],  [2.4161],  [1.3773]])
        lmb = np.array([ 4.1938,  3.1827,  3.1675,  4.2753,  3.9806,  4.2928,  4.2381,  2.4608,  4.0232,  4.1183])
        Mb  = np.array([ 7.6510, 50.3527, 50.7310,  2.0226, 20.4149,  0.7512,  4.6460, 63.3195, 18.0728, 12.4844])

        self.acquisition_fkt.update(model);
        #mu, var = self.model.predict(np.array(zb), full_cov=True)
        logP,dlogPdMu,dlogPdSigma,dlogPdMudMu = self.acquisition_fkt._joint_min(Mb, Var, with_derivatives=True)

        L = self.acquisition_fkt._get_gp_innovation_local(zb);
        print L(np.array([[2.0]]))

    @unittest.skip("skip dhmc_local")
    def test_dhmc_local(self):
        self.acquisition_fn = Entropy(self.model, self.X_lower, self.X_upper, Nb = 10)

        Var= np.array( [[  493.2476,  192.0936,  189.3891,  426.5045,  419.9972,  332.1155,  483.1472,  -30.6904,  437.6728,  475.2670],
                        [  192.0936,  267.2359,  265.4794,   88.2276,  288.7455,   45.6296,  146.3654, -112.2569,  278.3833,  242.1045],
                        [  189.3891,  265.4794,  263.7466,   86.6537,  285.6440,   44.7031,  144.0873, -112.2385,  275.2662,  239.0783],
                        [  426.5045,   88.2276,   86.6537,  498.2159,  274.9279,  469.3545,  471.2575,   -9.0593,  297.7650,  360.6824],
                        [  419.9972,  288.7455,  285.6440,  274.9279,  460.5593,  179.1422,  368.1959,  -69.6378,  463.5500,  455.9222],
                        [  332.1155,   45.6296,   44.7031,  469.3545,  179.1422,  498.5377,  396.1277,   -3.5359,  198.9042,  258.4307],
                        [  483.1472,  146.3654,  144.0873,  471.2575,  368.1959,  396.1277,  496.6195,  -19.5925,  389.6205,  441.9061],
                        [  -30.6904, -112.2569, -112.2385,   -9.0593,  -69.6378,   -3.5359,  -19.5925,  132.7377,  -63.3793,  -46.8737],
                        [  437.6728,  278.3833,  275.2662,  297.7650,  463.5500,  198.9042,  389.6205,  -63.3793,  468.7849,  467.3753],
                        [  475.2670,  242.1045,  239.0783,  360.6824,  455.9222,  258.4307,  441.9061,  -46.8737,  467.3753,  484.3670]])
        zb  = np.array([ [0.1569],  [6.4306],  [6.4735], [-2.6102],  [2.7876], [-4.3457], [-0.9570], [13.6003],  [2.4161],  [1.3773]])
        lmb = np.array([[ 4.1938,  3.1827,  3.1675,  4.2753,  3.9806,  4.2928,  4.2381,  2.4608,  4.0232,  4.1183]]).T
        Mb  = np.array([ 7.6510, 50.3527, 50.7310,  2.0226, 20.4149,  0.7512,  4.6460, 63.3195, 18.0728, 12.4844])

        # mu, var = self.model.predict(np.array(zb), full_cov=True)
        logP,dlogPdM,dlogPdV,ddlogPdMdM = self.acquisition_fn._joint_min(Mb, Var, with_derivatives=True)

        L = self.acquisition_fn._get_gp_innovation_local(zb)
        T = 200
        invertsign = False
        dH_fun = self.acquisition_fn.dh_mc_local(zb,logP,dlogPdM,dlogPdV,ddlogPdMdM,T,lmb,self.X_lower,self.X_upper,invertsign,logLoss)
        print dH_fun(np.array([[2.]]))


    def test_predict_info_gain(self):
        self.acquisition_fn = Entropy(self.model, self.X_lower, self.X_upper, Nb = 10)

        self.acquisition_fn.var= np.array( [[  493.2476,  192.0936,  189.3891,  426.5045,  419.9972,  332.1155,  483.1472,  -30.6904,  437.6728,  475.2670],
                        [  192.0936,  267.2359,  265.4794,   88.2276,  288.7455,   45.6296,  146.3654, -112.2569,  278.3833,  242.1045],
                        [  189.3891,  265.4794,  263.7466,   86.6537,  285.6440,   44.7031,  144.0873, -112.2385,  275.2662,  239.0783],
                        [  426.5045,   88.2276,   86.6537,  498.2159,  274.9279,  469.3545,  471.2575,   -9.0593,  297.7650,  360.6824],
                        [  419.9972,  288.7455,  285.6440,  274.9279,  460.5593,  179.1422,  368.1959,  -69.6378,  463.5500,  455.9222],
                        [  332.1155,   45.6296,   44.7031,  469.3545,  179.1422,  498.5377,  396.1277,   -3.5359,  198.9042,  258.4307],
                        [  483.1472,  146.3654,  144.0873,  471.2575,  368.1959,  396.1277,  496.6195,  -19.5925,  389.6205,  441.9061],
                        [  -30.6904, -112.2569, -112.2385,   -9.0593,  -69.6378,   -3.5359,  -19.5925,  132.7377,  -63.3793,  -46.8737],
                        [  437.6728,  278.3833,  275.2662,  297.7650,  463.5500,  198.9042,  389.6205,  -63.3793,  468.7849,  467.3753],
                        [  475.2670,  242.1045,  239.0783,  360.6824,  455.9222,  258.4307,  441.9061,  -46.8737,  467.3753,  484.3670]])
        self.acquisition_fn.zb  = np.array([ [0.1569],  [6.4306],  [6.4735], [-2.6102],  [2.7876], [-4.3457], [-0.9570], [13.6003],  [2.4161],  [1.3773]])
        self.acquisition_fn.lmb = np.array([[ 4.1938,  3.1827,  3.1675,  4.2753,  3.9806,  4.2928,  4.2381,  2.4608,  4.0232,  4.1183]]).T
        self.acquisition_fn.mu  = np.array([ 7.6510, 50.3527, 50.7310,  2.0226, 20.4149,  0.7512,  4.6460, 63.3195, 18.0728, 12.4844])

        # mu, var = self.model.predict(np.array(zb), full_cov=True)
        self.acquisition_fn.logP,self.acquisition_fn.dlogPdMu,self.acquisition_fn.dlogPdSigma,self.acquisition_fn.dlogPdMudMu = \
            self.acquisition_fn._joint_min(self.acquisition_fn.mu, self.acquisition_fn.var, with_derivatives=True)

        self.acquisition_fn.current_entropy = - np.sum (np.exp(self.acquisition_fn.logP) * (self.acquisition_fn.logP + self.acquisition_fn.lmb) )
        self.acquisition_fn.W = np.random.randn(1, self.acquisition_fn.T)
        #self.W = np.array([[-2.17462056629639, 0.144614651456100, 0.669033435302303, -0.848829159757920, -0.0570901191929413, 0.291786264353796, -0.614618540132659, -0.0136431885670602, 0.412113496148595, -0.937149595361607, 0.835157036407438, 1.09453846090965, 0.321180732968874, 0.0151128385440506, -1.16449254121656, 0.494840747422341, -1.21584898482425, 1.59078797675859, 0.526475048595550, 0.264818591763603, -0.393597890579411, -2.17631003833625, -1.19565443202559, 0.365328392431846, 2.41244305717062, 0.740247220579396, 0.309746388607370, -0.541987183113314, 0.122283184138499, 0.962279332770060, -0.253974843323265, 1.89175598926363, -1.22025413722763, -0.376237143988673, 0.0524291487772232, -1.95393370060853, 0.565420571160198, 0.0777944624002781, 0.0807011876263225, 0.794707809437465, 1.04368893647872, 0.802184648027535, -0.468824364835627, -0.0465923743265948, 0.353453034782426, -1.25680056318429, -1.03603143787954, -0.426962687763533, 0.0776686794320303, 1.79093025896060, 0.797263135406905, -0.442500740824704, -0.629091091514816, 1.53379627132611, 2.73354038131324, 0.167439077485047, -0.478865165602679, -1.41265388318180, 0.974988055068974, 0.289718984218535, 1.16540825109215, -0.908705616318882, 0.240575037229452, -0.654046335191623, 1.27495453830672, -0.551190610935942, -0.0874088682817872, -0.347820318259741, 0.993962534155598, 0.428436173023810, -1.00418528016197, -0.632687876663078, -1.04570999122494, 0.505707009458752, 0.331953721396756, -1.63541252268811, -1.90684020531619, -0.0403987210424983, 0.697193329646415, 0.168802575488670, -1.54379632573123, -1.55180239414170, 0.867077803619383, -0.145372990423397, -0.386241663817012, 1.31620411236280, -0.796464226593885, 0.135441371885595, 0.417806430208596, 0.819879645963700, -0.854353379120009, 0.357318588185117, 2.74846655587521, -1.51297699352819, 0.433979032773600, -0.229767733464022, -0.827086376299091, -0.831982615393436, 0.497899500583305, 2.31562610334450, -0.793825701652537, 0.540959569281976, -0.559061357111602, 1.97656605832667, 0.544660375289434, -0.137905557777941, 0.619875635750664, -0.00558278344490564, 1.10719924542420, -0.185590108437143, -1.12141781251190, 0.246449516399111, 1.56103723018519, -1.19662171456272, -0.242345277992204, 1.00482816344250, -1.92012580512738, 0.625445779460800, 0.752960417782594, 0.213483683522705, -0.770234072868573, -0.00713465252874278, 0.0931865247774352, 0.935251207390719, 0.663517921831603, -0.350232666735205, 1.61987743935497, -0.0508330964816129, -0.812697070766603, -0.438419934696927, 0.858609994635494, 0.195215935238338, 0.888862153882707, 0.0692218210392502, 2.48682132029734, -1.66564311788150, -0.415927944852021, -0.0842246064708947, 0.0892519475074891, 1.45614720047535, 0.219453518638082, -0.114877251919321, 0.0686024694830988, 0.751494856970101, -0.689427057923177, 0.450813584524417, -1.56502465785673, -0.0787959236624287, -0.941791969061035, -0.652861505816215, 0.282529239916988, -1.12507904298537, -0.988974087143185, -1.51585876104897, -2.22851047687315, -0.151521678303304, 1.16274436560636, -0.181936286155354, -0.233953540330954, -1.04673406988337, 1.58331887778704, -0.249413421524592, 1.29401900835506, 0.361980137920309, -0.263052508703654, 1.08214445191656, 0.982434080192891, -0.111443964288855, 1.29595078548944, 1.78129502150358, -0.0656148757390404, -0.255680865369883, 0.404638157039467, 0.351756753327098, 0.807511060208285, -0.255590492307781, 0.715113278785686, 1.04859731286360, 0.177548555016550, 0.153297684827984, -1.25432447831637, -1.17280085830884, -1.46144181507780, -1.24428798710640, -0.155099375334764, -1.61491381350160, 2.24606514978483, 0.739281132309719, -2.06764536263329, -1.15455058140430, -0.107191878127395, -1.71287908652886, -0.459894542126372, 1.09100010337293, -0.163880256756727, -0.0672404702443348, -1.00159245598463, -0.555661447134641, -1.35155634252200, 0.364211322734809]])
        self.acquisition_fn.K = self.model.K
        self.acquisition_fn.cK = self.model.cK.T
        self.acquisition_fn.kbX = self.model.kernel.K(self.acquisition_fn.zb,self.model.X)
        #self.L = self._gp_innovation_local(self.zb)
        self.acquisition_fn.logP = np.reshape(self.acquisition_fn.logP, (self.acquisition_fn.logP.shape[0], 1))

        invertsign = False

        Ne = 10

        dH_fun   = self.acquisition_fn.dh_fun_false
        dH_fun_p = self.acquisition_fn.dh_fun_true

        self.acquisition_fn.predict_info_gain(
            dH_fun, dH_fun_p, self.acquisition_fn.zb, self.acquisition_fn.logP, self.X_lower, self.X_upper, Ne)




if __name__=="__main__":
    unittest.main()
