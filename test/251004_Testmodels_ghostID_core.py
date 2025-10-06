# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 23:30:27 2025

@author: Daniel Koch
"""
if __name__ == "__main__":
   
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.integrate import solve_ivp
    import networkx as nx
    import os
    import sys
    import jax
    import jax.numpy as jnp
    from jax.nn import sigmoid
    from jax import jacfwd
    from matplotlib.colors import LogNorm
    
    #paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))
    
    import _core_251004 as fun
    import functions_v08 as funOld
     
    import _utils_251004 as utils
     
    
    options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}
    inCm = 1/2.54 # convert inch to cm for plotting
    
    def wunderling_model(t, Z, para_model):
        Z = np.asarray(Z, dtype=np.float64) 
        # print(np.round(t,3), end = '\r')
        
        # Unpack parameters
        d = para_model['d']
        GMT  = para_model['GMT']
        Tcrits = para_model['Tcrits'] 
        Taus = para_model['Taus'] 
        mat_inter = para_model['mat_inter'] 

        # intrinsic = a * x * (1 - x**2)
        intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
        # Coupling effects: sum over j of C_ij * x_j
        coupling = d/10* mat_inter @ (Z + 1)
        # Total derivative
        dZdt = (intrinsic + coupling)/Taus

        return np.asarray(dZdt)  

    
    def wunderling_modeljx(t, Z, para_model):
        print(np.round(t,3), end = '\r')
        
        # Unpack parameters
        d = para_model['d']
        GMT  = para_model['GMT']
        Tcrits = para_model['Tcrits'] 
        Taus = para_model['Taus'] 
        mat_inter = para_model['mat_inter'] 
    
        # intrinsic = a * x * (1 - x**2)
        intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
        # Coupling effects: sum over j of C_ij * x_j
        coupling = d/10* mat_inter @ (Z + 1)
        # Total derivative
        dZdt = (intrinsic+coupling)/Taus
    
        return jnp.array(dZdt)
    
    
    def ghostNetwork(t, x, para):
        x_off, a, b, c, d, tau, A = para  # unpack parameters
    
        dx = (a * (x + x_off)**3 + b * (x + x_off)**2 + c * (x + x_off) + d) / tau \
         + 6 * sigmoid(100 * (jnp.matmul(A, x) - 0.95))
    
        return dx
    
        
    # Wunderling model
    
    para_model = {
        "d": 0.2,
        "GMT": 1.51,
        "mat_inter": np.array([[1,0],[1,0]]),
        "Tcrits": np.array([1.5,1.5]),
        "Taus": np.array([50,50])
        }
    
    def flow_model(t,z):
        return wunderling_model(t,z,para_model)
    
    # Define grid
    xmin=-1.5;xmax=1.5
    ymin=-1.5;ymax=1.5
    
    Ng=100
    x_range=np.linspace(xmin,xmax,Ng)
    y_range=np.linspace(ymin,ymax,Ng)
    grid_ss = np.meshgrid(x_range, y_range)
    Xg,Yg=grid_ss
    
    # Q-values on grid
    Q, coords = fun.qOnGrid(wunderling_modeljx,para_model,coords=[x_range,y_range], jit=True)
    
    x0s = [jnp.array([-0.7,-0.7]), jnp.array([1,-0.6]), jnp.array([-0.6,1])]
    
    qminima = [fun.find_local_Qminimum(wunderling_modeljx, x0, para_model,tol_glob=1e-6,method="BFGS",verbose=False)[0] for x0 in x0s]


    #######
    # FP as last pt in trajectory:
    dt = 10#para_model['dt'] 
    t_eval = np.asarray(np.arange(0, 1e5, dt), dtype=np.float64)
    tF = t_eval[-1]

    Z0 = np.array([-0.6,-1])
    # Z0 = np.array([-1.4,1])
    sol = solve_ivp(wunderling_modeljx, [0,tF], Z0,  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)

    #%%
    
    plt.figure(figsize=(16*inCm,14*inCm))
    plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
    ax = plt.gca()
    
    U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
    ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)

    
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    # Define log scale range
    vmin = np.min(Q)   # Avoid zero or negative values
    vmax = np.max(Q)
    im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                    origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(im, label='Q value (log scale)')
     

    
    
    for i in range(3):
        plt.scatter(*x0s[i], color='k')  # base point
        plt.scatter(*qminima[i], color='w')  # qminima
    
    
    J_fun = utils.make_jacfun(wunderling_modeljx, para_model)

    ev = [jnp.linalg.eig(J_fun(q)) for q in qminima]         # eigenvalues

    eigVals = [jnp.real(ev[i][0]) for i in range(len(qminima))]
    eigVecs = [ev[i][1] for i in range(len(qminima))]
    
    for i in range(len(qminima)):
        for ii in range(2):
            plt.quiver(*qminima[i], *eigVecs[i][:,ii], scale=5, angles='xy', scale_units='xy', color=f'C{2*i+ii}', label=f'λ={eigVals[i][ii]:.7f}')
            
        idxmin = np.argmin(np.abs(eigVals[i]))
        
        step = 0.1
        direction = eigVecs[i][:, idxmin]
        new_pos = qminima[i] + step * direction
        plt.scatter(*new_pos, marker='x',color='red')
        new_pos = qminima[i] - step * direction
        plt.scatter(*new_pos, marker='x',color='blue')
        
        # plt.quiver(*qminima[i], *eigVecs[i][:,ii], scale=5, angles='xy', scale_units='xy', color='w',linestyle='dotted')

    
    plt.legend(loc='upper right')
    
   
    ax.plot(sol.y[0,:],sol.y[1,:],'--b',lw=1)
    plt.show()
    
    Trj=sol.y.T
    ghostSeq = fun.ghostID(wunderling_modeljx,para_model,dt,Trj,0.05,peak_kwargs={"prominence":2,"width":2*dt},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"log","ctrl_evplot":True})
    
    #%%
    ghostSeqs = fun.ghostID_phaseSpaceSample(wunderling_modeljx, para_model, 0, 1e5, dt, [x_range,y_range],epsilon_gid=0.05,n_samples=5,peak_kwargs={"prominence":2,"width":2*dt})#,ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"log","ctrl_evplot":True})
    
    ghostList = fun.uniqueGhosts(ghostSeqs)
    
    
    #%%
    
    
    # #%%######################
    # # Ghost networks        #
    # #########################
    
    # # define network topology
    
    # rnd_network = True # set to 'False' to analyse a simple linear chain leading to a ghost channel
    # rndNet = 2 # 0 no seed, 1: simple ghost structure, 2: complex ghost structure with oscillatory transients
    
    # if rndNet == 0:
    #     np.random.seed()
    #     n = np.random.randint(6,15)
    #     p_inhibitory = 0.4
    # elif rndNet == 1:
    #     rseed=5
    #     np.random.seed(rseed)
    #     n = 10
    #     p_inhibitory = 0.4
    # elif rndNet == 2:
    #     rseed=29
    #     np.random.seed(rseed)
    #     n = 15
    #     p_inhibitory = 0.6
    
    # if rnd_network:
    #     # define network topology
    #     if rndNet==0:
    #         G = nx.erdos_renyi_graph(n, 1/np.sqrt(n), directed=True)
    #     else:
    #         G = nx.erdos_renyi_graph(n, 1/np.sqrt(n), seed=rseed, directed=True)
            
    #     A = nx.to_numpy_array(G)
    
    #     # define colors for plotting
    #     colormap = plt.get_cmap('rainbow')
    #     colors = colormap(np.linspace(0, 1, n))
    
    #     # introduce inhibitory links
    #     for i in range(A.shape[0]):
    #         for j in range(A.shape[1]):
    #             if A[i,j] == 1:
    #                 if np.random.rand() < p_inhibitory:
    #                     A[i,j] = -1
    # else:
    #     A = np.array([
    #         [0,0,0],
    #         [1,0,0],
    #         [0,1,0]
    #         ])
    
    # n = A.shape[0]
    
    # # set parameters
    # tau_ = np.ones(n) # timescales
    # paraGNet = [-2,-0.5,2,0,-4.743,tau_,A]
    
    # # timecourse simulation
    
    # dt = 0.02; t_end = 350; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
    
    # if rnd_network:
    #     ICs1 = np.random.rand(n)*8
    #     ICs2 = np.random.rand(n)*8
    #     ICs3 = np.random.rand(n)*8
    # else:
    #     ICs1 = np.array([1,0,0])*8
    #     ICs2 = np.array([0,1,0])*8
    #     ICs3 = np.array([0,0,1])*8
    
    # sol1 = solve_ivp(ghostNetwork, (0,t_end), ICs1, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 
    # sol2 = solve_ivp(ghostNetwork, (0,t_end), ICs2, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 
    # sol3 = solve_ivp(ghostNetwork, (0,t_end), ICs3, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 
    
    # #%% ghost ID
    # Trj1 = sol1.y.T
    # Trj2 = sol2.y.T
    # Trj3 = sol3.y.T
    
    # ghostSeq1 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj1,0.05,peak_kwargs={"prominence":6},ctrlOutputs={"ctrl_qplot":True,"ctrl_evplot":False})
    # ghostSeq2 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj2,0.05,peak_kwargs={"prominence":6})
    # ghostSeq3 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj3,0.05,peak_kwargs={"prominence":6})
    
    # # rename ghosts for consistency accross trajectories
       
    # allSeqs = [ghostSeq1,ghostSeq2,ghostSeq3]
    
    # allSeqs_relabeled = fun.unifyIDs(allSeqs.copy())
    
    # # plot timecourses with labels
    # ghostSeq1_,ghostSeq2_,ghostSeq3_ = allSeqs_relabeled
    # plt.figure(figsize=(9, 14.5))
    # plt.subplot(3,1,1)
    # for i in range(n):
    #     plt.plot(time,sol1.y[i,:])
    # for i in range(len(ghostSeq1_)):
    #     plt.text(ghostSeq1_[i]["time"]-10, 7, ghostSeq1_[i]["id"])
    # plt.ylim(-0.1,8)
    # plt.xlabel('time'); plt.ylabel('value');
    # plt.subplot(3,1,2)
    # for i in range(n):
    #     plt.plot(time,sol2.y[i,:])
    # for i in range(len(ghostSeq2_)):
    #     plt.text(ghostSeq2_[i]["time"]-10, 7, ghostSeq2_[i]["id"])
    # plt.ylim(-0.1,8)
    # plt.xlabel('time'); plt.ylabel('value');
    # plt.subplot(3,1,3)
    # for i in range(n):
    #     plt.plot(time,sol3.y[i,:])
    # for i in range(len(ghostSeq3_)):
    #     plt.text(ghostSeq3_[i]["time"]-10, 7, ghostSeq3_[i]["id"])
    # plt.ylim(-0.1,8)
    # plt.xlabel('time'); plt.ylabel('value');
    
    
    # # Turn sequences into adjacency matrix:
    
    # labels = []
    
    # for s in allSeqs_relabeled:
    #     for i in s:
    #         if i["id"][:1]=="G" and not i["id"] in labels:
    #             labels.append(i["id"])
    # for s in allSeqs_relabeled:
    #     for i in s:
    #         if i["id"][:1]=="O" and not i["id"] in labels:
    #             labels.append(i["id"])
    
    # ng = len(labels)
    
    # adjM_g = np.zeros((ng,ng))
    
    
    # seqIDs = [[g["id"] for g in s] for s in allSeqs_relabeled]
    
    # for s in seqIDs:
    #     for i in range(len(s)-1):
    #         e_out = labels.index(s[i])
    #         e_in = labels.index(s[i+1])
    #         if adjM_g[e_out,e_in]==0:
    #             adjM_g[e_out,e_in]=1
                
    # # plot ghost structure graph
    
    # A = adjM_g
    # N = A.shape[0]
    # nodeColors = [(0.8, 0.8, 1, 1.0)] * N # all nodes have same color
    
    # fig = plt.figure(figsize=(9, 9))
    # fun.drawNetwork(A, nodeColors, labels)
    # plt.show()
    # # %%
