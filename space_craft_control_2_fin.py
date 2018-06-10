# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:26:32 2017

@author: 篤樹
"""

import numpy as np
import matplotlib.pyplot as plt
from space_craft_control_1_extend import *

Ixx = 1.9
Iyy = 1.6
Izz = 2.0

'''
ここは操作するところ
'''
#ステップ数
n = 2000
dt = 0.01
#観測データが入る時間
obs_time = 0.3
#ノイズの標準偏差(外乱も観測も同じと仮定)
dispersion = 0.01
#pの初期値
p_start = 0.7
'''
ここまで
'''

#観測データが入るステップ数
obs_step = int(obs_time/dt)

Mx = np.random.normal(0.0,dispersion,n+1)
My = np.random.normal(0.0,dispersion,n+1)
Mz = np.random.normal(0.0,dispersion,n+1)

#共分散行列は一定
Q = dispersion**2*np.identity(7)
R = dispersion**2*np.identity(3)

omega_s = [0.1,1.88,0.0]
q_s     = [1.0,0.0,0.0,0.0]

def main():
    #真値の設定
    q_true_list,omega_true_list,time = simulator(Mx,My,Mz,omega_s,q_s,n)

    omega_true_x = omega_true_list[0]
    omega_true_y = omega_true_list[1]
    omega_true_z = omega_true_list[2]

    q0_true = q_true_list[0]
    q1_true = q_true_list[1]
    q2_true = q_true_list[2]
    q3_true = q_true_list[3]
    
    '''
    for i in range(n):
        print q0_true[i]**2+q1_true[i]**2+q2_true[i]**2+q3_true[i]**2
    '''
    
    #初期値の設定
    est_start = 0.3*np.random.random(7)
    #est_start = [0.0,1.0,0.0,0.0,0.1,1.88,0.0]    
    #est_start = [1.0,0.0,0.0,0.0,0.1,1.88,0.0]+[-0.0121710,0.0898027,0.0898027,0.0898027,0.1,0.1,0.1]
    est = [est_start]
    
    P = p_start*np.identity(7)
    p_list_start = [P[0][0],P[1][1],P[2][2],P[3][3],P[4][4],P[5][5],P[6][6]]
    
    p_list = [p_list_start]
    
    #推定系の計算
    for i in range(n):
        #print est[i][0]**2+est[i][1]**2+est[i][2]**2+est[i][3]**2
        # i=>now
        # i+1=>next
        q_tent = [est[i][0],est[i][1],est[i][2],est[i][3]]
        omega_tent = [est[i][4],est[i][5],est[i][6]]        
        
        Mx_observe = [0.0]
        My_observe = [0.0]
        Mz_observe = [0.0]
        
        #[[next]]でリストを返す        
        q_observe_list,omega_observe_list,time2 = simulator(Mx_observe,My_observe,Mz_observe,omega_tent,q_tent,0)
        
        
        #リストの整理
        omega_observe_x = omega_observe_list[0][0]
        omega_observe_y = omega_observe_list[1][0]
        omega_observe_z = omega_observe_list[2][0]

        q0_observe = q_observe_list[0][0]
        q1_observe = q_observe_list[1][0]
        q2_observe = q_observe_list[2][0]
        q3_observe = q_observe_list[3][0]
        
        #nextをestに加える
        est_tend = [q0_observe,q1_observe,q2_observe,q3_observe,\
                    omega_observe_x,omega_observe_y,omega_observe_z]
        est.append(est_tend)
        
        #nextの値から M,Phai,Gammaを算出
        M,Phai,Gamma = estimate_calc(est_tend,P)
        
        if (i+1)%int(obs_step)!= 0:
            P = M
            
            #Pの格納
            p_list_tent = [P[0][0],P[1][1],P[2][2],P[3][3],P[4][4],P[5][5],P[6][6]]
            p_list.append(p_list_tent)
        
        else:    
            #観測値の取り入れ
            #得られるdcmのベクトルの選択      
            est_next = est[i+1]
            dcm_num = np.random.randint(1,4)
            dcm_observe  = DCM(q0_true[i+1],q1_true[i+1],q2_true[i+1],q3_true[i+1],dcm_num,1)
            dcm_estimate = DCM(est_next[0],est_next[1],est_next[2],est_next[3],\
                                  dcm_num,0)
            z = dcm_observe - dcm_estimate
        
            H = obs_matrix_calc(est_next,dcm_num)            
            
            #Pの更新
            P,K = Gain_calc(H,M)    
            
            #Pの格納
            p_list_tent = [P[0][0],P[1][1],P[2][2],P[3][3],P[4][4],P[5][5],P[6][6]]
            p_list.append(p_list_tent)
            
            #xを求める
            x = np.dot(K,z)
                
            #estの更新
            est[i+1] = est[i+1] + x
            norm = (est[i+1][0]**2+est[i+1][1]**2+est[i+1][2]**2+est[i+1][3]**2)**0.5
            
            for j in range(4):
                est[i+1][j] = est[i+1][j]/norm
            
    q0_est = []
    q1_est = []
    q2_est = []
    q3_est = []
    omega_est_x = []
    omega_est_y = []
    omega_est_z = []   
    
    p0_list = []
    p1_list = []
    p2_list = []
    p3_list = []
    p4_list = []
    p5_list = []
    p6_list = []
    
    for each_est in est:
        q0_est.append(each_est[0])
        q1_est.append(each_est[1])
        q2_est.append(each_est[2])
        q3_est.append(each_est[3])
        omega_est_x.append(each_est[4])
        omega_est_y.append(each_est[5])
        omega_est_z.append(each_est[6])
        
    for each_p in p_list:
        p0_list.append(each_p[0])
        p1_list.append(each_p[1])
        p2_list.append(each_p[2])
        p3_list.append(each_p[3])
        p4_list.append(each_p[4])
        p5_list.append(each_p[5])
        p6_list.append(each_p[6])
        
    q0_est = np.array(q0_est)
    q1_est = np.array(q1_est)
    q2_est = np.array(q2_est)
    q3_est = np.array(q3_est)
    omega_est_x = np.array(omega_est_x)
    omega_est_y = np.array(omega_est_y)
    omega_est_z = np.array(omega_est_z)
    
    p0_sqrt_list = [a**0.5 for a in p0_list]
    p1_sqrt_list = [a**0.5 for a in p1_list]
    p2_sqrt_list = [a**0.5 for a in p2_list]
    p3_sqrt_list = [a**0.5 for a in p3_list]
    p4_sqrt_list = [a**0.5 for a in p4_list]
    p5_sqrt_list = [a**0.5 for a in p5_list]
    p6_sqrt_list = [a**0.5 for a in p6_list]
    
    
    delta_q0 = q0_est - q0_true
    delta_q1 = q1_est - q1_true
    delta_q2 = q2_est - q2_true
    delta_q3 = q3_est - q3_true
    delta_omega_x = omega_est_x - omega_true_x
    delta_omega_y = omega_est_y - omega_true_y
    delta_omega_z = omega_est_z - omega_true_z
    
    plt.plot(time,p0_sqrt_list)
    plt.plot(time,delta_q0)
    plt.savefig("sqrt_p_q0")
    plt.show()
    
    plt.plot(time,p1_sqrt_list)
    plt.plot(time,delta_q1)
    plt.savefig("sqrt_p_q1")
    plt.show()
    
    plt.plot(time,p2_sqrt_list)
    plt.plot(time,delta_q2)
    plt.savefig("sqrt_p_q2")
    plt.show()
    
    plt.plot(time,p3_sqrt_list)
    plt.plot(time,delta_q3)
    plt.savefig("sqrt_p_q3")
    plt.show()
    
    plt.plot(time,p4_sqrt_list)
    plt.plot(time,delta_omega_x)
    plt.savefig("sqrt_p_omegax")
    plt.show()
    
    plt.plot(time,p5_sqrt_list)
    plt.plot(time,delta_omega_y)
    plt.savefig("sqrt_p_omegay")
    plt.show()
    
    plt.plot(time,p6_sqrt_list)
    plt.plot(time,delta_omega_z)
    plt.savefig("sqrt_p_omegaz")
    plt.show()
    
    '''    
    plt.plot(time,q0_est)
    plt.plot(time,q0_true)
    plt.savefig("q0_time")
    plt.show()
    
    plt.plot(time,q1_est)
    plt.plot(time,q1_true)
    plt.savefig("q1_time")
    plt.show()
    
    plt.plot(time,omega_est_x)
    plt.plot(time,omega_true_x)
    plt.savefig("omegax_time")
    plt.show()
    '''
    return 0
            

def matrix_calc(q0,q1,q2,q3,omega_x,omega_y,omega_z):
    #連続系の係数
    A = np.array([[0, -0.5*omega_x, -0.5*omega_y, -0.5*omega_z,-0.5*q1, -0.5*q2, -0.5*q3],\
                  [0.5*omega_x,  0,  0.5*omega_z, -0.5*omega_y, 0.5*q0, -0.5*q3,  0.5*q2],\
                  [0.5*omega_y, -0.5*omega_z,  0,  0.5*omega_x, 0.5*q3,  0.5*q0, -0.5*q1],\
                  [0.5*omega_z,  0.5*omega_y, -0.5*omega_x,  0,-0.5*q2,  0.5*q1,  0.5*q0],
                  [0, 0, 0, 0, 0, (Iyy-Izz)/Ixx*omega_z, (Iyy-Izz)/Ixx*omega_y],
                  [0, 0, 0, 0, (Izz-Ixx)/Iyy*omega_z, 0, (Izz-Ixx)/Iyy*omega_x],
                  [0, 0, 0, 0, (Ixx-Iyy)/Izz*omega_y, (Ixx-Iyy)/Izz*omega_x, 0]])
    
    B = np.array([[0]*7]*7)
    B[4][4] = 1.0/Ixx
    B[5][5] = 1.0/Iyy
    B[6][6] = 1.0/Izz
    
    #離散系の係数を求める
    Phai = np.identity(7) + dt*A
    Gamma = dt*B
    return Phai,Gamma
    
def estimate_calc(est_now,P):
    Phai,Gamma = matrix_calc(est_now[0],est_now[1],est_now[2],est_now[3],est_now[4],est_now[5],est_now[6])
    ''' 
    noise = np.random.normal(0.0,0.01,7)
    est_next = np.dot(Phai,est_now) + np.dot(Gamma,noise)
    '''
    P_Phai_T = np.dot(P,Phai.T)
    Q_Gamma_T = np.dot(Q,Gamma.T)
    M = np.dot(Phai,P_Phai_T) + np.dot(Gamma,Q_Gamma_T)
    
    return M,Phai,Gamma
    
def DCM(q0,q1,q2,q3,dcm_num,noise):
    #dcm_numでどのDCMが選択されたのか判別する
    #noiseで観測ノイズを入れるか判別
    #noiseは1 or 0しか入れてはいけない

    dcm1 = np.array([q0**2.0+q1**2.0-q2**2.0-q3**2.0, 2.0*(q1*q2+q0*q3), 2.0*(q1*q3-q0*q2)]) + noise*np.random.normal(0.0,0.01)
    dcm2 = np.array([2.0*(q1*q2-q0*q3), q0**2.0-q1**2.0+q2**2.0-q3**2.0, 2.0*(q2*q3+q0*q1)]) + noise*np.random.normal(0.0,0.01)
    dcm3 = np.array([2.0*(q1*q3+q0*q2), 2.0*(q2*q3-q0*q1), q0**2.0-q1**2.0-q2**2.0+q3**2.0]) + noise*np.random.normal(0.0,0.01)
    
    if dcm_num==1:
        return dcm1
    elif dcm_num==2:
        return dcm2
    elif dcm_num==3:
        return dcm3
    else:
        print "選択するDCMの数がわかりません"
        return 0

def Gain_calc(H,M):
    MH_T = np.dot(M,H.T)
    HMH_R = np.dot(H,MH_T) + R
    inv_HMH_R = np.linalg.inv(HMH_R)
    HM = np.dot(H,M)    
    MH_inv = np.dot(MH_T,inv_HMH_R)
    
    P_temp = M - np.dot(MH_inv,HM)
    
    PH_T = np.dot(P_temp,H.T)
    inv_R = np.linalg.inv(R)

    K = np.dot(PH_T,inv_R)    

    return P_temp,K    
    
def obs_matrix_calc(est_list,dcm_num):
    q0 = est_list[0]
    q1 = est_list[1]
    q2 = est_list[2]
    q3 = est_list[3]
    
    if dcm_num == 1:
        H = np.array([[2.0*q0, 2.0*q1,-2.0*q2,-2.0*q3,0,0,0],\
                      [2.0*q3, 2.0*q2, 2.0*q1, 2.0*q0,0,0,0],\
                      [-2.0*q2, 2.0*q3,-2.0*q0, 2.0*q1,0,0,0]])
        return H
    elif dcm_num == 2:
        H = np.array([[-2.0*q3, 2.0*q2, 2.0*q1,-2.0*q0,0,0,0],\
                      [2.0*q0,-2.0*q1, 2.0*q2,-2.0*q3,0,0,0],\
                      [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2,0,0,0]])
        return H
    elif dcm_num == 3:
        H = np.array([[ 2.0*q2, 2.0*q3, 2.0*q0, 2.0*q1,0,0,0],\
                      [-2.0*q1,-2.0*q0, 2.0*q3, 2.0*q2,0,0,0],\
                      [ 2.0*q0,-2.0*q1,-2.0*q2, 2.0*q3,0,0,0]])
        return H
    else:
        return -1

if __name__ == '__main__':
    main()