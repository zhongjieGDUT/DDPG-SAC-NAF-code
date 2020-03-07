
# VREP 中的三自由度机械臂环境,两个三自由度机械臂，
# 控制模式是动力学约束下的位置闭环控制,发送位置信息
from __future__ import division
import numpy as np
import vrep
import time
import gym
class Arm(object):
    action_bound = [-1, 1]  # 动作的幅度限制,轴角速度限幅
    action_dim = 6  # 动作维度：三自由度机械臂 两个
    state_dim = 22  # 状态维度
    dt = .05 # refresh rate sample time Ts
    get_point = False  # 到达期望点
    grab_counter = 0
    observation_space = np.zeros(state_dim, dtype=float)
    action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=[action_dim])
    jointNum = 3
    jointName = 'joint'
    linkName = 'link'
    # 超参数
    delta = 0.2
    p = 8
    d_ref = 0.2
    dis = 0
    initial_jp =20.*np.pi/180.
    # 需要的数据
    def __init__(self):
        # 建立通信
        vrep.simxFinish(-1)
        # 每隔0.2s检测一次，直到连接上V-rep
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
            if self.clientID != -1:
                break
            else:
                time.sleep(0.1)
                print("Failed connecting to remote API server!")
        print("Connection success!")
        #设置机械臂仿真步长，为保持API端与VREP端相同步长
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step,self.dt, vrep.simx_opmode_oneshot)
        # 打开同步模式
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        # 获取句柄joint
        # 机械臂1
        self.robot1_jointHandle = np.zeros((self.jointNum,), dtype=np.int)  # joint 句柄
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.jointName + str(i + 1),vrep.simx_opmode_blocking)
            self.robot1_jointHandle[i] = returnHandle
        # 机械臂2
        self.robot2_jointHandle = np.zeros((self.jointNum,), dtype=np.int)  # joint 句柄
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.jointName + str(i + 4),
                                                       vrep.simx_opmode_blocking)
            self.robot2_jointHandle[i] = returnHandle
        # 获取末端frame
        _,self.end_handle  = vrep.simxGetObjectHandle(self.clientID,'end',vrep.simx_opmode_blocking)
        _, self.end0_handle = vrep.simxGetObjectHandle(self.clientID, 'end0', vrep.simx_opmode_blocking)
        _,self.goal1_handle = vrep.simxGetObjectHandle(self.clientID,'goal_1',vrep.simx_opmode_blocking)
        _,self.goal2_handle = vrep.simxGetObjectHandle(self.clientID,'goal_2',vrep.simx_opmode_blocking)
        # 俩机械臂间距
        _,self.dist_handle = vrep.simxGetDistanceHandle(self.clientID,'robots_dist',vrep.simx_opmode_blocking)
        _,self.end_dis_handle = vrep.simxGetDistanceHandle(self.clientID,'robot1_goal',vrep.simx_opmode_blocking)
        _,self.end0_dis_handle = vrep.simxGetDistanceHandle(self.clientID,'robot2_goal',vrep.simx_opmode_blocking)
        # 获取link句柄
        self.link_handle1 = np.zeros((self.jointNum,), dtype=np.int)  #link 句柄
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.linkName + str(i + 1),
                                                       vrep.simx_opmode_blocking)
            self.link_handle1[i] = returnHandle
        self.link_handle2 = np.zeros((self.jointNum,), dtype=np.int)  # link 句柄
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.linkName + str(i + 4),
                                                       vrep.simx_opmode_blocking)
            self.link_handle2[i] = returnHandle
        print('Handles available!')
    def loop(self):
        lastCmdTime = vrep.simxGetLastCmdTime(self.clientID) / 1000 # 记录当前时间
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步
        t = 0
        while t<np.pi*2:
            currCmdTime = vrep.simxGetLastCmdTime(self.clientID) / 1000
            dt = currCmdTime - lastCmdTime
            t = t + dt

            vrep.simxPauseCommunication(self.clientID, True)
            vrep.simxSetJointTargetPosition(self.clientID,self.robot1_jointHandle[0],np.sin(t),vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[1], 0,
                                            vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[2], 0,
                                            vrep.simx_opmode_oneshot)

            vrep.simxPauseCommunication(self.clientID, False)
            lastCmdTime = currCmdTime
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxGetPingTime(self.clientID)

        time.sleep(2)
        t = 0
        while t<np.pi*2:
            currCmdTime = vrep.simxGetLastCmdTime(self.clientID) / 1000
            dt = currCmdTime - lastCmdTime
            t = t + dt

            vrep.simxPauseCommunication(self.clientID, True)
            vrep.simxSetJointTargetPosition(self.clientID,self.robot1_jointHandle[0],0,vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[1], np.sin(t),
                                            vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[2], 0,
                                            vrep.simx_opmode_oneshot)

            vrep.simxPauseCommunication(self.clientID, False)
            lastCmdTime = currCmdTime
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxGetPingTime(self.clientID)
    def reset(self):
        vrep.simxSynchronousTrigger(self.clientID)  # 让仿真走一步
        vrep.simxGetPingTime(self.clientID)
        vrep.simxPauseCommunication(self.clientID, True)
        vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[0],self.initial_jp, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[1], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[2], 0, vrep.simx_opmode_oneshot)

        vrep.simxSetJointTargetPosition(self.clientID, self.robot2_jointHandle[0], self.initial_jp,vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID, self.robot2_jointHandle[1], 0, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID, self.robot2_jointHandle[2], 0, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        self.JointPosition1  = np.zeros(3,dtype=float)
        for i in range(self.jointNum):
            _,self.JointPosition1[i] = vrep.simxGetJointPosition(self.clientID,self.robot1_jointHandle[i],vrep.simx_opmode_oneshot)
        self.JointPosition2 = np.zeros(3, dtype=float)
        for i in range(self.jointNum):
            _, self.JointPosition2[i] = vrep.simxGetJointPosition(self.clientID, self.robot2_jointHandle[i],vrep.simx_opmode_oneshot)
        while abs(self.JointPosition1[0]-self.initial_jp)>1e-4 or abs(self.JointPosition1[1])>1e-4 or abs(self.JointPosition1[2])>1e-4 \
                or abs(self.JointPosition2[0]-self.initial_jp)>1e-4 or abs(self.JointPosition2[1])>1e-4 or abs(self.JointPosition2[2])>1e-4:
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxGetPingTime(self.clientID)
            for i in range(self.jointNum):
                _,self.JointPosition1[i] = vrep.simxGetJointPosition(self.clientID, self.robot1_jointHandle[i],vrep.simx_opmode_oneshot)
            for i in range(self.jointNum):
                _, self.JointPosition2[i] = vrep.simxGetJointPosition(self.clientID, self.robot2_jointHandle[i],vrep.simx_opmode_oneshot)
        s = np.zeros(6, dtype=float)
        for i in range(self.jointNum):
            _, s[i] = vrep.simxGetJointPosition(self.clientID, self.robot1_jointHandle[i], vrep.simx_opmode_oneshot)
        for i in range(self.jointNum):
            _, s[i+3] = vrep.simxGetJointPosition(self.clientID, self.robot2_jointHandle[i], vrep.simx_opmode_oneshot)
        _, pos1 = vrep.simxGetObjectPosition(self.clientID, self.end_handle, self.goal1_handle,vrep.simx_opmode_oneshot)
        _, pos2 = vrep.simxGetObjectPosition(self.clientID, self.end0_handle, self.goal2_handle, vrep.simx_opmode_oneshot)
        del pos1[1]
        del pos2[1]
        pos = np.array(pos1+pos2)
        s = np.hstack((s,pos))
        linkPos = []
        for i in range(self.jointNum):
            _, linkpos = vrep.simxGetObjectPosition(self.clientID, self.link_handle1[i], self.goal1_handle,vrep.simx_opmode_oneshot) # link1 位置
            del linkpos[1]
            linkPos+=linkpos
        for i in range(self.jointNum):
            _, linkpos = vrep.simxGetObjectPosition(self.clientID, self.link_handle2[i], self.goal2_handle,vrep.simx_opmode_oneshot) # link1 位置
            del linkpos[1]
            linkPos+=linkpos
        linkPos = np.array(linkPos)
        s = np.hstack((s,linkPos))
        self.get_point = False
        self.grab_counter = 0
        self.action = np.zeros(6,dtype=float)
        return s
    def step(self,action):# action 是速度：rad/s
        currentPosition = np.zeros(6,dtype=float)
        for i in range(self.jointNum):
            _, currentPosition[i] = vrep.simxGetJointPosition(self.clientID,self.robot1_jointHandle[i],vrep.simx_opmode_oneshot)
        for i in range(self.jointNum):
            _, currentPosition[i+3] = vrep.simxGetJointPosition(self.clientID,self.robot2_jointHandle[i],vrep.simx_opmode_oneshot)
        action = np.clip(action,*self.action_bound)
        self.JointPosition=currentPosition+action*self.dt
        self.action = action
        vrep.simxPauseCommunication(self.clientID, True)
        for i in range(self.jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.robot1_jointHandle[i], self.JointPosition[i],vrep.simx_opmode_oneshot)
        for i in range(self.jointNum):
            vrep.simxSetJointTargetPosition(self.clientID, self.robot2_jointHandle[i], self.JointPosition[i+3],vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        #获取转移状态包括 关节位置，末端,连杆距离目标的差
        s = np.zeros(6, dtype=float)
        for i in range(self.jointNum):
            _, s[i] = vrep.simxGetJointPosition(self.clientID, self.robot1_jointHandle[i], vrep.simx_opmode_oneshot)
        for i in range(self.jointNum):
            _, s[i + 3] = vrep.simxGetJointPosition(self.clientID, self.robot2_jointHandle[i], vrep.simx_opmode_oneshot)
        _, pos1 = vrep.simxGetObjectPosition(self.clientID, self.end_handle, self.goal1_handle,
                                             vrep.simx_opmode_oneshot)
        _, pos2 = vrep.simxGetObjectPosition(self.clientID, self.end0_handle, self.goal2_handle,
                                             vrep.simx_opmode_oneshot)
        del pos1[1]
        del pos2[1]
        pos = np.array(pos1 + pos2)
        s = np.hstack((s,pos))
        linkPos = []
        for i in range(self.jointNum):
            _, linkpos = vrep.simxGetObjectPosition(self.clientID, self.link_handle1[i], self.goal1_handle,
                                                    vrep.simx_opmode_oneshot)  # link1 位置
            del linkpos[1]
            linkPos += linkpos
        for i in range(self.jointNum):
            _, linkpos = vrep.simxGetObjectPosition(self.clientID, self.link_handle2[i], self.goal2_handle,
                                                    vrep.simx_opmode_oneshot)  # link1 位置
            del linkpos[1]
            linkPos += linkpos
        linkPos = np.array(linkPos)
        s = np.hstack((s, linkPos))

        _,d1 = vrep.simxReadDistance(self.clientID,self.end_dis_handle,vrep.simx_opmode_oneshot)
        _,d2 = vrep.simxReadDistance(self.clientID,self.end0_dis_handle,vrep.simx_opmode_oneshot)
        # Ra = -np.sqrt(np.sum(action**2))
        _,do = vrep.simxReadDistance(self.clientID,self.dist_handle,vrep.simx_opmode_oneshot)
        Ro = -(self.d_ref/(self.d_ref+do))**8
        if d1<self.delta:
            Rt1 = -0.5*d1*d1
        else:
            Rt1= -self.delta*(d1-0.5*self.delta)
        if d2<self.delta:
            Rt2 = -0.5*d2*d2
        else:
            Rt2= -self.delta*(d2-0.5*self.delta)
        r = (Rt1+Rt2)*1000+Ro*1200 #DDPG 取100，Naf取500
        danger = False
        if do<0.02:
            danger = True
            r -= 100
        else:
            danger = False
        if d1<0.05 and d2<0.05 and self.get_point == False:
            self.grab_counter+=1
            if self.grab_counter>80:
                r+=100
                self.get_point =True
            r+=20
        else:
            self.get_point = False
            self.grab_counter=0
        return s,r,danger,self.get_point
    def sampleAction(self):
        a = (np.random.random((self.jointNum*2,))-0.5)*2
        return a
if  __name__ == '__main__':
    env = Arm()
    s = env.reset()
    t = 0
    for i in range(3):
        s = env.reset()
        time.sleep(0.5)
        ep_r = 0
        for i in range(200):
            a = env.sampleAction()
            s_,r ,_,_= env.step(a)
            # print(s_,r)
            # time.sleep(1)
            ep_r+=r
        print(ep_r)
    env.reset()