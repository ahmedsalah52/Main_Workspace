from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import  MerweScaledSigmaPoints
import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise



class object_to_predict:
    def __init__(self,age = 10,init_pos = (0.0,0.0),ID = 0,name = ""):
        self.age = age
        self.init_pos = init_pos
        self.ID = ID
        self.dt = 1/6
        self.name = name
        self.R_std = [0.09, 0.09]
        self.sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
        self.ukf = UKF(dim_x=4, dim_z=2, fx=self.f_cv,hx=self.h_cv, dt=self.dt, points=self.sigmas)
        self.ukf.x = np.array([init_pos[0], 0., init_pos[1], 0.])
        self.ukf.R = np.diag(self.R_std) 
        self.ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)
        self.ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)
        self.ukf.predict()


    
    def get_current_state(self):
        #self.ukf.update(last_reading)
        return self.ukf.x.copy()

    def get_next_state(self,dt):
        #self.ukf._dt = dt
        self.ukf.predict()
        return self.ukf.x.copy()


    def f_cv(self,x, dt):    
        F = np.array([[1, dt, 0,  0],
                        [0,  1, 0,  0],
                        [0,  0, 1, dt],
                        [0,  0, 0,  1]])
        return F @ x

    def h_cv(self,x):
        return x[[0, 2]]

    



class Prediction():
    def __init__(self,sudden_move_thres = 50,Max_objects_number = 30,age = 5):
        self.Max_objects_number = Max_objects_number
        self.age = age
        self.sudden_move_thres = sudden_move_thres

        self.dict_of_objects = dict()
    def set_object(self,pos,ID,name):
        if ID in self.dict_of_objects.keys():
            current_pos =  (self.dict_of_objects[ID].get_current_state()[0],self.dict_of_objects[ID].get_current_state()[2])
            current_vel =  (self.dict_of_objects[ID].get_current_state()[1],self.dict_of_objects[ID].get_current_state()[3])

            
            distance = (((current_pos[0] - pos[0])**2)+((current_pos[1] - pos[1])**2))**0.5
            #velocity = (((current_vel[0])**2)+((current_vel[1])**2))**0.5
            #print(current_vel)
            #print("object : ", name ," id ",ID ," with distance :",distance," velocity :",velocity)
            
            if distance <self.sudden_move_thres:
                delta_x = current_pos[0] - pos[0]
                delta_y = current_pos[1] - pos[1]
                distance_from_car = abs(720-40-current_pos[1])
                factor_1 = 0.5
                factor_2 = 0.05

                print(distance_from_car)
                R_std = [factor_1*abs(abs(delta_x) -abs(current_vel[0]) )+(factor_2*distance_from_car)  ,factor_1*abs(abs(delta_y) -abs(current_vel[1]))+(factor_2*distance_from_car)]
                
                print("object : ", name ," id ",ID ," from :",current_pos," to :",pos , " vel ",current_vel," std ",R_std)
                if R_std[0] < 0.03:
                    R_std[0] = 0.03
                if R_std[1] < 0.03:
                    R_std[1] = 0.03
                    
                self.dict_of_objects[ID].ukf.R =np.diag(R_std) 

                self.dict_of_objects[ID].ukf.update(pos)
                self.dict_of_objects[ID].age = self.age

        else:
            new_object = object_to_predict(age = self.age,init_pos= pos,ID = ID,name = name)
            self.dict_of_objects[ID] = new_object

        if (len(self.dict_of_objects)>self.Max_objects_number):
            print("num of objects : ",len(self.dict_of_objects))
    
    
    def kill_object(self,obj):
        del self.dict_of_objects[obj.ID]


    def life_iteration(self):
        for id , obj in list(self.dict_of_objects.items()):
            if obj.age > 0:
                obj.age -= 1
            else:
                del self.dict_of_objects[id]

    def get_object_current_states(self,ID):
        return self.dict_of_objects[ID].get_current_state()


    def get_object_next_states(self,ID ):
        return self.dict_of_objects[ID].get_next_state(20)


#example
'''
my_z = [(0.5,0.5),(1,1),(1,2),(1,3),(1,4),(1,5)]

pred_manager = Prediction()
pred_manager.set_object((0,0),1,"CAR")

for z in my_z:
    pred_manager.set_object(z,1,"CAR")
    print("reading ",z)
    print("current ",pred_manager.get_object_current_states(1))
    print("next    ",pred_manager.get_object_next_states(1))
    pred_manager.life_iteration()
'''