

import numpy as np


import matplotlib.pyplot as plt








class FeatureExtractor2:
    def __init__(self, ) -> None:
       
        aux = 255
        self._wallColor = (0, aux*0.33, 0)
        self._po0color = (0, 0, aux*0.25)
        self._po1color =  (aux*0.25, 0, 0)
        self._pobothcolor = (aux*0.25, 0, aux*0.25)
        self._gray = (127,127,127)
        self._green = (0,255,0)
        self._blue = (0,0,255)
        self._red =  (255,0,0)
        self._white = (255,255,255)
        self._lightGray = (200,200,200)
        self._orange =(255,100,10)
        self._yellow = (255,255,0)
        self._cyan = (0, 255, 255)

     

    '''X
    def getFeature(self,gs :GameState,player :int):
        
        pgs = gs.getPhysicalGameState()
        feature = np.zeros((3,pgs.getHeight(),pgs.getWidth()))
        for u in pgs.getUnits(player).values():
            if u.getPlayer() != player:
                continue
         
            
            
            
            ut = u.getType()
          
        
            if ut.getName() == "Base":
                color = self._white
                
            elif ut.getName() == "Barracks":
                color = self._lightGray
            
            elif ut.getName() == "Worker":
                color = self._gray
               
            
            elif ut.getName() == "Light":
                color = self._orange
         
            
            elif ut.getName() == "Heavy":
                color = self._yellow
            elif ut.getName() =="Ranged":
                color = self._cyan
        
            
            feature[0][u.getY()][u.getX()]=color[0]/255
            feature[1][u.getY()][u.getX()]=color[1]/255
            feature[2][u.getY()][u.getX()]=color[2]/255
            
  
        return feature
'''
    def viewFeature(self,feature,scale=10):
        shape =feature.shape
        vi =np.transpose(feature, (1, 2, 0))
        plt.imshow(vi)
        plt.show()
        
            


