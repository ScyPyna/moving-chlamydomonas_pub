import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class Output:
    def __init__(self,filename,column_names,data,formats):
        self.filename=filename
        self.column_names=column_names
        self.data=data
        self.formats=formats
    
    def out(self):
        np.savetxt(self.filename,np.array([]))   
        with open(self.filename, "a") as outputfile:
            outputfile.write(self.column_names)
            np.savetxt(outputfile,self.data,fmt=self.formats)
    
        

class Filters:
    def __init__(self,lifetime_min=0,threshold_dist=0,threshold_dist_tot=0,dist_step=1,strict=False,only_inside=False,blinking_ratio=0,blinking_tolerance=0.0,long_track_min_snaps=0):
        self.lifetime_min=lifetime_min              #How long do particles have to live?
        self.threshold_dist=threshold_dist          #How far do they have to go in dist_step to be considered real?
        self.threshold_dist_tot=threshold_dist_tot  #How far do theya have to go in total to be considered real?
        self.dist_step=dist_step
        self.strict=strict                          #Are the first and last frame of a trajectory always kept (False)?
        self.only_inside=only_inside                #Do we only consider particles within frame?
        self.blinking_ratio=blinking_ratio          #How short does the number of frames have to be wrt the time between first and last frame?
        self.blinking_tolerance=blinking_tolerance  #Tolerance for blinking detection
        self.long_track_min_snaps=long_track_min_snaps  #Minimum number of snaps for a long track
    
    def reset(self):
        self.lifetime_min=0
        self.threshold_dist=0
        self.threshold_dist_tot=0
        self.dist_step=1
        self.strict=False
        self.only_inside=False
        self.blinking_ratio=0
        self.blinking_tolerance=0.0
        self.long_track_min_snaps=0
        
class Speckle:
    def __init__(self,field,diam,pix,ctrp=0, lin=False, axis=0):
        self.pix = pix
        self.diam = diam
        self.field=field
        xs = np.linspace(0,len(self.field[0,:])*self.pix-self.pix,len(self.field[0,:]),endpoint=True)
        ys = np.linspace(0,len(self.field[:,0])*self.pix-self.pix,len(self.field[:,0]),endpoint=True)
        self.x_grid,self.y_grid=np.meshgrid(xs,ys)
        self.smooth=self.smoothing()
        self.gradient()
        self.Lx=len(self.field[0,:])*pix-pix
        self.Ly=len(self.field[:,0])*pix-pix
        self.ctr=ctrp
        self.lin = lin
        self.axis = axis
        if self.lin:
            self.center = np.array([self.Lx / 2, self.Ly / 2])
        else:
            self.center = np.array([self.Lx / 2 * (1+(self.ctr)*(1-self.axis)), self.Ly / 2 * (1+self.ctr*self.axis)])
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ["k", "b"])

    def smoothing(self):
        conv=int(self.diam/self.pix)-1
        mask=np.zeros((conv,conv))
        for i in range(-conv//2,conv//2):
            for j in range(-conv//2,conv//2):
                if ((i+.5)**2+(j+.5)**2)<=(conv//2)**2:
                    mask[i+conv//2,j+conv//2]=1
        mask_norm=np.sum(mask)
    
        return convolve2d(self.field, mask,boundary="symm")[conv//2:-conv//2+1,conv//2:-conv//2+1] / mask_norm
        
    def gradient(self):
        grad=np.gradient(self.smooth)
        self.grad_X=np.array(grad[1])
        self.grad_Y=np.array(grad[0])
        self.grad_tot=np.sqrt(self.grad_X**2+self.grad_Y**2)
    
    def radial_length(self,rad):
        rad=np.array(rad)
        open=np.zeros((len(rad)))
        for i, r in enumerate(rad):
            factor=-np.abs(self.ctr)+2
            if r<=min(self.Lx/factor,self.Ly/2):
                open[i]=factor*np.pi
            elif r>min(self.Lx/factor,self.Ly/2) and r<=max(self.Lx/factor,self.Ly/2):
                open[i]=factor*2*np.arctan2(self.Ly/2,np.sqrt(r**2-self.Ly**2/4))
            else:
                open[i]=factor*2*(np.arctan2(self.Ly/2,np.sqrt(r**2-self.Ly**2/4))-np.arctan2(np.sqrt(r**2-(self.Lx/factor)**2),self.Lx/factor))
        return open*rad

    def radialize(self, intp=7):
        radial_int = np.genfromtxt(f"..//inputSimulations//radialEnergyDensityMap_{intp}µW.txt", skip_header=1)
        int_converter = interp1d(np.arange(len(radial_int)) * self.pix, radial_int, kind='linear',bounds_error=False,fill_value=radial_int[-1])

        ij_dist = np.sqrt((self.x_grid-self.center[0]) ** 2 + (self.y_grid-self.center[1]) ** 2)
        ill_map = int_converter(ij_dist)

        return Speckle(ill_map,self.diam,self.pix,self.ctr)
               
class particle_class:
    def __init__(self,tag,snaps,xtraj,ytraj,speckle,dt):
        self.dt = dt
        self.tag=tag
        self.snaps=snaps
        self.xtraj=xtraj
        self.ytraj=ytraj
        self.total_snaps=(len(self.snaps))
        self.exist=True
        self.center=speckle.center
        self.lin = speckle.lin
        self.axis = speckle.axis
        self.Lx=speckle.Lx
        self.Ly=speckle.Ly
        self.center_dist()
        self.omega=None
        self.flux=np.zeros((len(snaps)))
        self.flux_snaps=np.zeros((len(snaps)))
        
        if self.total_snaps==0:
            self.first_last_snap=0
        else:
            self.first_last_snap=((self.snaps[-1]-self.snaps[0])+1)
    
    def center_dist(self):
        if self.lin:
            if self.axis==0:
                dist = np.abs(self.xtraj - self.center[0])
            else:
                dist = np.abs(self.ytraj - self.center[1])
        else:
            points = np.array([self.xtraj, self.ytraj]).transpose()
            dist = cdist(points, [self.center])[:, 0]
        self.dist=np.copy(dist)
    
    # def existance(self,filter):
    #     blinking_cond=self.first_last_snap*filter.blinking_ratio<self.total_snaps
    #     minimum_lifetime_cond=self.total_snaps*self.dt>filter.lifetime_min
    #     minimum_avg_displ_cond=np.mean(np.sqrt((self.xtraj[1:]-self.xtraj[:-1])**2+(self.ytraj[1:]-self.ytraj[:-1])**2))>filter.threshold_dist
    #     minimum_total_displ_cond=np.abs(self.xtraj[-1]-self.xtraj[0])>filter.threshold_dist_tot and np.abs(self.ytraj[-1]-self.ytraj[0])>filter.threshold_dist_tot
    #     self.exist=blinking_cond and minimum_lifetime_cond and minimum_avg_displ_cond and minimum_total_displ_cond

    def existance(self, filter):
        if self.first_last_snap <= 0 or self.total_snaps <= 1:
            self.exist = False
            return

        # --- BLINKING ---
        presence_ratio = self.total_snaps / self.first_last_snap

        tol = 0.0
        if self.total_snaps >= filter.long_track_min_snaps:
            tol = filter.blinking_tolerance

        effective_blink_threshold = max(0.0, filter.blinking_ratio - tol)

        blinking_cond = presence_ratio > effective_blink_threshold

        # --- LIFETIME ---
        minimum_lifetime_cond = self.total_snaps * self.dt > filter.lifetime_min

        # --- STEP (MEDIANA) ---
        steps = np.sqrt(
            (self.xtraj[1:] - self.xtraj[:-1])**2 +
            (self.ytraj[1:] - self.ytraj[:-1])**2
        )

        if len(steps) > 0:
            avg_step = np.median(steps)
        else:
            avg_step = 0.0

        minimum_avg_displ_cond = avg_step > filter.threshold_dist

        # --- TOTAL DISPLACEMENT (EUCLIDEO) ---
        dx_tot = self.xtraj[-1] - self.xtraj[0]
        dy_tot = self.ytraj[-1] - self.ytraj[0]

        disp_tot = np.sqrt(dx_tot**2 + dy_tot**2)

        minimum_total_displ_cond = disp_tot > filter.threshold_dist_tot

        # --- RISULTATO ---
        self.exist = (
            blinking_cond
            and minimum_lifetime_cond
            and minimum_avg_displ_cond
            and minimum_total_displ_cond
        )
    
    def traj_cleaner(self,filter):
        dy=self.ytraj[filter.dist_step::filter.dist_step]-self.ytraj[:-filter.dist_step:filter.dist_step]
        dx=self.xtraj[filter.dist_step::filter.dist_step]-self.xtraj[:-filter.dist_step:filter.dist_step]
        Dist_p_cond=np.sqrt(dy**2+dx**2)>filter.threshold_dist*filter.dist_step
        if filter.strict:
            Dist_p_cond_tot=np.append(Dist_p_cond,[True]) & np.append([True],Dist_p_cond)
        else:
            Dist_p_cond_tot=np.append(Dist_p_cond,[False]) | np.append([False],Dist_p_cond)
        
        inside_cond=((self.ytraj>=0) & (self.ytraj<=self.Ly) & (self.xtraj>=0) & (self.xtraj<=self.Lx))
        
        if not filter.only_inside:
            inside_cond=np.full((len(inside_cond)),True)
        
        if not np.any(Dist_p_cond_tot & inside_cond):
            print("particle {0:d} has no accepted snapshots".format(self.tag))
            self.exist=False 

        if (~inside_cond).any():
            print("particle {0:d} exits the frame".format(self.tag))
        
        self.snaps=self.snaps[Dist_p_cond_tot & inside_cond]
        self.xtraj=self.xtraj[Dist_p_cond_tot & inside_cond]
        self.ytraj=self.ytraj[Dist_p_cond_tot & inside_cond]
        self.center_dist()
           
    def grad_theta(self,speckle,particle_show):
    
        Ly_sp=speckle.Ly#len(speckle.field[:,0])*pix-pix
        Lx_sp=speckle.Lx#len(speckle.field[0,:])*pix-pix
        inside_cond=((self.ytraj>=0) & (self.ytraj<=Ly_sp) & (self.xtraj>=0) & (self.xtraj<=Lx_sp))
        
        if (~inside_cond).any():
            print("particle {0:d} exits the frame".format(self.tag))
            
        Xin=self.xtraj[inside_cond]
        Yin=self.ytraj[inside_cond]
        Sin=self.snaps[inside_cond]
        #
        #if not np.any(Dist_p_cond_tot & inside_cond):
        #    print("particle {0:d} has no accepted snapshots".format(self.tag))
        #    self.exist=False
    
        Vx=(Xin[particle_show:]-Xin[:-particle_show])/((Sin[particle_show:]-Sin[:-particle_show])*self.dt)
        Vy=(Yin[particle_show:]-Yin[:-particle_show])/((Sin[particle_show:]-Sin[:-particle_show])*self.dt)
        V=np.sqrt(Vx**2+Vy**2)
        I=np.zeros(len(Xin))
        
        #make sure not to be in the no gradient case
        #print(self.ytraj)
        #print(self.xtraj)
        #print(len(field[:,0])*pix-pix)
        #print(len(field[0,:])*pix-pix)
        #print(self.tag)
        
        pix=speckle.pix
        
        I=speckle.smooth[(Yin/pix).astype(int),(Xin/pix).astype(int)]
        I_gradx=speckle.grad_X[(Yin/pix).astype(int),(Xin/pix).astype(int)]
        I_grady=speckle.grad_Y[(Yin/pix).astype(int),(Xin/pix).astype(int)]
        I_grad=speckle.grad_tot[(Yin/pix).astype(int),(Xin/pix).astype(int)]
        
        #to visualize also uncomment fig2 outside of cycle
        #ax2.scatter(self.xtraj[::particle_show],self.ytraj[::particle_show],color=[cmap.to_rgba(I[j*particle_show]) for j in range(len(self.ytraj[::particle_show]))])
        
        I_avg=moving_average(I,particle_show+1)
        Igx_avg=moving_average(I_gradx,particle_show+1)
        Igy_avg=moving_average(I_grady,particle_show+1)
        Ig_avg=np.sqrt(Igx_avg**2+Igy_avg**2)#moving_average(I_grad,particle_show+1)
        dist_avg=moving_average(self.dist[inside_cond],particle_show+1)
        
        try:  
            cond_g=Ig_avg*V!=0#
            Theta=np.arccos(((Vx[cond_g]*Igx_avg[cond_g])+(Vy[cond_g]*Igy_avg[cond_g]))/(V[cond_g]*Ig_avg[cond_g]))#(((Vx[cond_g]*Igx_avg[cond_g])+(Vy[cond_g]*Igy_avg[cond_g]))/(V[cond_g]*Ig_avg[cond_g]))#
            I_avg=I_avg[cond_g]
            Ig_avg=Ig_avg[cond_g]
            dist_avg=dist_avg[cond_g]
        
        except:
            print("Very short trajectory for particle "+str(self.tag))
            Theta=np.full((len(I_avg)),np.nan)
        
        if (len(Theta)!=len(dist_avg)):
            print("No gradient for particle {0:d}".format(self.tag))

        '''
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        ax2.pcolormesh(speckle.x_grid, speckle.y_grid, speckle.field)
        xmp=moving_average(self.xtraj,particle_show+1)[cond_g][::particle_show]
        ymp=moving_average(self.ytraj,particle_show+1)[cond_g][::particle_show]
        ax2.quiver(xmp, ymp, Igx_avg[cond_g][::particle_show]/Ig_avg[::particle_show], Igy_avg[cond_g][::particle_show]/Ig_avg[::particle_show], cmap="plasma",color="g")
        ax2.quiver(xmp, ymp, Vx[cond_g][::particle_show] / V[cond_g][::particle_show],
                   Vy[cond_g][::particle_show] / V[cond_g][::particle_show], cmap="plasma", color="white")
        ax2.scatter(xmp, ymp, c=Theta[::particle_show], cmap="plasma",vmin=0, vmax=np.pi)
        plt.show()
        '''

        return I_avg[::particle_show], Ig_avg[::particle_show], Theta[::particle_show], dist_avg[::particle_show]

    def angular_velocity(self):
        
        
        Xin=np.copy(self.xtraj)
        Yin=np.copy(self.ytraj)
        Sin=np.copy(self.snaps)
        
        if len(Xin)<=2:
            self.omega=np.array([])
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        #
        #if not np.any(Dist_p_cond_tot & inside_cond):
        #    print("particle {0:d} has no accepted snapshots".format(self.tag))
        #    self.exist=False
       
        dx=Xin[1:]-Xin[:-1]
        dy=Yin[1:]-Yin[:-1]
        vx=(Xin[1:]-Xin[:-1])/((Sin[1:]-Sin[:-1])*self.dt)
        vy=(Yin[1:]-Yin[:-1])/((Sin[1:]-Sin[:-1])*self.dt)
        vabs=np.sqrt(vx**2+vy**2)
        angle=np.arctan2(dy,dx)
        da=angle[1:]-angle[:-1]
        da[da>np.pi]=da[da>np.pi]-2*np.pi
        da[da<-np.pi]=2*np.pi+da[da<-np.pi]
        omega=da/((Sin[2:]-Sin[:-2])*self.dt)
        self.omega=np.concatenate([[np.nan],omega,[np.nan]])
        
        
        #print("straight:",np.mean(vabs[self.omega[:-1]<=tumbling_threshold]))
        #print("tumbled:",np.mean(vabs[self.omega[:-1]>tumbling_threshold]))
        
        #figw, axw = plt.subplots(nrows=1,ncols=3,figsize=(16,9))
        #axw[2].hist((dx**2+dy**2))#/moving_average(Sin,2)
        #axw[2].set_title(r"$dr$",fontsize=20)
        #axw[1].hist(omega)
        #axw[1].set_title(r"$\omega$",fontsize=20)
        #axw[0].set_aspect(1)
        #axw[0].plot(Xin,Yin)
        #axw[0].quiver((Xin[1:]+Xin[:-1])/2,(Yin[1:]+Yin[:-1])/2,np.cos(angle),np.sin(angle))
        #axw[0].scatter(Xin[1:-1],Yin[1:-1],c=omega,cmap="bwr",vmin=-np.max(np.abs(omega)),vmax=np.max(np.abs(omega)))
        #axw[0].scatter(Xin[1:-1][np.abs(omega)>tumbling_threshold],Yin[1:-1][np.abs(omega)>tumbling_threshold],c="k",marker="x",s=100)
        #plt.show()

        return angle, moving_average(self.dist,2), vabs, self.omega, self.dist, Sin*self.dt
        
    def tumbling_duration(self,speckle,tumbling_threshold):
        if self.omega is None:
            self.angular_velocity()
        
        if len(self.xtraj)<=2:
            return np.array([]), np.array([]), np.array([])
            
        inside_cond=((self.ytraj>=0) & (self.ytraj<=speckle.Ly) & (self.xtraj>=0) & (self.xtraj<=speckle.Lx))
        
        if (~inside_cond).any():
            print("particle {0:d} exits the frame".format(self.tag))
        
        
        
        Xin=self.xtraj[inside_cond]
        Yin=self.ytraj[inside_cond]
        Sin=self.snaps[inside_cond] 
        absom=np.abs(self.omega[inside_cond])
        Din=self.dist[inside_cond]    
        
        #for "tumble" parts
        untumble_start=(absom[1:]<=tumbling_threshold) & (absom[:-1]>tumbling_threshold)
        untumble_start=np.concatenate([[untumble_start[0]],untumble_start])
        untumbles=1+(np.cumsum(untumble_start))
        untumbles[absom<=tumbling_threshold]=0
        if absom[1]<=tumbling_threshold:
            untumbles[0]=0
        if absom[-2]<=tumbling_threshold:
            untumbles[-1]=0
        
        vec_tduration=np.full((np.max(untumbles)),np.nan)
        vec_ttime=np.full((np.max(untumbles)),np.nan)
        vec_tdist=np.full((np.max(untumbles)),np.nan)
        
        for i in range(np.max(untumbles)):
            
            untumbli=untumbles==i+1
            
            if np.sum(untumbli)==0:
                continue
            
            vec_tduration[i]=np.sum(untumbli)
            vec_ttime[i]=np.mean(Sin[untumbli])*self.dt
            vec_tdist[i]=np.mean(Din[untumbli])
        
        #figw, axw = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
        #axw.plot(Xin,Yin)  
        #axw.quiver((Xin[1:]+Xin[:-1])/2,(Yin[1:]+Yin[:-1])/2,np.cos(angle),np.sin(angle))
        #axw.scatter(Xin,Yin,c=untumbles%10,cmap="hsv",vmin=0,vmax=10)
        #axw.scatter(Xin[absom>tumbling_threshold],Yin[absom>tumbling_threshold],c="k",marker="x",s=100)
        #axw.set_xlim(np.min(Xin),np.max(Xin))
        #axw.set_ylim(np.min(Yin),np.max(Yin))
        #axw.set_aspect(1)
        #plt.show()
        
        return vec_tdist[vec_tduration>1], vec_tduration[vec_tduration>1]*self.dt, vec_ttime[vec_tduration>1]
        
    
    def tumbling_orientation(self, speckle, tumbling_threshold):
        if self.omega is None:
            self.angular_velocity()
        
        if len(self.xtraj)<=2:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        inside_cond=((self.ytraj>=0) & (self.ytraj<=speckle.Ly) & (self.xtraj>=0) & (self.xtraj<=speckle.Lx))
        
        if (~inside_cond).any():
            print("particle {0:d} exits the frame".format(self.tag))
        
        
        
        Xin=self.xtraj[inside_cond]
        Yin=self.ytraj[inside_cond]
        Sin=self.snaps[inside_cond] 
        absom=np.abs(self.omega[inside_cond])
        Din=self.dist[inside_cond]
        #for "run" parts
        tumble_start=(absom[1:]>tumbling_threshold) & (absom[:-1]<=tumbling_threshold)
        tumble_start=np.concatenate([[tumble_start[0]],tumble_start])
        #tumble_end=(absom[1:]<=tumbling_threshold) & (absom[:-1]>tumbling_threshold)
        tumbles=1+(np.cumsum(tumble_start))
        tumbles[absom>tumbling_threshold]=0
        if absom[1]>tumbling_threshold:
            tumbles[0]=0
        if absom[-2]>tumbling_threshold:
            tumbles[-1]=0
        
            
        vec_start_x=np.full((np.max(tumbles)),np.nan)
        vec_start_y=np.full((np.max(tumbles)),np.nan)
        vec_end_x=np.full((np.max(tumbles)),np.nan)
        vec_end_y=np.full((np.max(tumbles)),np.nan)
        vec_dx=np.full((np.max(tumbles)),np.nan)
        vec_dy=np.full((np.max(tumbles)),np.nan)
        vec_I_dx=np.full((np.max(tumbles)),np.nan)
        vec_I_dy=np.full((np.max(tumbles)),np.nan)
        vec_duration=np.full((np.max(tumbles)),np.nan)
        vec_time=np.full((np.max(tumbles)),np.nan)
        vec_dist=np.full((np.max(tumbles)),np.nan)

        pix=speckle.pix

        
        for i in range(np.max(tumbles)):
            
            tumbli=tumbles==i+1
            
            if np.sum(tumbli)==0:
                continue

            vec_start_x[i]=Xin[tumbli][0]
            vec_start_y[i]=Yin[tumbli][0]
            vec_end_x[i]=Xin[tumbli][-1]
            vec_end_y[i]=Yin[tumbli][-1]
            
            vec_I_dx[i]=np.mean(speckle.grad_X[(Yin[tumbli]/pix).astype(int),(Xin[tumbli]/pix).astype(int)])
            vec_I_dy[i]=np.mean(speckle.grad_Y[(Yin[tumbli]/pix).astype(int),(Xin[tumbli]/pix).astype(int)])
            
            vec_duration[i]=np.sum(tumbli)*self.dt
            vec_time[i]=np.mean(Sin[tumbli])*self.dt
            vec_dist[i]=np.mean(Din[tumbli])
        
        vec_dx=vec_end_x-vec_start_x
        vec_dy=vec_end_y-vec_start_y
        vec_theta=np.arccos(((vec_dx*vec_I_dx)+(vec_dy*vec_I_dy))/np.sqrt((vec_dx**2+vec_dy**2)*(vec_I_dx**2+vec_I_dy**2)))
        
       
        #figw, axw = plt.subplots(nrows=1,ncols=2,figsize=(8,8))
        #axw[0].plot(Xin,Yin)  
        #axw[0].quiver((Xin[1:]+Xin[:-1])/2,(Yin[1:]+Yin[:-1])/2,np.cos(angle),np.sin(angle))
        ##axw.scatter(Xin,Yin,c=self.omega,cmap="bwr",vmin=-np.nanmax(absom),vmax=np.nanmax(absom))
        #axw[0].scatter(Xin,Yin,c=tumbles,cmap="hsv",vmin=np.min(tumbles),vmax=np.max(tumbles))
        #axw[0].scatter(Xin[absom>tumbling_threshold],Yin[absom>tumbling_threshold],c="k",marker="x",s=100)
        #axw[0].scatter(vec_start_x,vec_start_y,c="b",marker="x")
        #axw[0].scatter(vec_end_x,vec_end_y,c="r",marker="x")
        #axw[0].set_xlim(np.min(Xin),np.max(Xin))
        #axw[0].set_ylim(np.min(Yin),np.max(Yin))
        #
        #axw[1].pcolormesh(x_gridS,y_gridS,speckle.field,cmap="afmhot",vmin=np.min(speckle.field),vmax=np.max(speckle.field))
        #axw[1].scatter(vec_start_x,vec_start_y,c="b",marker="x")
        #axw[1].scatter(vec_end_x,vec_end_y,c="r",marker="x")        
        ##axw[1].quiver(vec_start_x,vec_start_y, vec_dx, vec_dy, vec_theta, cmap="bwr", scale=1) 
        #for i in range(len(vec_start_x)):
        #    axw[1].arrow(vec_start_x[i], vec_start_y[i], vec_dx[i], vec_dy[i], ec=plt.cm.bwr((vec_theta[i]-np.nanmin(vec_theta))/(np.nanmax(vec_theta)-np.nanmin(vec_theta))), head_width=5, head_length=5)
        #
        ##axw.scatter(Xin,Yin,c=self.omega,cmap="bwr",vmin=-np.nanmax(absom),vmax=np.nanmax(absom))
        #axw[1].set_xlim(np.min(Xin),np.max(Xin))
        #axw[1].set_ylim(np.min(Yin),np.max(Yin))
        #plt.show()
        
        return vec_theta[~np.isnan(vec_theta)], vec_dist[~np.isnan(vec_theta)], vec_duration[~np.isnan(vec_theta)], vec_time[~np.isnan(vec_theta)]
            
        

    def flux_calculator(self,disc_radius,cg):
        exit=(self.dist[cg::cg]>disc_radius) & (self.dist[:-cg:cg]<=disc_radius)
        enter=(self.dist[cg::cg]<=disc_radius) & (self.dist[:-cg:cg]>disc_radius)
        self.flux=enter.astype(int)-exit.astype(int)
        self.flux_snaps=(self.snaps[cg::cg]+self.snaps[:-cg:cg])/2
