#-------------------> PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import csv
from scipy.ndimage import rotate
import random

#-------------------> CONSTANTS

VNa = 115
VK = -12
VL = 10.613
Temp = 6.3 # C
Cm = 1
gK = 36
gNa = 120
gL = 0.3
phi = 1#3**((Temp-6.3)/10)
dt = 0.0075#0.0075
I_value =2.95#2.95#2.95 #15.2 for 120bpm


print(1.7*I_value**(-0.45))


#-------------------> MODEL FUNCTIONS

#myocardium potassium conductance as a function of heart rate
gKcorr = (1/-0.017)*(0.42*((1.7*I_value**(-0.45))**(1/3))-0.72)

#slow membrane potential function using the corrected potassium conductance, used in myocardium layers
def V_func_slow(V,n,m,c,I):
    global dt
    ma = (0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1))/((0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1)) + (4*phi*np.exp(-V/18)))
    out = (dt/Cm)*(I-gKcorr*n**4*(V-(VK))-120*ma**3*(c-n)*(V-VNa))+V
    return(out)

#regular membrane potential function used everywhere else
def V_func_fast(V,n,m,c,I):
    global dt
    ma = (0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1))/((0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1)) + (4*phi*np.exp(-V/18)))
    out = (dt/Cm)*(I-36*n**4*(V-VK)-120*ma**3*(c-n)*(V-VNa))+V
    return(out)


def V_func_prk(V,n,m,c,I):
    global dt
    ma = (0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1))/((0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1)) + (4*phi*np.exp(-V/18)))
    out = (dt/Cm)*(I-36*n**4*(V-VK)-120*ma**3*(c-n)*(V-VNa))+V
    return(out)
#HH2D n function
def n_func(V,n,m):
    global dt
    alpha_n = 0.01*phi*(-V+10)/(np.exp((-V+10)/10)-1)
    beta_n = 0.125*phi*np.exp(-V/80)
    out = dt*(alpha_n*(1-n)-beta_n*n)+n
    return(out)

def n_func_ventricle(V,n,m):
    global dt
    alpha_n = 0.01*phi*(-V+10)/(np.exp((-V+10)/10)-1)
    beta_n = 0.125*phi*np.exp(-V/80)
    out = dt*(alpha_n*(1-n)-beta_n*n)+n
    return(out)

def m_func(V,n,m):
    global dt
    alpha_m = 0.1*phi*(-V+25)/(np.exp((-V+25)/10)-1)
    beta_m = 4*phi*np.exp(-V/18)
    out = dt*(alpha_m*(1-m)-beta_m*m)+m
    return(out)
	

#HH2D I branch function
def const(I):
    if I < 2:
        return(1)
    else:
        return(1.046*I**(-0.077))

#SAN stimulus function
def stimulus(t):
        return(I_value)

#function to correct cell weight for ECG computation
def sphere(x,y,x0,y0,r):
	return(np.sqrt(r**2-(x-x0)**2-(y-y0)**2))


#-------------------> GEOMETRY AND DIFFUSION

Dx, Dy =1.0,0.0

# side lenght of each cavity's square (in cells)
ra = 76
la = 76
rv = 76
lv = 76

# bundle lenght as a function input current on SAN
bndl = int(0.7*(-0.69*((60/(1.7*I_value**(-0.45)))-60)+85.7))+30

Dmax= max(Dx, Dy)

gap = 10
bach = 7

#Numerical integration constants

T, Tint = 90000, 1       # number of iterades, and intermediate iterates

M=100             #number of nodes in the spatial region


gamma= (1.0/6.0)

gamma1, gamma2 = 1.0/9.0, 1.0/36.0

gx1, gy1 = gamma1*Dx/Dmax, gamma1*Dy/Dmax

gx2, gy2 = gamma2*Dx/Dmax, gamma2*Dy/Dmax

gx1, gx2 = np.empty((ra,ra)), np.empty((ra,ra))
d1v1, d1v2 = np.empty((ra,ra)), np.empty((ra,ra))
d2v1, d2v2 = np.empty((ra,ra)), np.empty((ra,ra))
d4v1, d4v2 = np.empty((ra,ra)), np.empty((ra,ra))
d5v1, d5v2 = np.empty((ra,ra)), np.empty((ra,ra))
dx=np.sqrt(max(Dx,Dy)*dt/gamma) 
gx1.fill(gamma1)
gx2.fill(gamma2)

d1v1.fill(gamma1)
d1v2.fill(gamma2)

d2v1.fill(gamma1)
d2v2.fill(gamma2)

d4v1.fill(gamma1)
d4v2.fill(gamma2)

d5v1.fill(gamma1)
d5v2.fill(gamma2)

Total=T*Tint*dt


#------------------------------------> INITIAL CONDITIONS


V, n, m = np.zeros((2*M+gap,2*M+gap)), np.zeros((2*M+gap,2*M+gap)), np.zeros((2*M+gap,2*M+gap))


Vaux1, naux1, maux1 =np.zeros((ra,ra)), np.zeros((ra,ra)), np.zeros((ra,ra))
Vaux2, naux2, maux2 =np.zeros((la,la)), np.zeros((la,la)), np.zeros((la,la))
Vaux3, naux3, maux3 =np.zeros((rv,rv)), np.zeros((rv,rv)), np.zeros((rv,rv))
Vaux4, naux4, maux4 =np.zeros((lv,lv)), np.zeros((lv,lv)), np.zeros((lv,lv))
Vaux5, naux5, maux5 =np.zeros((lv,lv)), np.zeros((lv,lv)), np.zeros((lv,lv))
Vaux6, naux6, maux6 =np.zeros((lv,lv)), np.zeros((lv,lv)), np.zeros((lv,lv))
Vbundle_aux, nbundle_aux, mbundle_aux =np.zeros(bndl), np.zeros(bndl), np.zeros(bndl)
V_full = np.full((2*ra+10,2*ra+10),0)


V1 = np.full((ra,ra),-10.9506)

D1 = np.full((ra,ra),0)
c1 = np.full((ra,ra),0)
V2 = np.full((la,la),-10.9506)
V3 = np.full((rv,rv),-10.9506)
V4 = np.full((lv,lv),-10.9506)
V5 = np.full((lv,lv),-10.9506)
V6 = np.full((lv,lv),-10.9506)
V8 = np.full((la,la),0)
Vbundle = np.full(bndl,-10.9506)

n1 = np.full((ra,ra),0.1702)
n2 = np.full((la,la),0.1702)
n3 = np.full((rv,rv),0.1702)
n4 = np.full((lv,lv),0.1702)
n5 = np.full((lv,lv),0.1702)
n6 = np.full((lv,lv),0.1702)
nbundle = np.full(bndl,0.1702)

m1 = np.full((ra,ra),0.1702)
m2 = np.full((la,la),0.1702)
m3 = np.full((rv,rv),0.1702)
m4 = np.full((lv,lv),0.1702)
m5 = np.full((lv,lv),0.1702)
m6 = np.full((lv,lv),0.1702)
mbundle = np.full(bndl,0.1702)


V1_coeff = np.zeros((ra,ra))
V3_coeff = np.zeros((rv,rv))
V5_coeff = np.zeros((lv,lv))
V6_coeff = np.zeros((lv,lv))

#------------------------------------> VISUALIZATION


#spec = gridspec.GridSpec(ncols=2, nrows=1,
#                         width_ratios=[1, 2], wspace=0.2,
#                         hspace=0, height_ratios=[1])

# Creates the figure window with 3 subplots, cores cmap=plt.cm.BuPu_r, in reverse order, or .cm.OrRd, or .cm.RdBu

fig=plt.figure(figsize=(3,3)) # the dimensions of the fig in inches


lista_ecgI = list()
lista_ecgII = list()
lista_ecgIII = list()

cmap = plt.get_cmap('RdBu_r')
cmap.set_bad('black',alpha = 1.)

#ax1 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot()

im1 = ax1.imshow(V1, cmap=cmap,interpolation='gaussian',vmin=-12,vmax=120,animated=True,origin="lower")
#cmap.set_bad('black')
ax1.set_xlabel("x")

ax1.set_ylabel("y")

ax1.axis('off')



branch = 160
Py = 20

listax= list()
listaxx= list()
listaxxx= list()

list_time = list()


front_list = list()
front_list_ra = list()
front_list_la = list()
front_list_rv = list()
front_list_lv = list()

back_list1 = list()
back_list2 = list()
back_list3 = list()
back_list4 = list()
front_list_la = list()
back_list1_la = list()
back_list2_la = list()
back_list3_la = list()
back_list4_la = list()

lista_ra = list()
lista_la = list()
lista_rv = list()
lista_lv = list()

V1middle = list()
V2middle = list()
V3middle = list()
V4middle = list()


#generate spirals

#V3[27:28,16:65] = 120
#n3[28:40,16:65] = 0.99


radius_artery = 0 
mv_lenght = 20
mv_height = 10
m1[0:30,30] = 0.80
m1[0:30,29] = 0.4

svc_center = (20,15)

svc_size = 10

ivc_center = (10,45)

ivc_size = 3

pv1_center = (10,10)
pv2_center = (50,10)
pv3_center = (10,50)
pv4_center = (50,50)

pv1_size = 3
pv2_size = 3
pv3_size = 3
pv4_size = 3

mv_lenght = int(ra/3)+1
mv_height = int(mv_lenght/2)
svc_center = (int(ra/3),int(ra/10))

svc_size = 4

ivc_center = (int(0.20*ra),int(0.80*ra))

ivc_size = 4

pv1_center = (int((1/6)*ra),int((1/6)*ra))
pv2_center = (int((5/6)*ra),int((1/6)*ra))
pv3_center = (int((1/6)*ra),int((5/6)*ra))
pv4_center = (int((5/6)*ra),int((5/6)*ra))


pv1_size = 4
pv2_size = 4
pv3_size = 4
pv4_size = 4



SAN_list = list()
bach1_list = list()
bundle1_list = list()
avn_list = list()
ventricle1 = list()
ventricle2 = list()
ventricle3 = list()
epi1 = list()

# Listing all of the coordinates belonging to the front hemisphere (which is used to compute the ECG)
for x in range(ra):
	for y in range(ra):
		a = x
		b = y
		if b <= a+ ra/2 and b >= a- ra/2 and b >= -a + ra/2 and b < -a+ 3*ra/2:
			front_list_ra.append((x,y))
			front_list_la.append((x+ra+gap,y))
			front_list_rv.append((x,y+ra+gap))   
			front_list_lv.append((x+ra+gap,y+ra+gap))

# Correction coefficient to account for curvature of heart surface for ECG signal computation
somaaa = 0
for x in range(0,ra):
	for y in range(0,ra):
		if (x,y) in front_list_ra and y < ra - mv_height:
			V5_coeff[x,y] = sphere(x,y,int(ra/2),int(ra/2),int(ra/2))/int(ra/2)
			V6_coeff[x,y] = sphere(x,y,int(ra/2),int(ra/2),int(ra/2))/int(ra/2)
			somaaa += 2
		if (x,y) in front_list_ra and y > mv_height:	#and x != int(ra/2) and x != int(ra/2)+1 and x != int(ra/2)-1
			V1_coeff[x,y] = sphere(x,y,int(ra/2),int(ra/2),int(ra/2))/int(ra/2)
			V3_coeff[x,y] = sphere(x,y,int(ra/2),int(ra/2),int(ra/2))/int(ra/2)	
			somaaa += 20



import random

#ischemic square region on the Right Atrium#for x in range(25,50):
	for y in range(25,50):
		num = random.random()
		if 0.8 >num:
			val = random.random()*np.sqrt((x-37.5)**2+(y-37.5)**2)/300
			d5v1[x,y] = gamma1*val
			d5v2[x,y] = gamma2*val
			print(val)
		elif num >0.85:
			d5v1[x,y] = 0
			d5v2[x,y] = 0


#-------------------------> MAIN FUNCTION
def update(t):

	global Vaux, naux,maux, V, n,m, con, Tint
	
	I = 0

	for i in range(Tint):


		c1 = np.full((ra,ra),1)

		#normal cells
		Vaux5[1:ra-1,1:ra-1] = V_func_fast(V5[1:ra-1,1:ra-1],n5[1:ra-1,1:ra-1],m5[1:ra-1,1:ra-1], const(I),I) + d5v1[1:ra-1,1:ra-1]*(V5[0:ra-2,1:ra-1]+V5[2:ra,1:ra-1]+V5[1:ra-1,0:ra-2]+V5[1:ra-1,2:ra]-4.0*V5[1:ra-1,1:ra-1]) + d5v2[1:ra-1,1:ra-1]*(V5[0:ra-2,0:ra-2]+V5[0:ra-2,2:ra]+V5[2:ra,0:ra-2]+V5[2:ra,2:ra]-4.0*V5[1:ra-1,1:ra-1])
		naux5[1:ra-1,1:ra-1] = n_func(V5[1:ra-1,1:ra-1],n5[1:ra-1,1:ra-1],m5[1:ra-1,1:ra-1]) 
		
        #left wall
		Vaux5[0,1:ra-1] = V_func_fast(V5[0,1:ra-1],n5[0,1:ra-1],m5[0,1:ra-1], const(I),I) + d5v1[0,1:ra-1]*(V5[1,1:ra-1]+V5[0,0:ra-2]+V5[0,2:ra]+V5[0,ra-2:0:-1]-4.0*V5[0,1:ra-1]) + d5v2[0,1:ra-1]*(V5[1,0:ra-2]+V5[1,2:ra]+V5[0,ra-1:1:-1]+V5[0,ra-3::-1]-4.0*V5[0,1:ra-1])
		naux5[0,1:ra-1] = n_func(V5[0,1:ra-1],n5[0,1:ra-1],m5[0,1:ra-1])
		
		#right wall
		Vaux5[ra-1,1:ra-1]= V_func_fast(V5[ra-1,1:ra-1],n5[ra-1,1:ra-1],m5[ra-1,1:ra-1], const(I),I) + d5v1[ra-1,1:ra-1]*(V5[ra-2,1:ra-1]+V5[ra-1,0:ra-2]+V5[ra-1,2:ra]+V5[ra-1,ra-2:0:-1]-4.0*V5[ra-1,1:ra-1]) + d5v2[ra-1,1:ra-1]*(V5[ra-2,0:ra-2]+V5[ra-2,2:ra]+V5[ra-1,ra-1:1:-1]+V5[ra-1,ra-3::-1]-4.0*V5[ra-1,1:ra-1])
		naux5[ra-1,1:ra-1]= n_func(V5[ra-1,1:ra-1],n5[ra-1,1:ra-1],m5[ra-1,1:ra-1])
		
		#top wall
		Vaux5[1:ra-1,0]= V_func_fast(V5[1:ra-1,0],n5[1:ra-1,0],m5[1:ra-1,0], const(I),I) + d5v1[1:ra-1,0]*(V5[1:ra-1,1]+V5[0:ra-2,0]+V5[2:ra,0]+V5[ra-2:0:-1,0]-4.0*V5[1:ra-1,0]) + d5v2[1:ra-1,0]*(V5[0:ra-2,1]+V5[2:ra,1]+V5[ra-1:1:-1,0]+V5[ra-3::-1,0]-4.0*V5[1:ra-1,0])
		naux5[1:ra-1,0]= n_func(V5[1:ra-1,0],n5[1:ra-1,0],m5[1:ra-1,0])
		
		#bottora wall
		Vaux5[1:ra-1,ra-1]= V_func_fast(V5[1:ra-1,ra-1],n5[1:ra-1,ra-1],m5[1:ra-1,ra-1], const(I),I) + d5v1[1:ra-1,ra-1]*(V5[1:ra-1,ra-2]+V5[0:ra-2,ra-1]+V5[2:ra,ra-1]+V5[ra-2:0:-1,ra-1]-4.0*V5[1:ra-1,ra-1]) + d5v2[1:ra-1,ra-1]*(V5[0:ra-2,ra-2]+V5[2:ra,ra-2]+V5[ra-1:1:-1,ra-1]+V5[ra-3::-1,ra-1]-4.0*V5[1:ra-1,ra-1])
		naux5[1:ra-1,ra-1]= n_func(V5[1:ra-1,ra-1],n5[1:ra-1,ra-1],m5[1:ra-1,ra-1])
		

		Vaux5[0,0]= V_func_fast(V5[0,0],n5[0,0],m5[0,0], const(I),I) + d5v1[0,0]*(V5[1,0]+V5[0,1]+V5[ra-1,0]+V5[0,ra-1]-4.0*V5[0,0]) + d5v2[0,0]*(V5[1,1]+V5[ra-1,ra-1]+V5[0,ra-2]+V5[ra-2,0]-4*V5[0,0])
		naux5[0,0]= n_func(V5[0,0],n5[0,0],m5[0,0])
		

		Vaux5[0,ra-1]= V_func_fast(V5[0,ra-1],n5[0,ra-1],m5[0,ra-1], const(I),I) + d5v1[0,ra-1]*(V5[1,ra-1]+V5[0,ra-2]+V5[0,0]+V5[ra-1,ra-1]-4.0*V5[0,ra-1]) + d5v2[0,ra-1]*(V5[1,ra-2]+V5[0,1]+V5[1,0]+V5[ra-1,0]-4*V5[0,ra-1])
		naux5[0,ra-1]= n_func(V5[0,ra-1],n5[0,ra-1],m5[0,ra-1])
		

		Vaux5[ra-1,0]= V_func_fast(V5[ra-1,0],n5[ra-1,0],m5[ra-1,0], const(I),I) + d5v1[ra-1,0]*(V5[ra-1,1]+V5[ra-2,0]+V5[0,0]+V5[ra-1,ra-1]-4.0*V5[ra-1,0]) + d5v2[ra-1,0]*(V5[ra-2,1]+V5[1,0]+V5[0,1]+V5[0,ra-1]-4*V5[ra-1,0])
		naux5[ra-1,0]= n_func(V5[ra-1,0],n5[ra-1,0],m5[ra-1,0])
		

		Vaux5[ra-1,ra-1]= V_func_fast(V5[ra-1,ra-1],n5[ra-1,ra-1],m5[ra-1,ra-1], const(I),I) + d5v1[ra-1,ra-1]*(V5[ra-2,ra-1]+V5[ra-1,ra-2]+V5[ra-1,0]+V5[0,ra-1]-4.0*V5[ra-1,ra-1]) + d5v2[ra-1,ra-1]*(V5[ra-2,ra-2]+V5[1,ra-1]+V5[ra-1,1]+V5[0,0]-4*V5[ra-1,ra-1])
		naux5[ra-1,ra-1]= n_func(V5[ra-1,ra-1],n5[ra-1,ra-1],m5[ra-1,ra-1])
		

		Vaux5[int(ra/4),int(ra/4)]= V_func_fast(V5[int(ra/4),int(ra/4)],n5[int(ra/4),int(ra/4)],m5[int(ra/4),int(ra/4)], const(stimulus(t)),stimulus(t))
		naux5[int(ra/4),int(ra/4)]= n_func(V5[int(ra/4),int(ra/4)],n5[int(ra/4),int(ra/4)],m5[int(ra/4),int(ra/4)])
		
		#mitral valve

		Vaux5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height:ra] = 0
		naux5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height:ra] = 0

		Vaux5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]= V_func_slow(V5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],n5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],m5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1], c1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],I) + d5v1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]*(V5[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,ra-mv_height-1]+V5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-2]+V5[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,ra-mv_height-1]-3.0*V5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]) + d5v2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]*(V5[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,ra-mv_height-2]+V5[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,ra-mv_height-2]-2.0*V5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1])
		naux5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]= n_func(V5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],n5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],m5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1])
		
		Vaux5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]= V_func_slow(V5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],n5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],m5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1], c1[ra-1,0],I) + d5v1[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]*(V5[int(ra/2+mv_lenght/2)+1,ra-mv_height:ra-1]+V5[int(ra/2+mv_lenght/2)+1,ra-mv_height-1:ra-2]+V5[int(ra/2+mv_lenght/2)+1,ra-mv_height+1:ra]-3.0*V5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]) + d5v2[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]*(V5[int(ra/2+mv_lenght/2)+1,ra-mv_height+1:ra]+V5[int(ra/2+mv_lenght/2)+1,ra-mv_height-1:ra-2]-2*V5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1])
		naux5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]=      n_func(V5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],n5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],m5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1])
		
		Vaux5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]= V_func_slow(V5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],n5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1], c1[ra-1,0],I) + d5v1[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]*(V5[int(ra/2-mv_lenght/2)-2,ra-mv_height:ra-1]+V5[int(ra/2-mv_lenght/2)-1,ra-mv_height-1:ra-2]+V5[int(ra/2-mv_lenght/2)-1,ra-mv_height+1:ra]-3.0*V5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]) + d5v2[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]*(V5[int(ra/2-mv_lenght/2)-1-1,ra-mv_height+1:ra]+V5[int(ra/2-mv_lenght/2)-1-1,ra-mv_height-1:ra-2]-2*V5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1])
		naux5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]=      n_func(V5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],n5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1])
		
		Vaux5[int(ra/2+mv_lenght/2),ra-1]= V_func_slow(V5[int(ra/2+mv_lenght/2),ra-1],n5[int(ra/2+mv_lenght/2),ra-1],m5[int(ra/2+mv_lenght/2),ra-1], c1[int(ra/2+mv_lenght/2),ra-1],I) + d5v1[int(ra/2+mv_lenght/2),ra-1]*(V5[int(ra/2+mv_lenght/2),ra-2]+V5[int(ra/2-mv_lenght/2)-1,ra-1]+V5[int(ra/2+mv_lenght/2)+1,ra-1]-3.0*V5[int(ra/2+mv_lenght/2),ra-1]) + d5v2[int(ra/2+mv_lenght/2),ra-1]*(V5[int(ra/2+mv_lenght/2)+1,ra-2]+V5[int(ra/2-mv_lenght/2)-1,ra-1]-2*V5[int(ra/2+mv_lenght/2),ra-1])
		naux5[int(ra/2+mv_lenght/2),ra-1]= n_func(V5[int(ra/2+mv_lenght/2),ra-1],n5[int(ra/2+mv_lenght/2),ra-1],m5[int(ra/2+mv_lenght/2),ra-1])
		
		Vaux5[int(ra/2-mv_lenght/2)-1,ra-1]= V_func_slow(V5[int(ra/2-mv_lenght/2)-1,ra-1],n5[int(ra/2-mv_lenght/2)-1,ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-1], c1[int(ra/2-mv_lenght/2)-1,ra-1],I) + d5v1[int(ra/2-mv_lenght/2)-1,ra-1]*(V5[int(ra/2-mv_lenght/2)-1,ra-2]+V5[int(ra/2-mv_lenght/2)-1,ra-1]+V5[int(ra/2+mv_lenght/2),ra-1]-3.0*V5[int(ra/2-mv_lenght/2)-1,ra-1]) + d5v2[int(ra/2-mv_lenght/2)-1,ra-1]*(V5[int(ra/2+mv_lenght/2),ra-2]+V5[int(ra/2-mv_lenght/2)-1,ra-1]-2*V5[int(ra/2-mv_lenght/2)-1,ra-1])
		naux5[int(ra/2-mv_lenght/2)-1,ra-1]= n_func(V5[int(ra/2-mv_lenght/2)-1,ra-1],n5[int(ra/2-mv_lenght/2)-1,ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-1])
		

		#superior vena cava

		Vaux5[svc_center[0]-svc_size:svc_center[0]+svc_size,svc_center[1]-svc_size:svc_center[1]+svc_size] = 0
		naux5[svc_center[0]-svc_size:svc_center[0]+svc_size,svc_center[1]-svc_size:svc_center[1]+svc_size] = 0

		#svc left wall
		Vaux5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size] = V_func_fast(V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size],n5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size],m5[0,svc_center[1]-svc_size:svc_center[1]+svc_size], const(I),I) + d5v1[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size-2,svc_center[1]-svc_size:svc_center[1]+svc_size]+V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1:svc_center[1]+svc_size-1]+V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size+1:svc_center[1]+svc_size+1]-3.0*V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size]) + d5v2[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1:svc_center[1]+svc_size-1]+V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size+1:svc_center[1]+svc_size+1]-2.0*V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size])
		naux5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size] =      n_func(V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size],n5[svc_center[0]-svc_size-1,svc_center[1]-svc_size:svc_center[1]+svc_size],m5[0,svc_center[1]-svc_size:svc_center[1]+svc_size])
		
		#svc right wall
		Vaux5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size] = V_func_fast(V5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size],n5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size],m5[0,svc_center[1]-svc_size:svc_center[1]+svc_size], const(I),I) + d5v1[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size]*(V5[svc_center[0]+svc_size+1,svc_center[1]-svc_size:svc_center[1]+svc_size]+V5[svc_center[0] + svc_size ,svc_center[1]-svc_size-1:svc_center[1]+svc_size-1]+V5[svc_center[0] + svc_size ,svc_center[1]-svc_size+1:svc_center[1]+svc_size+1]-3.0*V5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size]) + d5v2[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size]*(V5[svc_center[0] + svc_size +1,svc_center[1]-svc_size-1:svc_center[1]+svc_size-1]+V5[svc_center[0] + svc_size+1 ,svc_center[1]-svc_size+1:svc_center[1]+svc_size+1]-2.0*V5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size])
		naux5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size] =      n_func(V5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size],n5[svc_center[0] + svc_size ,svc_center[1]-svc_size:svc_center[1]+svc_size],m5[0,svc_center[1]-svc_size:svc_center[1]+svc_size])
		
		#svc top wall
		Vaux5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1] = V_func_fast(V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1],n5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1],m5[0,svc_center[1]-svc_size-1], const(I),I) + d5v1[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1]*(V5[svc_center[0]-svc_size+1:svc_center[0]+svc_size+1 ,svc_center[1]-svc_size-1]+V5[svc_center[0]-svc_size-1:svc_center[0]+svc_size-1 ,svc_center[1]-svc_size-1]+V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1-1]-3.0*V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1]) + d5v2[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1]*(V5[svc_center[0]-svc_size+1:svc_center[0]+svc_size +1,svc_center[1]-svc_size-1-1]+V5[svc_center[0]-svc_size-1:svc_center[0]+svc_size-1 ,svc_center[1]-svc_size-1-1]-2.0*V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1])
		naux5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1] =      n_func(V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1],n5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]-svc_size-1],m5[0,svc_center[1]-svc_size-1])
		
		#svc bottom wall
		Vaux5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size] = V_func_fast(V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size],n5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size],m5[0,svc_center[1]+svc_size], const(I),I) + d5v1[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size+1:svc_center[0]+svc_size+1 ,svc_center[1]+svc_size]+V5[svc_center[0]-svc_size-1:svc_center[0]+svc_size-1 ,svc_center[1]+svc_size]+V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size+1]-3.0*V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size]) + d5v2[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size+1:svc_center[0]+svc_size +1,svc_center[1]+svc_size+1]+V5[svc_center[0]-svc_size-1:svc_center[0]+svc_size-1 ,svc_center[1]+svc_size+1]-2.0*V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size])
		naux5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size] =      n_func(V5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size],n5[svc_center[0]-svc_size:svc_center[0]+svc_size ,svc_center[1]+svc_size],m5[0,svc_center[1]+svc_size])
		
		#svc corner ++
		Vaux5[svc_center[0]+svc_size,svc_center[1]+svc_size] =      V_func_fast(V5[svc_center[0]+svc_size,svc_center[1]+svc_size],n5[svc_center[0]+svc_size,svc_center[1]+svc_size],m6[0,svc_center[1]+svc_size], const(I),I) + d2v1[svc_center[0]+svc_size,svc_center[1]+svc_size]*(V5[svc_center[0]+svc_size+1,svc_center[1]+svc_size]+V5[svc_center[0]+svc_size-1,svc_center[1]+svc_size]+V5[svc_center[0]+svc_size,svc_center[1]+svc_size+1]+V5[svc_center[0]+svc_size,svc_center[1]+svc_size-1]-4.0*V5[svc_center[0]+svc_size,svc_center[1]+svc_size]) + d2v2[svc_center[0]+svc_size,svc_center[1]+svc_size]*(V5[svc_center[0]+svc_size+1,svc_center[1]+svc_size+1]+V5[svc_center[0]+svc_size+1,svc_center[1]+svc_size-1]+V5[svc_center[0]+svc_size-1,svc_center[1]+svc_size+1]-3.0*V5[svc_center[0]+svc_size,svc_center[1]+svc_size])
		naux5[svc_center[0]+svc_size,svc_center[1]+svc_size] =      n_func(V5[svc_center[0]+svc_size,svc_center[1]+svc_size],n5[svc_center[0]+svc_size,svc_center[1]+svc_size],m6[0,svc_center[1]+svc_size])
		#svc corner +-
		Vaux5[svc_center[0]+svc_size,svc_center[1]-svc_size-1] =      V_func_fast(V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1],n5[svc_center[0]+svc_size,svc_center[1]-svc_size-1],m6[0,svc_center[1]-svc_size-1], const(I),I) + d2v1[svc_center[0]+svc_size,svc_center[1]-svc_size-1]*(V5[svc_center[0]+svc_size+1,svc_center[1]-svc_size-1]+V5[svc_center[0]+svc_size-1,svc_center[1]-svc_size-1]+V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1+1]+V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1-1]-4.0*V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1]) + d2v2[svc_center[0]+svc_size,svc_center[1]-svc_size-1]*(V5[svc_center[0]+svc_size+1,svc_center[1]-svc_size-1+1]+V5[svc_center[0]+svc_size-1,svc_center[1]-svc_size-1-1]+V5[svc_center[0]+svc_size+1,svc_center[1]-svc_size-1-1]-3.0*V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1])
		naux5[svc_center[0]+svc_size,svc_center[1]-svc_size-1] =      n_func(V5[svc_center[0]+svc_size,svc_center[1]-svc_size-1],n5[svc_center[0]+svc_size,svc_center[1]-svc_size-1],m6[0,svc_center[1]-svc_size-1])
		#svc corner --
		Vaux5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1] =      V_func_fast(V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1],n5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1],m6[0,svc_center[1]-svc_size-1], const(I),I) + d2v1[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1]*(V5[svc_center[0]-svc_size-1+1,svc_center[1]-svc_size-1]+V5[svc_center[0]-svc_size-1-1,svc_center[1]-svc_size-1]+V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1+1]+V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1-1]-4.0*V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1]) + d2v2[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1]*(V5[svc_center[0]-svc_size-1-1,svc_center[1]-svc_size-1-1]+V5[svc_center[0]-svc_size-1+1,svc_center[1]-svc_size-1-1]+V5[svc_center[0]-svc_size-1-1,svc_center[1]-svc_size-1+1]-3.0*V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1])
		naux5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1] =      n_func(V5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1],n5[svc_center[0]-svc_size-1,svc_center[1]-svc_size-1],m6[0,svc_center[1]-svc_size-1])
		#svc corner -+
		Vaux5[svc_center[0]-svc_size-1,svc_center[1]+svc_size] =      V_func_fast(V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size],n5[svc_center[0]-svc_size-1,svc_center[1]+svc_size],m6[0,svc_center[1]+svc_size], const(I),I) + d2v1[svc_center[0]-svc_size-1,svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size-1+1,svc_center[1]+svc_size]+V5[svc_center[0]-svc_size-1-1,svc_center[1]+svc_size]+V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size+1]+V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size-1]-4.0*V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size]) + d2v2[svc_center[0]-svc_size-1,svc_center[1]+svc_size]*(V5[svc_center[0]-svc_size-1+1,svc_center[1]+svc_size+1]+V5[svc_center[0]-svc_size-1-1,svc_center[1]+svc_size-1]+V5[svc_center[0]-svc_size-1-1,svc_center[1]+svc_size+1]-3.0*V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size])
		naux5[svc_center[0]-svc_size-1,svc_center[1]+svc_size] =      n_func(V5[svc_center[0]-svc_size-1,svc_center[1]+svc_size],n5[svc_center[0]-svc_size-1,svc_center[1]+svc_size],m6[0,svc_center[1]+svc_size])

		#inferior vena cava

		Vaux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] = 0
		naux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] = 0

		#ivc left wall
		Vaux5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] = V_func_fast(V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],m5[0,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size], const(I),I) + d5v1[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size-2,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1:ivc_center[1]+ivc_size-1]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size+1:ivc_center[1]+ivc_size+1]-3.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]) + d5v2[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1:ivc_center[1]+ivc_size-1]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size+1:ivc_center[1]+ivc_size+1]-2.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size])
		naux5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] =      n_func(V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],m5[0,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size])
		
		#ivc right wall
		Vaux5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] = V_func_fast(V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],n5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],m5[0,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size], const(I),I) + d5v1[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]*(V5[ivc_center[0]+ivc_size+1,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]+V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size-1:ivc_center[1]+ivc_size-1]+V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size+1:ivc_center[1]+ivc_size+1]-3.0*V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]) + d5v2[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size]*(V5[ivc_center[0] + ivc_size +1,ivc_center[1]-ivc_size-1:ivc_center[1]+ivc_size-1]+V5[ivc_center[0] + ivc_size+1 ,ivc_center[1]-ivc_size+1:ivc_center[1]+ivc_size+1]-2.0*V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size])
		naux5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size] =      n_func(V5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],n5[ivc_center[0] + ivc_size ,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size],m5[0,ivc_center[1]-ivc_size:ivc_center[1]+ivc_size])
		
		#ivc top wall
		Vaux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1] = V_func_fast(V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1],n5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1],m5[0,ivc_center[1]-ivc_size-1], const(I),I) + d5v1[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]-ivc_size+1:ivc_center[0]+ivc_size+1 ,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]-ivc_size-1:ivc_center[0]+ivc_size-1 ,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1-1]-3.0*V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1]) + d5v2[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]-ivc_size+1:ivc_center[0]+ivc_size +1,ivc_center[1]-ivc_size-1-1]+V5[ivc_center[0]-ivc_size-1:ivc_center[0]+ivc_size-1 ,ivc_center[1]-ivc_size-1-1]-2.0*V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1])
		naux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1] =      n_func(V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1],n5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]-ivc_size-1],m5[0,ivc_center[1]-ivc_size-1])
		
		#ivc bottom wall
		Vaux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size] = V_func_fast(V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size],m5[0,ivc_center[1]+ivc_size], const(I),I) + d5v1[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size+1:ivc_center[0]+ivc_size+1 ,ivc_center[1]+ivc_size]+V5[ivc_center[0]-ivc_size-1:ivc_center[0]+ivc_size-1 ,ivc_center[1]+ivc_size]+V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size+1]-3.0*V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size]) + d5v2[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size+1:ivc_center[0]+ivc_size +1,ivc_center[1]+ivc_size+1]+V5[ivc_center[0]-ivc_size-1:ivc_center[0]+ivc_size-1 ,ivc_center[1]+ivc_size+1]-2.0*V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size])
		naux5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size] =      n_func(V5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size:ivc_center[0]+ivc_size ,ivc_center[1]+ivc_size],m5[0,ivc_center[1]+ivc_size])
				
		#ivc corner ++
		Vaux5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size] =      V_func_fast(V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size],n5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size],m6[0,ivc_center[1]+ivc_size], const(I),I) + d2v1[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size]*(V5[ivc_center[0]+ivc_size+1,ivc_center[1]+ivc_size]+V5[ivc_center[0]+ivc_size-1,ivc_center[1]+ivc_size]+V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size+1]+V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size-1]-4.0*V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size]) + d2v2[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size]*(V5[ivc_center[0]+ivc_size+1,ivc_center[1]+ivc_size+1]+V5[ivc_center[0]+ivc_size+1,ivc_center[1]+ivc_size-1]+V5[ivc_center[0]+ivc_size-1,ivc_center[1]+ivc_size+1]-3.0*V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size])
		naux5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size] =      n_func(V5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size],n5[ivc_center[0]+ivc_size,ivc_center[1]+ivc_size],m6[0,ivc_center[1]+ivc_size])
		#ivc corner +-
		Vaux5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1] =      V_func_fast(V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1],n5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1],m6[0,ivc_center[1]-ivc_size-1], const(I),I) + d2v1[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]+ivc_size+1,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]+ivc_size-1,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1+1]+V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1-1]-4.0*V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1]) + d2v2[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]+ivc_size+1,ivc_center[1]-ivc_size-1+1]+V5[ivc_center[0]+ivc_size-1,ivc_center[1]-ivc_size-1-1]+V5[ivc_center[0]+ivc_size+1,ivc_center[1]-ivc_size-1-1]-3.0*V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1])
		naux5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1] =      n_func(V5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1],n5[ivc_center[0]+ivc_size,ivc_center[1]-ivc_size-1],m6[0,ivc_center[1]-ivc_size-1])
		#ivc corner --
		Vaux5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1] =      V_func_fast(V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1],n5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1],m6[0,ivc_center[1]-ivc_size-1], const(I),I) + d2v1[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]-ivc_size-1+1,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]-ivc_size-1]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1+1]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1-1]-4.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1]) + d2v2[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1]*(V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]-ivc_size-1-1]+V5[ivc_center[0]-ivc_size-1+1,ivc_center[1]-ivc_size-1-1]+V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]-ivc_size-1+1]-3.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1])
		naux5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1] =      n_func(V5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1],n5[ivc_center[0]-ivc_size-1,ivc_center[1]-ivc_size-1],m6[0,ivc_center[1]-ivc_size-1])
		#ivc corner -+
		Vaux5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size] =      V_func_fast(V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size],m6[0,ivc_center[1]+ivc_size], const(I),I) + d2v1[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size-1+1,ivc_center[1]+ivc_size]+V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]+ivc_size]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size+1]+V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size-1]-4.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size]) + d2v2[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size]*(V5[ivc_center[0]-ivc_size-1+1,ivc_center[1]+ivc_size+1]+V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]+ivc_size-1]+V5[ivc_center[0]-ivc_size-1-1,ivc_center[1]+ivc_size+1]-3.0*V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size])
		naux5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size] =      n_func(V5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size],n5[ivc_center[0]-ivc_size-1,ivc_center[1]+ivc_size],m6[0,ivc_center[1]+ivc_size])

		#normal cells
		Vaux6[1:la-1,1:la-1] = V_func_fast(V6[1:la-1,1:la-1],n6[1:la-1,1:la-1],m5[1:la-1,1:la-1], const(I),I) + d2v1[1:la-1,1:la-1]*(V6[0:la-2,1:la-1]+V6[2:la,1:la-1]+V6[1:la-1,0:la-2]+V6[1:la-1,2:la]-4.0*V6[1:la-1,1:la-1]) + d2v2[1:la-1,1:la-1]*(V6[0:la-2,0:la-2]+V6[0:la-2,2:la]+V6[2:la,0:la-2]+V6[2:la,2:la]-4.0*V6[1:la-1,1:la-1])
		naux6[1:la-1,1:la-1] = n_func(V6[1:la-1,1:la-1],n6[1:la-1,1:la-1],m5[1:la-1,1:la-1]) 
		
        #left wall
		Vaux6[0,1:la-1] = V_func_fast(V6[0,1:la-1],n6[0,1:la-1],m5[0,1:la-1], const(I),I) + d2v1[0,1:la-1]*(V6[1,1:la-1]+V6[0,0:la-2]+V6[0,2:la]+V6[0,la-2:0:-1]-4.0*V6[0,1:la-1]) + d2v2[0,1:la-1]*(V6[1,0:la-2]+V6[1,2:la]+V6[0,la-1:1:-1]+V6[0,la-3::-1]-4.0*V6[0,1:la-1])
		naux6[0,1:la-1] = n_func(V6[0,1:la-1],n6[0,1:la-1],m5[0,1:la-1])
		
		#right wall
		Vaux6[la-1,1:la-1]= V_func_fast(V6[la-1,1:la-1],n6[la-1,1:la-1],m5[la-1,1:la-1], const(I),I) + d2v1[la-1,1:la-1]*(V6[la-2,1:la-1]+V6[la-1,0:la-2]+V6[la-1,2:la]+V6[la-1,la-2:0:-1]-4.0*V6[la-1,1:la-1]) + d2v2[la-1,1:la-1]*(V6[la-2,0:la-2]+V6[la-2,2:la]+V6[la-1,la-1:1:-1]+V6[la-1,la-3::-1]-4.0*V6[la-1,1:la-1])
		naux6[la-1,1:la-1]= n_func(V6[la-1,1:la-1],n6[la-1,1:la-1],m5[la-1,1:la-1])
		
		#top wall
		Vaux6[1:la-1,0]= V_func_fast(V6[1:la-1,0],n6[1:la-1,0],m5[1:la-1,0], const(I),I) + d2v1[1:la-1,0]*(V6[1:la-1,1]+V6[0:la-2,0]+V6[2:la,0]+V6[la-2:0:-1,0]-4.0*V6[1:la-1,0]) + d2v2[1:la-1,0]*(V6[0:la-2,1]+V6[2:la,1]+V6[la-1:1:-1,0]+V6[la-3::-1,0]-4.0*V6[1:la-1,0])
		naux6[1:la-1,0]= n_func(V6[1:la-1,0],n6[1:la-1,0],m5[1:la-1,0])
		
		#bottola wall
		Vaux6[1:la-1,la-1]= V_func_fast(V6[1:la-1,la-1],n6[1:la-1,la-1],m5[1:la-1,la-1], const(I),I) + d2v1[1:la-1,la-1]*(V6[1:la-1,la-2]+V6[0:la-2,la-1]+V6[2:la,la-1]+V6[la-2:0:-1,la-1]-4.0*V6[1:la-1,la-1]) + d2v2[1:la-1,la-1]*(V6[0:la-2,la-2]+V6[2:la,la-2]+V6[la-1:1:-1,la-1]+V6[la-3::-1,la-1]-4.0*V6[1:la-1,la-1])
		naux6[1:la-1,la-1]= n_func(V6[1:la-1,la-1],n6[1:la-1,la-1],m5[1:la-1,la-1])
		

		Vaux6[0,0]= V_func_fast(V6[0,0],n6[0,0],m5[0,0], const(I),I) + d2v1[0,0]*(V6[1,0]+V6[0,1]+V6[la-1,0]+V6[0,la-1]-4.0*V6[0,0]) + d2v2[0,0]*(V6[1,1]+V6[la-1,la-1]+V6[0,la-2]+V6[la-2,0]-4*V6[0,0])
		naux6[0,0]= n_func(V6[0,0],n6[0,0],m5[0,0])
		

		Vaux6[0,la-1]= V_func_fast(V6[0,la-1],n6[0,la-1],m5[0,la-1], const(I),I) + d2v1[0,la-1]*(V6[1,la-1]+V6[0,la-2]+V6[0,0]+V6[la-1,la-1]-4.0*V6[0,la-1]) + d2v2[0,la-1]*(V6[1,la-2]+V6[1,0]+V6[0,1]+V6[la-1,0]-4*V6[0,la-1])
		naux6[0,la-1]= n_func(V6[0,la-1],n6[0,la-1],m5[0,la-1])
		

		Vaux6[la-1,0]= V_func_fast(V6[la-1,0],n6[la-1,0],m5[la-1,0], const(I),I) + d2v1[la-1,0]*(V6[la-1,1]+V6[la-2,0]+V6[0,0]+V6[la-1,la-1]-4.0*V6[la-1,0]) + d2v2[la-1,0]*(V6[la-2,1]+V6[1,0]+V6[0,1]+V6[0,la-1]-4*V6[la-1,0])
		naux6[la-1,0]= n_func(V6[la-1,0],n6[la-1,0],m5[la-1,0])
		

		Vaux6[la-1,la-1]= V_func_fast(V6[la-1,la-1],n6[la-1,la-1],m5[la-1,la-1], const(I),I) + d2v1[la-1,la-1]*(V6[la-2,la-1]+V6[la-1,la-2]+V6[la-1,0]+V6[0,la-1]-4.0*V6[la-1,la-1]) + d2v2[la-1,la-1]*(V6[la-2,la-2]+V6[1,la-1]+V6[la-1,1]+V6[0,0]-4*V6[la-1,la-1])
		naux6[la-1,la-1]= n_func(V6[la-1,la-1],n6[la-1,la-1],m5[la-1,la-1])
		

		Vaux6[int(la/4),int(la/4)]= V_func_fast(V6[int(la/4),int(la/4)],n6[int(la/4),int(la/4)],m6[int(la/4),int(la/4)], const(0),0) + 1/6*(V5[int(ra*0.4),int(ra/4)]-V6[int(la/4),int(la/4)])
		naux6[int(la/4),int(la/4)]= n_func(V6[int(la/4),int(la/4)],n6[int(la/4),int(la/4)],m6[int(la/4),int(la/4)])
		maux6[int(la/4),int(la/4)]= m_func(V6[int(la/4),int(la/4)],n6[int(la/4),int(la/4)],m6[int(la/4),int(la/4)])

		#mitral valve

		Vaux6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height:ra] = 0
		naux6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height:ra] = 0

		Vaux6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]= V_func_slow(V6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],n6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],m5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1], c1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],I) + d2v1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]*(V6[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,ra-mv_height-1]+V6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-2]+V6[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,ra-mv_height-1]-3.0*V6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]) + d2v2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]*(V6[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,ra-mv_height-2]+V6[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,ra-mv_height-2]-2.0*V6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1])
		naux6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1]= n_func(V6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],n6[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1],m5[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),ra-mv_height-1])
		
		Vaux6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]= V_func_slow(V6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],n6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],m5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1], c1[ra-1,0],I) + d2v1[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]*(V6[int(ra/2+mv_lenght/2)+1,ra-mv_height:ra-1]+V6[int(ra/2+mv_lenght/2)+1,ra-mv_height-1:ra-2]+V6[int(ra/2+mv_lenght/2)+1,ra-mv_height+1:ra]-3.0*V6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]) + d2v2[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]*(V6[int(ra/2+mv_lenght/2)+1,ra-mv_height+1:ra]+V6[int(ra/2+mv_lenght/2)+1,ra-mv_height-1:ra-2]-2*V6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1])
		naux6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1]=      n_func(V6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],n6[int(ra/2+mv_lenght/2),ra-mv_height:ra-1],m5[int(ra/2+mv_lenght/2),ra-mv_height:ra-1])
		
		Vaux6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]= V_func_slow(V6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],n6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1], c1[ra-1,0],I) + d2v1[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]*(V6[int(ra/2-mv_lenght/2)-2,ra-mv_height:ra-1]+V6[int(ra/2-mv_lenght/2)-1,ra-mv_height-1:ra-2]+V6[int(ra/2-mv_lenght/2)-1,ra-mv_height+1:ra]-3.0*V6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]) + d2v2[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]*(V6[int(ra/2-mv_lenght/2)-1-1,ra-mv_height+1:ra]+V6[int(ra/2-mv_lenght/2)-1-1,ra-mv_height-1:ra-2]-2*V6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1])
		naux6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1]=      n_func(V6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],n6[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-mv_height:ra-1])
		
		Vaux6[int(ra/2+mv_lenght/2),ra-1]= V_func_slow(V6[int(ra/2+mv_lenght/2),ra-1],n6[int(ra/2+mv_lenght/2),ra-1],m5[int(ra/2+mv_lenght/2),ra-1], c1[int(ra/2+mv_lenght/2),ra-1],I) + d2v1[int(ra/2+mv_lenght/2),ra-1]*(V6[int(ra/2+mv_lenght/2),ra-2]+V6[int(ra/2-mv_lenght/2)-1,ra-1]+V6[int(ra/2+mv_lenght/2)+1,ra-1]-3.0*V6[int(ra/2+mv_lenght/2),ra-1]) + d2v2[int(ra/2+mv_lenght/2),ra-1]*(V6[int(ra/2+mv_lenght/2)+1,ra-2]+V6[int(ra/2-mv_lenght/2)-1,ra-1]-2*V6[int(ra/2+mv_lenght/2),ra-1])
		naux6[int(ra/2+mv_lenght/2),ra-1]= n_func(V6[int(ra/2+mv_lenght/2),ra-1],n6[int(ra/2+mv_lenght/2),ra-1],m5[int(ra/2+mv_lenght/2),ra-1])
		
		Vaux6[int(ra/2-mv_lenght/2)-1,ra-1]= V_func_slow(V6[int(ra/2-mv_lenght/2)-1,ra-1],n6[int(ra/2-mv_lenght/2)-1,ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-1], c1[int(ra/2-mv_lenght/2)-1,ra-1],I) + d2v1[int(ra/2-mv_lenght/2)-1,ra-1]*(V6[int(ra/2-mv_lenght/2)-1,ra-2]+V6[int(ra/2-mv_lenght/2)-1,ra-1]+V6[int(ra/2+mv_lenght/2),ra-1]-3.0*V6[int(ra/2-mv_lenght/2)-1,ra-1]) + d2v2[int(ra/2-mv_lenght/2)-1,ra-1]*(V6[int(ra/2+mv_lenght/2),ra-2]+V6[int(ra/2-mv_lenght/2)-1,ra-1]-2*V6[int(ra/2-mv_lenght/2)-1,ra-1])
		naux6[int(ra/2-mv_lenght/2)-1,ra-1]= n_func(V6[int(ra/2-mv_lenght/2)-1,ra-1],n6[int(ra/2-mv_lenght/2)-1,ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-1])
		maux6[int(ra/2-mv_lenght/2)-1,ra-1]= m_func(V6[int(ra/2-mv_lenght/2)-1,ra-1],n6[int(ra/2-mv_lenght/2)-1,ra-1],m5[int(ra/2-mv_lenght/2)-1,ra-1])
		
		#pulmonary veins
		Vaux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] = 0
		naux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] = 0

		Vaux6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] = V_func_fast(V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],m5[0,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size], const(I),I) + d2v1[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size-2,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1:pv1_center[1]+pv1_size-1]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size+1:pv1_center[1]+pv1_size+1]-3.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]) + d2v2[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1:pv1_center[1]+pv1_size-1]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size+1:pv1_center[1]+pv1_size+1]-2.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size])
		naux6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] =      n_func(V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],m5[0,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size])
		
		#pv1 right wall
		Vaux6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] = V_func_fast(V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],n6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],m6[0,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size], const(I),I) + d2v1[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]*(V6[pv1_center[0]+pv1_size+1,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]+V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size-1:pv1_center[1]+pv1_size-1]+V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size+1:pv1_center[1]+pv1_size+1]-3.0*V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]) + d2v2[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size]*(V6[pv1_center[0] + pv1_size +1,pv1_center[1]-pv1_size-1:pv1_center[1]+pv1_size-1]+V6[pv1_center[0] + pv1_size+1 ,pv1_center[1]-pv1_size+1:pv1_center[1]+pv1_size+1]-2.0*V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size])
		naux6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size] =      n_func(V6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],n6[pv1_center[0] + pv1_size ,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size],m6[0,pv1_center[1]-pv1_size:pv1_center[1]+pv1_size])
		
		#pv1 top wall
		Vaux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1] = V_func_fast(V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1],n6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1], const(I),I) + d2v1[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]-pv1_size+1:pv1_center[0]+pv1_size+1 ,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]-pv1_size-1:pv1_center[0]+pv1_size-1 ,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1-1]-3.0*V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1]) + d2v2[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]-pv1_size+1:pv1_center[0]+pv1_size +1,pv1_center[1]-pv1_size-1-1]+V6[pv1_center[0]-pv1_size-1:pv1_center[0]+pv1_size-1 ,pv1_center[1]-pv1_size-1-1]-2.0*V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1])
		naux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1] =      n_func(V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1],n6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1])
		
		#pv1 bottom wall
		Vaux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size] = V_func_fast(V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size], const(I),I) + d2v1[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size+1:pv1_center[0]+pv1_size+1 ,pv1_center[1]+pv1_size]+V6[pv1_center[0]-pv1_size-1:pv1_center[0]+pv1_size-1 ,pv1_center[1]+pv1_size]+V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size+1]-3.0*V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size]) + d2v2[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size+1:pv1_center[0]+pv1_size +1,pv1_center[1]+pv1_size+1]+V6[pv1_center[0]-pv1_size-1:pv1_center[0]+pv1_size-1 ,pv1_center[1]+pv1_size+1]-2.0*V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size])
		naux6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size] =      n_func(V6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size:pv1_center[0]+pv1_size ,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size])
		
		#pv1 corner ++
		Vaux6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size] =      V_func_fast(V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size],n6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size], const(I),I) + d2v1[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size]*(V6[pv1_center[0]+pv1_size+1,pv1_center[1]+pv1_size]+V6[pv1_center[0]+pv1_size-1,pv1_center[1]+pv1_size]+V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size+1]+V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size-1]-4.0*V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size]) + d2v2[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size]*(V6[pv1_center[0]+pv1_size+1,pv1_center[1]+pv1_size+1]+V6[pv1_center[0]+pv1_size+1,pv1_center[1]+pv1_size-1]+V6[pv1_center[0]+pv1_size-1,pv1_center[1]+pv1_size+1]-3.0*V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size])
		naux6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size] =      n_func(V6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size],n6[pv1_center[0]+pv1_size,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size])
		#pv1 corner +-
		Vaux6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1] =      V_func_fast(V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1],n6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1], const(I),I) + d2v1[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]+pv1_size+1,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]+pv1_size-1,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1+1]+V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1-1]-4.0*V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1]) + d2v2[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]+pv1_size+1,pv1_center[1]-pv1_size-1+1]+V6[pv1_center[0]+pv1_size-1,pv1_center[1]-pv1_size-1-1]+V6[pv1_center[0]+pv1_size+1,pv1_center[1]-pv1_size-1-1]-3.0*V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1])
		naux6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1] =      n_func(V6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1],n6[pv1_center[0]+pv1_size,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1])
		#pv1 corner --
		Vaux6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1] =      V_func_fast(V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1],n6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1], const(I),I) + d2v1[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]-pv1_size-1+1,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]-pv1_size-1]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1+1]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1-1]-4.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1]) + d2v2[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1]*(V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]-pv1_size-1-1]+V6[pv1_center[0]-pv1_size-1+1,pv1_center[1]-pv1_size-1-1]+V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]-pv1_size-1+1]-3.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1])
		naux6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1] =      n_func(V6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1],n6[pv1_center[0]-pv1_size-1,pv1_center[1]-pv1_size-1],m6[0,pv1_center[1]-pv1_size-1])
		#pv1 corner -+
		Vaux6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size] =      V_func_fast(V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size], const(I),I) + d2v1[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size-1+1,pv1_center[1]+pv1_size]+V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]+pv1_size]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size+1]+V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size-1]-4.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size]) + d2v2[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size]*(V6[pv1_center[0]-pv1_size-1+1,pv1_center[1]+pv1_size+1]+V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]+pv1_size-1]+V6[pv1_center[0]-pv1_size-1-1,pv1_center[1]+pv1_size+1]-3.0*V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size])
		naux6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size] =      n_func(V6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size],n6[pv1_center[0]-pv1_size-1,pv1_center[1]+pv1_size],m6[0,pv1_center[1]+pv1_size])
		
		#pv2

		Vaux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] = 0
		naux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] = 0

		Vaux6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] = V_func_fast(V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],m6[0,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size], const(I),I) + d2v1[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size-2,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1:pv2_center[1]+pv2_size-1]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size+1:pv2_center[1]+pv2_size+1]-3.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]) + d2v2[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1:pv2_center[1]+pv2_size-1]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size+1:pv2_center[1]+pv2_size+1]-2.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size])
		naux6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] =      n_func(V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],m6[0,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size])
		
		#pv2 right wall
		Vaux6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] = V_func_fast(V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],n6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],m6[0,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size], const(I),I) + d2v1[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]*(V6[pv2_center[0]+pv2_size+1,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]+V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size-1:pv2_center[1]+pv2_size-1]+V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size+1:pv2_center[1]+pv2_size+1]-3.0*V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]) + d2v2[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size]*(V6[pv2_center[0] + pv2_size +1,pv2_center[1]-pv2_size-1:pv2_center[1]+pv2_size-1]+V6[pv2_center[0] + pv2_size+1 ,pv2_center[1]-pv2_size+1:pv2_center[1]+pv2_size+1]-2.0*V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size])
		naux6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size] =      n_func(V6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],n6[pv2_center[0] + pv2_size ,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size],m6[0,pv2_center[1]-pv2_size:pv2_center[1]+pv2_size])
				
		#pv2 top wall
		Vaux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1] = V_func_fast(V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1],n6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1], const(I),I) + d2v1[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]-pv2_size+1:pv2_center[0]+pv2_size+1 ,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]-pv2_size-1:pv2_center[0]+pv2_size-1 ,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1-1]-3.0*V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1]) + d2v2[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]-pv2_size+1:pv2_center[0]+pv2_size +1,pv2_center[1]-pv2_size-1-1]+V6[pv2_center[0]-pv2_size-1:pv2_center[0]+pv2_size-1 ,pv2_center[1]-pv2_size-1-1]-2.0*V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1])
		naux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1] =      n_func(V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1],n6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1])
				
		#pv2 bottom wall
		Vaux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size] = V_func_fast(V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size], const(I),I) + d2v1[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size+1:pv2_center[0]+pv2_size+1 ,pv2_center[1]+pv2_size]+V6[pv2_center[0]-pv2_size-1:pv2_center[0]+pv2_size-1 ,pv2_center[1]+pv2_size]+V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size+1]-3.0*V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size]) + d2v2[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size+1:pv2_center[0]+pv2_size +1,pv2_center[1]+pv2_size+1]+V6[pv2_center[0]-pv2_size-1:pv2_center[0]+pv2_size-1 ,pv2_center[1]+pv2_size+1]-2.0*V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size])
		naux6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size] =      n_func(V6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size:pv2_center[0]+pv2_size ,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size])
		
		#pv2 corner ++
		Vaux6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size] =      V_func_fast(V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size],n6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size], const(I),I) + d2v1[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size]*(V6[pv2_center[0]+pv2_size+1,pv2_center[1]+pv2_size]+V6[pv2_center[0]+pv2_size-1,pv2_center[1]+pv2_size]+V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size+1]+V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size-1]-4.0*V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size]) + d2v2[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size]*(V6[pv2_center[0]+pv2_size+1,pv2_center[1]+pv2_size+1]+V6[pv2_center[0]+pv2_size+1,pv2_center[1]+pv2_size-1]+V6[pv2_center[0]+pv2_size-1,pv2_center[1]+pv2_size+1]-3.0*V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size])
		naux6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size] =      n_func(V6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size],n6[pv2_center[0]+pv2_size,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size])
		#pv2 corner +-
		Vaux6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1] =      V_func_fast(V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1],n6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1], const(I),I) + d2v1[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]+pv2_size+1,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]+pv2_size-1,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1+1]+V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1-1]-4.0*V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1]) + d2v2[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]+pv2_size+1,pv2_center[1]-pv2_size-1+1]+V6[pv2_center[0]+pv2_size-1,pv2_center[1]-pv2_size-1-1]+V6[pv2_center[0]+pv2_size+1,pv2_center[1]-pv2_size-1-1]-3.0*V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1])
		naux6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1] =      n_func(V6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1],n6[pv2_center[0]+pv2_size,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1])
		#pv2 corner --
		Vaux6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1] =      V_func_fast(V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1],n6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1], const(I),I) + d2v1[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]-pv2_size-1+1,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]-pv2_size-1]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1+1]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1-1]-4.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1]) + d2v2[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1]*(V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]-pv2_size-1-1]+V6[pv2_center[0]-pv2_size-1+1,pv2_center[1]-pv2_size-1-1]+V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]-pv2_size-1+1]-3.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1])
		naux6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1] =      n_func(V6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1],n6[pv2_center[0]-pv2_size-1,pv2_center[1]-pv2_size-1],m6[0,pv2_center[1]-pv2_size-1])
		#pv2 corner -+
		Vaux6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size] =      V_func_fast(V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size], const(I),I) + d2v1[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size-1+1,pv2_center[1]+pv2_size]+V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]+pv2_size]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size+1]+V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size-1]-4.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size]) + d2v2[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size]*(V6[pv2_center[0]-pv2_size-1+1,pv2_center[1]+pv2_size+1]+V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]+pv2_size-1]+V6[pv2_center[0]-pv2_size-1-1,pv2_center[1]+pv2_size+1]-3.0*V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size])
		naux6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size] =      n_func(V6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size],n6[pv2_center[0]-pv2_size-1,pv2_center[1]+pv2_size],m6[0,pv2_center[1]+pv2_size])
		

		#pv3

		Vaux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] = 0
		naux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] = 0

		Vaux6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] = V_func_fast(V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],m6[0,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size], const(I),I) + d2v1[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size-2,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1:pv3_center[1]+pv3_size-1]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size+1:pv3_center[1]+pv3_size+1]-3.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]) + d2v2[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1:pv3_center[1]+pv3_size-1]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size+1:pv3_center[1]+pv3_size+1]-2.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size])
		naux6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] =      n_func(V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],m6[0,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size])
		
		#pv3 right wall
		Vaux6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] = V_func_fast(V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],n6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],m6[0,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size], const(I),I) + d2v1[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]*(V6[pv3_center[0]+pv3_size+1,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]+V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size-1:pv3_center[1]+pv3_size-1]+V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size+1:pv3_center[1]+pv3_size+1]-3.0*V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]) + d2v2[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size]*(V6[pv3_center[0] + pv3_size +1,pv3_center[1]-pv3_size-1:pv3_center[1]+pv3_size-1]+V6[pv3_center[0] + pv3_size+1 ,pv3_center[1]-pv3_size+1:pv3_center[1]+pv3_size+1]-2.0*V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size])
		naux6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size] =      n_func(V6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],n6[pv3_center[0] + pv3_size ,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size],m6[0,pv3_center[1]-pv3_size:pv3_center[1]+pv3_size])
				
		#pv3 top wall
		Vaux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1] = V_func_fast(V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1],n6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1], const(I),I) + d2v1[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]-pv3_size+1:pv3_center[0]+pv3_size+1 ,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]-pv3_size-1:pv3_center[0]+pv3_size-1 ,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1-1]-3.0*V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1]) + d2v2[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]-pv3_size+1:pv3_center[0]+pv3_size +1,pv3_center[1]-pv3_size-1-1]+V6[pv3_center[0]-pv3_size-1:pv3_center[0]+pv3_size-1 ,pv3_center[1]-pv3_size-1-1]-2.0*V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1])
		naux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1] =      n_func(V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1],n6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1])
				
		#pv3 bottom wall
		Vaux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size] = V_func_fast(V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size], const(I),I) + d2v1[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size+1:pv3_center[0]+pv3_size+1 ,pv3_center[1]+pv3_size]+V6[pv3_center[0]-pv3_size-1:pv3_center[0]+pv3_size-1 ,pv3_center[1]+pv3_size]+V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size+1]-3.0*V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size]) + d2v2[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size+1:pv3_center[0]+pv3_size +1,pv3_center[1]+pv3_size+1]+V6[pv3_center[0]-pv3_size-1:pv3_center[0]+pv3_size-1 ,pv3_center[1]+pv3_size+1]-2.0*V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size])
		naux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size] =      n_func(V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size])
		maux6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size] =      m_func(V6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size:pv3_center[0]+pv3_size ,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size])

		#pv3 corner ++
		Vaux6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size] =      V_func_fast(V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size],n6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size], const(I),I) + d2v1[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size]*(V6[pv3_center[0]+pv3_size+1,pv3_center[1]+pv3_size]+V6[pv3_center[0]+pv3_size-1,pv3_center[1]+pv3_size]+V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size+1]+V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size-1]-4.0*V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size]) + d2v2[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size]*(V6[pv3_center[0]+pv3_size+1,pv3_center[1]+pv3_size+1]+V6[pv3_center[0]+pv3_size+1,pv3_center[1]+pv3_size-1]+V6[pv3_center[0]+pv3_size-1,pv3_center[1]+pv3_size+1]-3.0*V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size])
		naux6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size] =      n_func(V6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size],n6[pv3_center[0]+pv3_size,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size])
		#pv3 corner +-
		Vaux6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1] =      V_func_fast(V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1],n6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1], const(I),I) + d2v1[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]+pv3_size+1,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]+pv3_size-1,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1+1]+V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1-1]-4.0*V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1]) + d2v2[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]+pv3_size+1,pv3_center[1]-pv3_size-1+1]+V6[pv3_center[0]+pv3_size-1,pv3_center[1]-pv3_size-1-1]+V6[pv3_center[0]+pv3_size+1,pv3_center[1]-pv3_size-1-1]-3.0*V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1])
		naux6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1] =      n_func(V6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1],n6[pv3_center[0]+pv3_size,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1])
		#pv3 corner --
		Vaux6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1] =      V_func_fast(V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1],n6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1], const(I),I) + d2v1[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]-pv3_size-1+1,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]-pv3_size-1]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1+1]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1-1]-4.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1]) + d2v2[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1]*(V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]-pv3_size-1-1]+V6[pv3_center[0]-pv3_size-1+1,pv3_center[1]-pv3_size-1-1]+V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]-pv3_size-1+1]-3.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1])
		naux6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1] =      n_func(V6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1],n6[pv3_center[0]-pv3_size-1,pv3_center[1]-pv3_size-1],m6[0,pv3_center[1]-pv3_size-1])
		#pv3 corner -+
		Vaux6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size] =      V_func_fast(V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size], const(I),I) + d2v1[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size-1+1,pv3_center[1]+pv3_size]+V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]+pv3_size]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size+1]+V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size-1]-4.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size]) + d2v2[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size]*(V6[pv3_center[0]-pv3_size-1+1,pv3_center[1]+pv3_size+1]+V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]+pv3_size-1]+V6[pv3_center[0]-pv3_size-1-1,pv3_center[1]+pv3_size+1]-3.0*V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size])
		naux6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size] =      n_func(V6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size],n6[pv3_center[0]-pv3_size-1,pv3_center[1]+pv3_size],m6[0,pv3_center[1]+pv3_size])
		
		
		#pv4

		Vaux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] = 0
		naux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] = 0

		Vaux6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] = V_func_fast(V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],m6[0,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size], const(I),I) + d2v1[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size-2,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1:pv4_center[1]+pv4_size-1]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size+1:pv4_center[1]+pv4_size+1]-3.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]) + d2v2[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1:pv4_center[1]+pv4_size-1]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size+1:pv4_center[1]+pv4_size+1]-2.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size])
		naux6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] =      n_func(V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],m6[0,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size])
		
		#pv4 right wall
		Vaux6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] = V_func_fast(V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],n6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],m6[0,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size], const(I),I) + d2v1[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]*(V6[pv4_center[0]+pv4_size+1,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]+V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size-1:pv4_center[1]+pv4_size-1]+V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size+1:pv4_center[1]+pv4_size+1]-3.0*V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]) + d2v2[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size]*(V6[pv4_center[0] + pv4_size +1,pv4_center[1]-pv4_size-1:pv4_center[1]+pv4_size-1]+V6[pv4_center[0] + pv4_size+1 ,pv4_center[1]-pv4_size+1:pv4_center[1]+pv4_size+1]-2.0*V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size])
		naux6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size] =      n_func(V6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],n6[pv4_center[0] + pv4_size ,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size],m6[0,pv4_center[1]-pv4_size:pv4_center[1]+pv4_size])
				
		#pv4 top wall
		Vaux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1] = V_func_fast(V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1],n6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1], const(I),I) + d2v1[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]-pv4_size+1:pv4_center[0]+pv4_size+1 ,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]-pv4_size-1:pv4_center[0]+pv4_size-1 ,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1-1]-3.0*V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1]) + d2v2[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]-pv4_size+1:pv4_center[0]+pv4_size +1,pv4_center[1]-pv4_size-1-1]+V6[pv4_center[0]-pv4_size-1:pv4_center[0]+pv4_size-1 ,pv4_center[1]-pv4_size-1-1]-2.0*V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1])
		naux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1] =      n_func(V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1],n6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1])
		
		#pv4 bottom wall
		Vaux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size] = V_func_fast(V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size], const(I),I) + d2v1[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size+1:pv4_center[0]+pv4_size+1 ,pv4_center[1]+pv4_size]+V6[pv4_center[0]-pv4_size-1:pv4_center[0]+pv4_size-1 ,pv4_center[1]+pv4_size]+V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size+1]-3.0*V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size]) + d2v2[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size+1:pv4_center[0]+pv4_size +1,pv4_center[1]+pv4_size+1]+V6[pv4_center[0]-pv4_size-1:pv4_center[0]+pv4_size-1 ,pv4_center[1]+pv4_size+1]-2.0*V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size])
		naux6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size] =      n_func(V6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size:pv4_center[0]+pv4_size ,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size])
		
		#pv3 corner ++
		Vaux6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size] =      V_func_fast(V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size],n6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size], const(I),I) + d2v1[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size]*(V6[pv4_center[0]+pv4_size+1,pv4_center[1]+pv4_size]+V6[pv4_center[0]+pv4_size-1,pv4_center[1]+pv4_size]+V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size+1]+V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size-1]-4.0*V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size]) + d2v2[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size]*(V6[pv4_center[0]+pv4_size+1,pv4_center[1]+pv4_size+1]+V6[pv4_center[0]+pv4_size+1,pv4_center[1]+pv4_size-1]+V6[pv4_center[0]+pv4_size-1,pv4_center[1]+pv4_size+1]-3.0*V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size])
		naux6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size] =      n_func(V6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size],n6[pv4_center[0]+pv4_size,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size])
		#pv4 corner +-
		Vaux6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1] =      V_func_fast(V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1],n6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1], const(I),I) + d2v1[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]+pv4_size+1,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]+pv4_size-1,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1+1]+V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1-1]-4.0*V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1]) + d2v2[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]+pv4_size+1,pv4_center[1]-pv4_size-1+1]+V6[pv4_center[0]+pv4_size-1,pv4_center[1]-pv4_size-1-1]+V6[pv4_center[0]+pv4_size+1,pv4_center[1]-pv4_size-1-1]-3.0*V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1])
		naux6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1] =      n_func(V6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1],n6[pv4_center[0]+pv4_size,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1])
		#pv4 corner --
		Vaux6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1] =      V_func_fast(V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1],n6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1], const(I),I) + d2v1[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]-pv4_size-1+1,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]-pv4_size-1]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1+1]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1-1]-4.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1]) + d2v2[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1]*(V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]-pv4_size-1-1]+V6[pv4_center[0]-pv4_size-1+1,pv4_center[1]-pv4_size-1-1]+V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]-pv4_size-1+1]-3.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1])
		naux6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1] =      n_func(V6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1],n6[pv4_center[0]-pv4_size-1,pv4_center[1]-pv4_size-1],m6[0,pv4_center[1]-pv4_size-1])
		#pv4 corner -+
		Vaux6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size] =      V_func_fast(V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size], const(I),I) + d2v1[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size-1+1,pv4_center[1]+pv4_size]+V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]+pv4_size]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size+1]+V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size-1]-4.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size]) + d2v2[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size]*(V6[pv4_center[0]-pv4_size-1+1,pv4_center[1]+pv4_size+1]+V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]+pv4_size-1]+V6[pv4_center[0]-pv4_size-1-1,pv4_center[1]+pv4_size+1]-3.0*V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size])
		naux6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size] =      n_func(V6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size],n6[pv4_center[0]-pv4_size-1,pv4_center[1]+pv4_size],m6[0,pv4_center[1]+pv4_size])
		

		
		Vbundle_aux[0] = V_func_prk(Vbundle[0],nbundle[0],m1[0],0,0) + 1/6*(V5[int(3*ra/4),int(3*ra/4)]-Vbundle[0])
		nbundle_aux[0] = n_func(Vbundle[0],nbundle[0],m1[0])

		Vbundle_aux[1:bndl-1] = V_func_prk(Vbundle[1:bndl-1],nbundle[1:bndl-1],m1[0],const(0),0) + (1/6)*(Vbundle[0:bndl-2]+Vbundle[2:bndl]-2*Vbundle[1:bndl-1])
		nbundle_aux[1:bndl-1] = n_func(Vbundle[1:bndl-1],nbundle[1:bndl-1],m1[0])

		Vbundle_aux[bndl-1] = V_func_prk(Vbundle[bndl-1],nbundle[bndl-1],m1[0],const(0),0) + (1/6)*(Vbundle[bndl-2]-Vbundle[bndl-1])
		nbundle_aux[bndl-1] = n_func(Vbundle[bndl-1],nbundle[bndl-1],m1[0])



		#normal cells
		Vaux1[1:ra-1,1:ra-1] = V_func_slow(V1[1:ra-1,1:ra-1],n1[1:ra-1,1:ra-1],m1[1:ra-1,1:ra-1], c1[1:ra-1,1:ra-1],I) + d1v1[1:ra-1,1:ra-1]*(V1[0:ra-2,1:ra-1]+V1[2:ra,1:ra-1]+V1[1:ra-1,0:ra-2]+V1[1:ra-1,2:ra]-4.0*V1[1:ra-1,1:ra-1]) + d1v2[1:ra-1,1:ra-1]*(V1[0:ra-2,0:ra-2]+V1[0:ra-2,2:ra]+V1[2:ra,0:ra-2]+V1[2:ra,2:ra]-4.0*V1[1:ra-1,1:ra-1])
		naux1[1:ra-1,1:ra-1] = n_func(V1[1:ra-1,1:ra-1],n1[1:ra-1,1:ra-1],m1[1:ra-1,1:ra-1]) 
		
		Vaux1[int(ra/2):ra-1,1:ra-1] = V_func_slow(V1[int(ra/2):ra-1,1:ra-1],n1[int(ra/2):ra-1,1:ra-1],m1[int(ra/2):ra-1,1:ra-1], c1[int(ra/2):ra-1,1:ra-1],I) + d1v1[int(ra/2):ra-1,1:ra-1]*(V1[int(ra/2)-1:ra-2,1:ra-1]+V1[int(ra/2)+1:ra,1:ra-1]+V1[int(ra/2):ra-1,1-1:ra-2]+V1[int(ra/2):ra-1,1+1:ra]+V2[int(ra/2):ra-1,1:ra-1]-5.0*V1[int(ra/2):ra-1,1:ra-1]) + d1v2[int(ra/2):ra-1,1:ra-1]*(V1[int(ra/2)-1:ra-2,1-1:ra-2]+V1[int(ra/2)-1:ra-2,1+1:ra]+V1[int(ra/2)+1:ra,1-1:ra-2]+V1[int(ra/2)+1:ra,1+1:ra]-4.0*V1[int(ra/2):ra-1,1:ra-1])
		naux1[int(ra/2):ra-1,1:ra-1] = n_func(V1[int(ra/2):ra-1,1:ra-1],n1[int(ra/2):ra-1,1:ra-1],m1[int(ra/2):ra-1,1:ra-1]) 
		
		Vaux1[1:ra-1,int(2*ra/3):ra-1] = V_func_slow(V1[1:ra-1,int(2*ra/3):ra-1],n1[1:ra-1,int(2*ra/3):ra-1],m1[1:ra-1,int(2*ra/3):ra-1], c1[1:ra-1,int(2*ra/3):ra-1],I) + d1v1[1:ra-1,int(2*ra/3):ra-1]*(V1[1-1:ra-2,int(2*ra/3):ra-1]+V1[1+1:ra,int(2*ra/3):ra-1]+V1[1:ra-1,int(2*ra/3)-1:ra-2]+V1[1:ra-1,int(2*ra/3)+1:ra]+V2[1:ra-1,int(2*ra/3):ra-1]-5.0*V1[1:ra-1,int(2*ra/3):ra-1]) + d1v2[1:ra-1,int(2*ra/3):ra-1]*(V1[1-1:ra-2,int(2*ra/3)-1:ra-2]+V1[1-1:ra-2,int(2*ra/3)+1:ra]+V1[1+1:ra,int(2*ra/3)-1:ra-2]+V1[1+1:ra,int(2*ra/3)+1:ra]-4.0*V1[1:ra-1,int(2*ra/3):ra-1])
		naux1[1:ra-1,int(2*ra/3):ra-1] = n_func(V1[1:ra-1,int(2*ra/3):ra-1],n1[1:ra-1,int(2*ra/3):ra-1],m1[1:ra-1,int(2*ra/3):ra-1]) 
		
        #left wall
		Vaux1[0,1:ra-1] = V_func_slow(V1[0,1:ra-1],n1[0,1:ra-1],m1[0,1:ra-1], c1[0,1:ra-1],I) + d1v1[0,1:ra-1]*(V1[1,1:ra-1]+V1[0,0:ra-2]+V1[0,2:ra]+V1[0,ra-2:0:-1]+V2[0,1:ra-1]-5.0*V1[0,1:ra-1]) + d1v2[0,1:ra-1]*(V1[1,0:ra-2]+V1[1,2:ra]+V1[0,ra-1:1:-1]+V1[0,ra-3::-1]-4.0*V1[0,1:ra-1])
		naux1[0,1:ra-1] = n_func(V1[0,1:ra-1],n1[0,1:ra-1],m1[0,1:ra-1])
		
		Vaux1[0,1:int(2*ra/3)-1] = V_func_slow(V1[0,1:int(2*ra/3)-1],n1[0,1:int(2*ra/3)-1],m1[0,1:int(2*ra/3)-1], c1[0,1:int(2*ra/3)-1],I) + d1v1[0,1:int(2*ra/3)-1]*(V1[1,1:int(2*ra/3)-1]+V1[0,0:int(2*ra/3)-2]+V1[0,2:int(2*ra/3)]+V1[0,int(2*ra/3)-2:0:-1]-4.0*V1[0,1:int(2*ra/3)-1]) + d1v2[0,1:int(2*ra/3)-1]*(V1[1,0:int(2*ra/3)-2]+V1[1,2:int(2*ra/3)]+V1[0,int(2*ra/3)-1:1:-1]+V1[0,int(2*ra/3)-3::-1]-4.0*V1[0,1:int(2*ra/3)-1])
		naux1[0,1:int(2*ra/3)-1] = n_func(V1[0,1:int(2*ra/3)-1],n1[0,1:int(2*ra/3)-1],m1[0,1:int(2*ra/3)-1])
		
		#right wall
		Vaux1[ra-1,1:ra-1]= V_func_slow(V1[ra-1,1:ra-1],n1[ra-1,1:ra-1],m1[ra-1,1:ra-1], c1[ra-1,1:ra-1],I) + d1v1[ra-1,1:ra-1]*(V1[ra-2,1:ra-1]+V1[ra-1,0:ra-2]+V1[ra-1,2:ra]+V1[ra-1,ra-2:0:-1]+V2[ra-1,1:ra-1]-5.0*V1[ra-1,1:ra-1]) + d1v2[ra-1,1:ra-1]*(V1[ra-2,0:ra-2]+V1[ra-2,2:ra]+V1[ra-1,ra-1:1:-1]+V1[ra-1,ra-3::-1]-4.0*V1[ra-1,1:ra-1])
		naux1[ra-1,1:ra-1]= n_func(V1[ra-1,1:ra-1],n1[ra-1,1:ra-1],m1[ra-1,1:ra-1])
		
		#top wall

		Vaux1[1:ra-1,0]= V_func_slow(V1[1:ra-1,0],n1[1:ra-1,0],m1[1:ra-1,0], c1[1:ra-1,0],I) + d1v1[1:ra-1,0]*(V1[1:ra-1,1]+V1[0:ra-2,0]+V1[2:ra,0]+V1[ra-2:0:-1,0]+V2[1:ra-1,0]-5.0*V1[1:ra-1,0]) + d1v2[1:ra-1,0]*(V1[0:ra-2,1]+V1[2:ra,1]+V1[ra-1:1:-1,0]+V1[ra-3::-1,0]-4.0*V1[1:ra-1,0])
		naux1[1:ra-1,0]= n_func(V1[1:ra-1,0],n1[1:ra-1,0],m1[1:ra-1,0])
		
		Vaux1[1:int(ra/2-mv_lenght/2),0]= V_func_slow(V1[1:int(ra/2-mv_lenght/2),0],n1[1:int(ra/2-mv_lenght/2),0],m1[1:int(ra/2-mv_lenght/2),0], c1[1:int(ra/2-mv_lenght/2),0],I) + d1v1[1:int(ra/2-mv_lenght/2),0]*(V1[1:int(ra/2-mv_lenght/2),1]+V1[0:int(ra/2-mv_lenght/2)-1,0]+V1[2:int(ra/2-mv_lenght/2)+1,0]+V1[ra-2:int(ra/2+mv_lenght/2)-1:-1,0]-4.0*V1[1:int(ra/2-mv_lenght/2),0]) + d1v2[1:int(ra/2-mv_lenght/2),0]*(V1[0:int(ra/2-mv_lenght/2)-1,1]+V1[2:int(ra/2-mv_lenght/2)+1,1]+V1[ra-1:int(ra/2+mv_lenght/2):-1,0]+V1[ra-3:int(ra/2+mv_lenght/2)-2:-1,0]-4.0*V1[1:int(ra/2-mv_lenght/2),0])
		naux1[1:int(ra/2-mv_lenght/2),0]=      n_func(V1[1:int(ra/2-mv_lenght/2),0],n1[1:int(ra/2-mv_lenght/2),0],m1[1:int(ra/2-mv_lenght/2),0])
		
		
		#bottora wall
		Vaux1[1:ra-1,ra-1]= V_func_slow(V1[1:ra-1,ra-1],n1[1:ra-1,ra-1],m1[1:ra-1,ra-1], c1[1:ra-1,ra-1],I) + d1v1[1:ra-1,ra-1]*(V1[1:ra-1,ra-2]+V1[0:ra-2,ra-1]+V1[2:ra,ra-1]+V1[ra-2:0:-1,ra-1]+V2[1:ra-1,ra-1]-5.0*V1[1:ra-1,ra-1]) + d1v2[1:ra-1,ra-1]*(V1[0:ra-2,ra-2]+V1[2:ra,ra-2]+V1[ra-1:1:-1,ra-1]+V1[ra-3::-1,ra-1]-4.0*V1[1:ra-1,ra-1])
		naux1[1:ra-1,ra-1]= n_func(V1[1:ra-1,ra-1],n1[1:ra-1,ra-1],m1[1:ra-1,ra-1])
		
		Vaux1[0,0]= V_func_slow(V1[0,0],n1[0,0],m1[0,0], c1[0,0],I) + d1v1[0,0]*(V1[1,0]+V1[0,1]+V1[ra-1,0]+V1[0,ra-1]+V2[0,0]-5.0*V1[0,0]) + d1v2[0,0]*(V1[1,1]+V1[ra-1,ra-1]+V1[0,ra-2]+V1[ra-2,0]-4*V1[0,0])
		naux1[0,0]= n_func(V1[0,0],n1[0,0],m1[0,0])
		
		Vaux1[0,ra-1]= V_func_slow(V1[0,ra-1],n1[0,ra-1],m1[0,ra-1], c1[0,ra-1],I) + d1v1[0,ra-1]*(V1[1,ra-1]+V1[0,ra-2]+V1[0,0]+V1[ra-1,ra-1]+V2[0,ra-1]-5.0*V1[0,ra-1]) + d1v2[0,ra-1]*(V1[1,ra-2]+V1[1,0]+V1[0,1]+V1[ra-1,0]-4*V1[0,ra-1])
		naux1[0,ra-1]= n_func(V1[0,ra-1],n1[0,ra-1],m1[0,ra-1])
		
		Vaux1[ra-1,0]= V_func_slow(V1[ra-1,0],n1[ra-1,0],m1[ra-1,0], c1[ra-1,0],I) + d1v1[ra-1,0]*(V1[ra-1,1]+V1[ra-2,0]+V1[0,0]+V1[ra-1,ra-1]+V2[ra-1,0]-5.0*V1[ra-1,0]) + d1v2[ra-1,0]*(V1[ra-2,1]+V1[1,0]+V1[0,1]+V1[0,ra-1]-4*V1[ra-1,0])
		naux1[ra-1,0]= n_func(V1[ra-1,0],n1[ra-1,0],m1[ra-1,0])
		
		Vaux1[ra-1,ra-1]= V_func_slow(V1[ra-1,ra-1],n1[ra-1,ra-1],m1[ra-1,ra-1], c1[ra-1,ra-1],I) + d1v1[ra-1,ra-1]*(V1[ra-2,ra-1]+V1[ra-1,ra-2]+V1[ra-1,0]+V1[0,ra-1]+V2[ra-1,ra-1]-5.0*V1[ra-1,ra-1]) + d1v2[ra-1,ra-1]*(V1[ra-2,ra-2]+V1[1,ra-1]+V1[ra-1,1]+V1[0,0]-4*V1[ra-1,ra-1])
		naux1[ra-1,ra-1]= n_func(V1[ra-1,ra-1],n1[ra-1,ra-1],m1[ra-1,ra-1])
		
		Vaux1[int(ra/4),int(ra/4)]= V_func_slow(V1[int(ra/4),int(ra/4)],n1[int(ra/4),int(ra/4)],m1[int(ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-30]-V1[int(ra/4),int(ra/4)])
		naux1[int(ra/4),int(ra/4)]=      n_func(V1[int(ra/4),int(ra/4)],n1[int(ra/4),int(ra/4)],m1[int(ra/4),int(ra/4)])
		
		###WOLF PARK
		#Vaux1[int(3*ra/4),int(ra/4)]= V_func_slow(V1[int(3*ra/4),int(ra/4)],n1[int(3*ra/4),int(ra/4)],m3[int(3*ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-60]-V1[int(3*ra/4),int(ra/4)])
		#naux1[int(3*ra/4),int(ra/4)]= n_func(V1[int(3*ra/4),int(ra/4)],n1[int(3*ra/4),int(ra/4)],m3[int(3*ra/4),int(ra/4)])

		Vaux1[int(ra/2),ra-1]= V_func_slow(V1[int(ra/2),ra-1],n1[int(ra/2),ra-1],m1[int(ra/2),ra-1], const(I),I) + 1/6*(Vbundle[-10]-V1[int(ra/2),ra-1])#1/6*(V1[int(ra/4),int(ra/4)+7]-V1[int(ra/2),ra-1])
		naux1[int(ra/2),ra-1]= n_func(V1[int(ra/2),ra-1],n1[int(ra/2),ra-1],m1[int(ra/2),ra-1])

		Vaux1[int(3*ra/4),int(ra/4)]= V_func_slow(V1[int(3*ra/4),int(ra/4)],n1[int(3*ra/4),int(ra/4)],m1[int(3*ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-1]-V1[int(3*ra/4),int(ra/4)])#1/6*(V1[int(ra/4),int(ra/4)+7]-V1[int(3*ra/4),int(ra/4)])
		naux1[int(3*ra/4),int(ra/4)]= n_func(V1[int(3*ra/4),int(ra/4)],n1[int(3*ra/4),int(ra/4)],m1[int(3*ra/4),int(ra/4)])
		
		
		#mitral valve
		Vaux1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0
		naux1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0

		Vaux1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]= V_func_slow(V1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],n1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],m3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],I) + d1v1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]*(V1[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,mv_height]+V1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height+1]+V1[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,mv_height]-3.0*V1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]) + d1v2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]*(V1[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,mv_height+1]+V1[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,mv_height+1]-2.0*V1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height])
		naux1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]= n_func(V1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],n1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],m3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height])
		
		Vaux1[int(ra/2+mv_lenght/2),1:mv_height]= V_func_slow(V1[int(ra/2+mv_lenght/2),1:mv_height],n1[int(ra/2+mv_lenght/2),1:mv_height],m3[int(ra/2+mv_lenght/2),1:mv_height], c1[ra-1,0],I) + d1v1[int(ra/2+mv_lenght/2),1:mv_height]*(V1[int(ra/2+mv_lenght/2)+1,1:mv_height]+V1[int(ra/2+mv_lenght/2)+1,0:mv_height-1]+V1[int(ra/2+mv_lenght/2)+1,2:mv_height+1]-3.0*V1[int(ra/2+mv_lenght/2),1:mv_height]) + d1v2[int(ra/2+mv_lenght/2),1:mv_height]*(V1[int(ra/2+mv_lenght/2)+1,2:mv_height+1]+V1[int(ra/2+mv_lenght/2)+1,0:mv_height-1]-2*V1[int(ra/2+mv_lenght/2),1:mv_height])
		naux1[int(ra/2+mv_lenght/2),1:mv_height]=      n_func(V1[int(ra/2+mv_lenght/2),1:mv_height],n1[int(ra/2+mv_lenght/2),1:mv_height],m3[int(ra/2+mv_lenght/2),1:mv_height])
		
		Vaux1[int(ra/2-mv_lenght/2)-1,1:mv_height]= V_func_slow(V1[int(ra/2-mv_lenght/2)-1,1:mv_height],n1[int(ra/2-mv_lenght/2)-1,1:mv_height],m3[int(ra/2-mv_lenght/2)-1,1:mv_height], c1[ra-1,0],I) + d1v1[int(ra/2-mv_lenght/2)-1,1:mv_height]*(V1[int(ra/2-mv_lenght/2)-2,1:mv_height]+V1[int(ra/2-mv_lenght/2)-1,0:mv_height-1]+V1[int(ra/2-mv_lenght/2)-1,2:mv_height+1]-3.0*V1[int(ra/2-mv_lenght/2)-1,1:mv_height]) + d1v2[int(ra/2-mv_lenght/2)-1,1:mv_height]*(V1[int(ra/2-mv_lenght/2)-1-1,2:mv_height+1]+V1[int(ra/2-mv_lenght/2)-1-1,0:mv_height-1]-2*V1[int(ra/2-mv_lenght/2)-1,1:mv_height])
		naux1[int(ra/2-mv_lenght/2)-1,1:mv_height]=      n_func(V1[int(ra/2-mv_lenght/2)-1,1:mv_height],n1[int(ra/2-mv_lenght/2)-1,1:mv_height],m3[int(ra/2-mv_lenght/2)-1,1:mv_height])
		
		Vaux1[int(ra/2+mv_lenght/2),0]= V_func_slow(V1[int(ra/2+mv_lenght/2),0],n1[int(ra/2+mv_lenght/2),0],m3[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d1v1[int(ra/2+mv_lenght/2),0]*(V1[int(ra/2+mv_lenght/2),1]+V1[int(ra/2-mv_lenght/2)-1,0]+V1[int(ra/2+mv_lenght/2)+1,0]-3.0*V1[int(ra/2+mv_lenght/2),0]) + d1v2[int(ra/2+mv_lenght/2),0]*(V1[int(ra/2+mv_lenght/2)+1,1]+V1[int(ra/2-mv_lenght/2)-1,0]-2*V1[int(ra/2+mv_lenght/2),0])
		naux1[int(ra/2+mv_lenght/2),0]= n_func(V1[int(ra/2+mv_lenght/2),0],n1[int(ra/2+mv_lenght/2),0],m3[int(ra/2+mv_lenght/2),0])
		
		Vaux1[int(ra/2-mv_lenght/2)-1,0]= V_func_slow(V1[int(ra/2-mv_lenght/2)-1,0],n1[int(ra/2-mv_lenght/2)-1,0],m3[int(ra/2-mv_lenght/2)-1,0], c1[int(ra/2-mv_lenght/2)-1,0],I) + d1v1[int(ra/2-mv_lenght/2)-1,0]*(V1[int(ra/2-mv_lenght/2)-1,1]+V1[int(ra/2-mv_lenght/2)-1,0]+V1[int(ra/2+mv_lenght/2),0]-3.0*V1[int(ra/2-mv_lenght/2)-1,0]) + d1v2[int(ra/2-mv_lenght/2)-1,0]*(V1[int(ra/2+mv_lenght/2),1]+V1[int(ra/2-mv_lenght/2)-1,0]-2*V1[int(ra/2-mv_lenght/2)-1,0])
		naux1[int(ra/2-mv_lenght/2)-1,0]= n_func(V1[int(ra/2-mv_lenght/2)-1,0],n1[int(ra/2-mv_lenght/2)-1,0],m3[int(ra/2-mv_lenght/2)-1,0])
		
		Vaux1[0,1:int(ra/2)] = V_func_slow(V1[0,1:int(ra/2)],n1[0,1:int(ra/2)],m1[0,1:int(ra/2)], c1[0,1:int(ra/2)],I) + d1v1[0,1:int(ra/2)]*(V1[1,1:int(ra/2)]+V1[0,0:int(ra/2)-1]+V1[0,2:int(ra/2)+1]+V1[0,ra-1:int(ra/2):-1]-4.0*V1[0,1:int(ra/2)]) #+ d1v2[0,1:int(ra/2)]*(V1[1,0:int(ra/2)-1]+V1[1,2:int(ra/2)+1]+V1[0,int(ra/2):1:-1]+V1[0,ra-3::-1]-4.0*V1[0,1:int(ra/2)])
		naux1[0,1:int(ra/2)] = n_func(V1[0,1:int(ra/2)],n1[0,1:int(ra/2)],m1[0,1:int(ra/2)])

		Vaux2[int(ra/2):ra-1,1:ra-1] = V_func_fast(V2[int(ra/2):ra-1,1:ra-1],n2[int(ra/2):ra-1,1:ra-1],m4[int(ra/2):ra-1,1:ra-1], c1[int(ra/2):ra-1,1:ra-1],I) + d1v1[int(ra/2):ra-1,1:ra-1]*(V2[int(ra/2)-1:ra-2,1:ra-1]+V2[int(ra/2)+1:ra,1:ra-1]+V2[int(ra/2):ra-1,1-1:ra-2]+V2[int(ra/2):ra-1,1+1:ra]+V1[int(ra/2):ra-1,1:ra-1]-5.0*V2[int(ra/2):ra-1,1:ra-1]) + d1v2[int(ra/2):ra-1,1:ra-1]*(V2[int(ra/2)-1:ra-2,1-1:ra-2]+V2[int(ra/2)-1:ra-2,1+1:ra]+V2[int(ra/2)+1:ra,1-1:ra-2]+V2[int(ra/2)+1:ra,1+1:ra]-4.0*V2[int(ra/2):ra-1,1:ra-1])
		naux2[int(ra/2):ra-1,1:ra-1] = n_func_ventricle(V2[int(ra/2):ra-1,1:ra-1],n2[int(ra/2):ra-1,1:ra-1],m4[int(ra/2):ra-1,1:ra-1]) 
		
		Vaux2[1:int(ra/2)+1-1,int(2*ra/3):ra-1] = V_func_fast(V2[1:int(ra/2)+1-1,int(2*ra/3):ra-1],n2[1:int(ra/2)+1-1,int(2*ra/3):ra-1],m4[1:int(ra/2)+1-1,int(2*ra/3):ra-1], c1[1:int(ra/2)+1-1,int(2*ra/3):ra-1],I) + d1v1[1:int(ra/2)+1-1,int(2*ra/3):ra-1]*(V2[1-1:int(ra/2)+1-2,int(2*ra/3):ra-1]+V2[1+1:int(ra/2)+1,int(2*ra/3):ra-1]+V2[1:int(ra/2)+1-1,int(2*ra/3)-1:ra-2]+V2[1:int(ra/2)+1-1,int(2*ra/3)+1:ra]+V1[1:int(ra/2)+1-1,int(2*ra/3):ra-1]-5.0*V2[1:int(ra/2)+1-1,int(2*ra/3):ra-1]) + d1v2[1:int(ra/2)+1-1,int(2*ra/3):ra-1]*(V2[1-1:int(ra/2)+1-2,int(2*ra/3)-1:ra-2]+V2[1-1:int(ra/2)+1-2,int(2*ra/3)+1:ra]+V2[1+1:int(ra/2)+1,int(2*ra/3)-1:ra-2]+V2[1+1:int(ra/2)+1,int(2*ra/3)+1:ra]-4.0*V2[1:int(ra/2)+1-1,int(2*ra/3):ra-1])
		naux2[1:int(ra/2)+1-1,int(2*ra/3):ra-1] = n_func_ventricle(V2[1:int(ra/2)+1-1,int(2*ra/3):ra-1],n2[1:int(ra/2)+1-1,int(2*ra/3):ra-1],m4[1:int(ra/2)+1-1,int(2*ra/3):ra-1]) 
		
        #left wall
		Vaux2[0,int(2*ra/3):ra-1] = V_func_fast(V2[0,int(2*ra/3):ra-1],n2[0,int(2*ra/3):ra-1],m[0,int(2*ra/3):ra-1], const(I),I) + gx1[0,int(2*ra/3):ra-1]*(V2[1,int(2*ra/3):ra-1]+V2[0,int(2*ra/3)-1:ra-2]+V2[0,int(2*ra/3)+1:ra]+V2[0,ra-2:int(2*ra/3)-1:-1]+V1[0,int(2*ra/3):ra-1]-5.0*V2[0,int(2*ra/3):ra-1]) + gx2[0,int(2*ra/3):ra-1]*(V2[1,int(2*ra/3)-1:ra-2]+V2[1,int(2*ra/3)+1:ra]+V2[0,ra-1:int(2*ra/3):-1]+V2[0,ra-3:int(2*ra/3)-2:-1]-4.0*V2[0,int(2*ra/3):ra-1])
		naux2[0,int(2*ra/3):ra-1] = n_func_ventricle(V2[0,int(2*ra/3):ra-1],n2[0,int(2*ra/3):ra-1],m[0,int(2*ra/3):ra-1])
		
		#right wall
		Vaux2[ra-1,1:ra-1]= V_func_fast(V2[ra-1,1:ra-1],n2[ra-1,1:ra-1],m[ra-1,1:ra-1], const(I),I) + gx1[ra-1,1:ra-1]*(V2[ra-2,1:ra-1]+V2[ra-1,0:ra-2]+V2[ra-1,2:ra]+V2[ra-1,ra-2:0:-1]-4.0*V2[ra-1,1:ra-1]) + gx2[ra-1,1:ra-1]*(V2[ra-2,0:ra-2]+V2[ra-2,2:ra]+V2[ra-1,ra-1:1:-1]+V2[ra-1,ra-3::-1]-4.0*V2[ra-1,1:ra-1])
		naux2[ra-1,1:ra-1]= n_func_ventricle(V2[ra-1,1:ra-1],n2[ra-1,1:ra-1],m[ra-1,1:ra-1])
		
		#top wall
		Vaux2[int(ra/2):ra-1,0]= V_func_fast(V2[int(ra/2):ra-1,0],n2[int(ra/2):ra-1,0],m[int(ra/2):ra-1,0], const(I),I) + gx1[int(ra/2):ra-1,0]*(V2[int(ra/2):ra-1,1]+V2[int(ra/2)-1:ra-2,0]+V2[int(ra/2)+1:ra,0]-3.0*V2[int(ra/2):ra-1,0]) + gx2[int(ra/2):ra-1,0]*(V2[int(ra/2)-1:ra-2,1]+V2[int(ra/2)+1:ra,1]-2.0*V2[int(ra/2):ra-1,0])
		naux2[int(ra/2):ra-1,0]= n_func_ventricle(V2[int(ra/2):ra-1,0],n2[int(ra/2):ra-1,0],m[int(ra/2):ra-1,0])
		

		#bottora wall
		Vaux2[1:ra-1,ra-1]= V_func_fast(V2[1:ra-1,ra-1],n2[1:ra-1,ra-1],m[1:ra-1,ra-1], const(I),I) + gx1[1:ra-1,ra-1]*(V2[1:ra-1,ra-2]+V2[0:ra-2,ra-1]+V2[2:ra,ra-1]+V2[ra-2:0:-1,ra-1]-4.0*V2[1:ra-1,ra-1]) + gx2[1:ra-1,ra-1]*(V2[0:ra-2,ra-2]+V2[2:ra,ra-2]+V2[ra-1:1:-1,ra-1]+V2[ra-3::-1,ra-1]-4.0*V2[1:ra-1,ra-1])
		naux2[1:ra-1,ra-1]= n_func_ventricle(V2[1:ra-1,ra-1],n2[1:ra-1,ra-1],m[1:ra-1,ra-1])
		
		Vaux2[int(ra/2),0]= V_func_fast(V2[int(ra/2),0],n2[int(ra/2),0],m[int(ra/2),0], const(I),I) + gx1[int(ra/2),0]*(V2[int(ra/2)+1,0]+V2[int(ra/2),1]+V2[ra-int(ra/2),0]+V1[int(ra/2),0]-4.0*V2[int(ra/2),0]) + gx2[int(ra/2),0]*(V2[int(ra/2)+1,int(ra/2)+1]-V2[int(ra/2),0])
		naux2[int(ra/2),0]= n_func_ventricle(V2[int(ra/2),0],n2[int(ra/2),0],m[int(ra/2),0])
		
		Vaux2[0,ra-1]= V_func_fast(V2[0,ra-1],n2[0,ra-1],m[0,ra-1], const(I),I) + gx1[0,ra-1]*(V2[1,ra-1]+V2[0,ra-2]+V2[ra-1,ra-1]+V1[0,ra-1]-4.0*V2[0,ra-1]) + gx2[0,ra-1]*(V2[1,ra-2]+V2[ra-1,0]-2*V2[0,ra-1])
		naux2[0,ra-1]= n_func_ventricle(V2[0,ra-1],n2[0,ra-1],m[0,ra-1])
		
		Vaux2[ra-1,0]= V_func_fast(V2[ra-1,0],n2[ra-1,0],m[ra-1,0], const(I),I) + gx1[ra-1,0]*(V2[ra-1,1]+V2[ra-2,0]+V2[ra-1,ra-1]+V1[ra-1,0]-4.0*V2[ra-1,0]) + gx2[ra-1,0]*(V2[ra-2,1]+V2[0,ra-1]-2*V2[ra-1,0])
		naux2[ra-1,0]= n_func_ventricle(V2[ra-1,0],n2[ra-1,0],m[ra-1,0])
		
		Vaux2[ra-1,ra-1]= V_func_fast(V2[ra-1,ra-1],n2[ra-1,ra-1],m[ra-1,ra-1], const(I),I) + gx1[ra-1,ra-1]*(V2[ra-2,ra-1]+V2[ra-1,ra-2]+V2[ra-1,0]+V2[0,ra-1]+V1[ra-1,ra-1]-5.0*V2[ra-1,ra-1]) + gx2[ra-1,ra-1]*(V2[ra-2,ra-2]+V2[1,ra-1]+V2[ra-1,1]-3*V2[ra-1,ra-1])
		naux2[ra-1,ra-1]= n_func_ventricle(V2[ra-1,ra-1],n2[ra-1,ra-1],m[ra-1,ra-1])
		
		Vaux2[int(ra/2),1:int(2*ra/3)] =V_func_fast(V2[int(ra/2),1:int(2*ra/3)],n2[int(ra/2),1:int(2*ra/3)],m4[int(ra/2),1:int(2*ra/3)], c1[int(ra/2),1:int(2*ra/3)],I) + d1v1[int(ra/2),1:int(2*ra/3)]*(V2[int(ra/2)+1,1:int(2*ra/3)]+V2[int(ra/2),0:int(2*ra/3)-1]+V2[int(ra/2),1:int(2*ra/3)]-3.0*V2[int(ra/2),1:int(2*ra/3)]) + d1v2[int(ra/2),1:int(2*ra/3)]*(V2[int(ra/2)+1,0:int(2*ra/3)-1]+V2[int(ra/2)+1,1+1:int(2*ra/3)+1]-2.0*V2[int(ra/2),1:int(2*ra/3)])
		naux2[int(ra/2),1:int(2*ra/3)] =           n_func_ventricle(V2[int(ra/2),1:int(2*ra/3)],n2[int(ra/2),1:int(2*ra/3)],m4[int(ra/2),1:int(2*ra/3)]) 
		
		Vaux2[1:int(ra/2),int(2*ra/3)] = V_func_fast(V2[1:int(ra/2),int(2*ra/3)],n2[1:int(ra/2),int(2*ra/3)],m4[1:int(ra/2),int(2*ra/3)], c1[1:int(ra/2),int(2*ra/3)],I) + d1v1[1:int(ra/2),int(2*ra/3)]*(V2[0:int(ra/2)-1,int(2*ra/3)]+V2[2:int(ra/2)+1,int(2*ra/3)]+V2[1:int(ra/2),int(2*ra/3)+1]+V1[1:int(ra/2),int(2*ra/3)]-4.0*V2[1:int(ra/2),int(2*ra/3)]) + d1v2[1:int(ra/2),int(2*ra/3)]*(V2[0:int(ra/2)-1,int(2*ra/3)+1]+V2[1+1:int(ra/2)+1,int(2*ra/3)+1]-2.0*V2[1:int(ra/2),int(2*ra/3)])
		naux2[1:int(ra/2),int(2*ra/3)] = n_func_ventricle(V2[1:int(ra/2),int(2*ra/3)],n2[1:int(ra/2),int(2*ra/3)],m4[1:int(ra/2),int(2*ra/3)]) 
		
		
		
		

		Vaux2[int(ra/2):int(ra/2+mv_lenght/2),mv_height]= V_func_fast(V2[int(ra/2):int(ra/2+mv_lenght/2),mv_height],n2[int(ra/2):int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2):int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2):int(ra/2+mv_lenght/2),mv_height],I) + d1v1[int(ra/2):int(ra/2+mv_lenght/2),mv_height]*(V2[int(ra/2)+1:int(ra/2+mv_lenght/2)+1,mv_height]+V2[int(ra/2):int(ra/2+mv_lenght/2),mv_height+1]+V2[int(ra/2)-1:int(ra/2+mv_lenght/2)-1,mv_height]-3.0*V2[int(ra/2):int(ra/2+mv_lenght/2),mv_height]) + d1v2[int(ra/2):int(ra/2+mv_lenght/2),mv_height]*(V2[int(ra/2)-1:int(ra/2+mv_lenght/2)-1,mv_height+1]+V2[int(ra/2)+1:int(ra/2+mv_lenght/2)+1,mv_height+1]-2.0*V2[int(ra/2):int(ra/2+mv_lenght/2),mv_height])
		naux2[int(ra/2):int(ra/2+mv_lenght/2),mv_height]= n_func(V2[int(ra/2):int(ra/2+mv_lenght/2),mv_height],n2[int(ra/2):int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2):int(ra/2+mv_lenght/2),mv_height])
		
		Vaux2[int(ra/2+mv_lenght/2),1:mv_height]= V_func_fast(V2[int(ra/2+mv_lenght/2),1:mv_height],n2[int(ra/2+mv_lenght/2),1:mv_height],m4[int(ra/2+mv_lenght/2),1:mv_height], c1[ra-1,0],I) + d1v1[int(ra/2+mv_lenght/2),1:mv_height]*(V2[int(ra/2+mv_lenght/2)+1,1:mv_height]+V2[int(ra/2+mv_lenght/2)+1,0:mv_height-1]+V2[int(ra/2+mv_lenght/2)+1,2:mv_height+1]-3.0*V2[int(ra/2+mv_lenght/2),1:mv_height]) + d1v2[int(ra/2+mv_lenght/2),1:mv_height]*(V2[int(ra/2+mv_lenght/2)+1,2:mv_height+1]+V2[int(ra/2+mv_lenght/2)+1,0:mv_height-1]-2*V2[int(ra/2+mv_lenght/2),1:mv_height])
		naux2[int(ra/2+mv_lenght/2),1:mv_height]=      n_func(V2[int(ra/2+mv_lenght/2),1:mv_height],n2[int(ra/2+mv_lenght/2),1:mv_height],m4[int(ra/2+mv_lenght/2),1:mv_height])
		
		
		Vaux2[int(ra/2+mv_lenght/2),0]= V_func_fast(V2[int(ra/2+mv_lenght/2),0],n2[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d1v1[int(ra/2+mv_lenght/2),0]*(V2[int(ra/2+mv_lenght/2),1]+V2[int(ra/2)-1-1,0]+V2[int(ra/2+mv_lenght/2)+1,0]-3.0*V2[int(ra/2+mv_lenght/2),0]) + d1v2[int(ra/2+mv_lenght/2),0]*(V2[int(ra/2+mv_lenght/2)+1,1]+V2[int(ra/2)-1-1,0]-2*V2[int(ra/2+mv_lenght/2),0])
		naux2[int(ra/2+mv_lenght/2),0]= n_func(V2[int(ra/2+mv_lenght/2),0],n2[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0])
		
		
		Vaux2[int(ra/2),mv_height]=      V_func_fast(V2[int(ra/2),mv_height],n2[int(ra/2),mv_height],m[int(ra/2),mv_height], const(I),I) + d1v1[int(ra/2),mv_height]*(V2[int(ra/2)+1,mv_height]+V2[int(ra/2),mv_height+1]+V1[int(ra/2),mv_height]-3.0*V2[int(ra/2),mv_height]) + d1v2[int(ra/2)+1,mv_height+1]*(V2[int(ra/2)+1,mv_height+1]-V2[int(ra/2),mv_height])
		naux2[int(ra/2),mv_height]= n_func_ventricle(V2[int(ra/2),mv_height],n2[int(ra/2),mv_height],m[int(ra/2),mv_height])
		
		Vaux2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0
		naux2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0

		Vaux2[int(ra/2),int(2*ra/3)] = V_func_fast(V2[int(ra/2),int(2*ra/3)],n2[int(ra/2),int(2*ra/3)],m4[int(ra/2),int(2*ra/3)], c1[int(ra/2),int(2*ra/3)],I) + d1v1[int(ra/2),int(2*ra/3)]*(V2[int(ra/2),int(2*ra/3)+1]+V2[int(ra/2)+1,int(2*ra/3)]+V2[int(ra/2)-1,int(2*ra/3)]+V2[int(ra/2),int(2*ra/3)-1]+V1[int(ra/2),int(2*ra/3)]-5.0*V2[int(ra/2),int(2*ra/3)]) + d1v2[int(ra/2),int(2*ra/3)]*(V2[int(ra/2)+1,int(2*ra/3)+1]+V2[int(ra/2)-1,int(2*ra/3)+1]+V2[int(ra/2)-1,int(2*ra/3)-1]-3.0*V2[int(ra/2),int(2*ra/3)])
		naux2[int(ra/2),int(2*ra/3)] = n_func_ventricle(V2[int(ra/2),int(2*ra/3)],n2[int(ra/2),int(2*ra/3)],m4[int(ra/2),int(2*ra/3)]) 


		Vaux2[0,int(2*ra/3)] = V_func_fast(V2[0,int(2*ra/3)],n2[0,int(2*ra/3)],m4[0,int(2*ra/3)], c1[0,int(2*ra/3)],I) + d1v1[0,int(2*ra/3)]*(V2[0,int(2*ra/3)+1]+V2[0+1,int(2*ra/3)]+V1[0,int(2*ra/3)]-3.0*V2[0,int(2*ra/3)]) + d1v2[0,int(2*ra/3)]*(V2[1,int(2*ra/3)+1]-V2[0,int(2*ra/3)])
		naux2[0,int(2*ra/3)] = n_func_ventricle(V2[0,int(2*ra/3)],n2[0,int(2*ra/3)],m4[0,int(2*ra/3)]) 

		Vaux2[int(ra/2+mv_lenght/2),mv_height] = V_func_fast(V2[int(ra/2+mv_lenght/2),mv_height],n2[int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2+mv_lenght/2),mv_height],I) + d1v1[int(ra/2+mv_lenght/2),mv_height]*(V2[int(ra/2+mv_lenght/2),mv_height+1]+V2[int(ra/2+mv_lenght/2)+1,mv_height]+V2[int(ra/2+mv_lenght/2)-1,mv_height]+V2[int(ra/2+mv_lenght/2),mv_height-1]+V1[int(ra/2+mv_lenght/2),mv_height]-5.0*V2[int(ra/2+mv_lenght/2),mv_height]) + d1v2[int(ra/2+mv_lenght/2),mv_height]*(V2[int(ra/2+mv_lenght/2)+1,mv_height+1]+V2[int(ra/2+mv_lenght/2)-1,mv_height+1]+V2[int(ra/2+mv_lenght/2)-1,mv_height-1]-3.0*V2[int(ra/2+mv_lenght/2),mv_height])
		naux2[int(ra/2+mv_lenght/2),mv_height] = n_func_ventricle(V2[int(ra/2+mv_lenght/2),mv_height],n2[int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2+mv_lenght/2),mv_height]) 

		Vaux2[int(ra/2+mv_lenght/2),0] = V_func_fast(V2[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d1v1[int(ra/2+mv_lenght/2),0]*(V2[int(ra/2+mv_lenght/2),0+1]+V2[int(ra/2+mv_lenght/2)+1,0]-2.0*V2[int(ra/2+mv_lenght/2),0]) + d1v2[int(ra/2+mv_lenght/2),0]*(V2[int(ra/2+mv_lenght/2)+1,1]-V2[int(ra/2+mv_lenght/2),0])
		naux2[int(ra/2+mv_lenght/2),0] = n_func_ventricle(V2[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0]) 

		#normal cells
		Vaux3[1:ra-1,1:ra-1] = V_func_slow(V3[1:ra-1,1:ra-1],n3[1:ra-1,1:ra-1],m3[1:ra-1,1:ra-1], c1[1:ra-1,1:ra-1],I) + d4v1[1:ra-1,1:ra-1]*(V3[0:ra-2,1:ra-1]+V3[2:ra,1:ra-1]+V3[1:ra-1,0:ra-2]+V3[1:ra-1,2:ra]-4.0*V3[1:ra-1,1:ra-1]) + d4v2[1:ra-1,1:ra-1]*(V3[0:ra-2,0:ra-2]+V3[0:ra-2,2:ra]+V3[2:ra,0:ra-2]+V3[2:ra,2:ra]-4.0*V3[1:ra-1,1:ra-1])
		naux3[1:ra-1,1:ra-1] = n_func(V3[1:ra-1,1:ra-1],n3[1:ra-1,1:ra-1],m3[1:ra-1,1:ra-1]) 
		
		Vaux3[int(ra/2):ra-1,1:ra-1] = V_func_slow(V3[int(ra/2):ra-1,1:ra-1],n3[int(ra/2):ra-1,1:ra-1],m3[int(ra/2):ra-1,1:ra-1], c1[int(ra/2):ra-1,1:ra-1],I) + d4v1[int(ra/2):ra-1,1:ra-1]*(V3[int(ra/2)-1:ra-2,1:ra-1]+V3[int(ra/2)+1:ra,1:ra-1]+V3[int(ra/2):ra-1,1-1:ra-2]+V3[int(ra/2):ra-1,1+1:ra]+V4[int(ra/2):ra-1,1:ra-1]-5.0*V3[int(ra/2):ra-1,1:ra-1]) + d4v2[int(ra/2):ra-1,1:ra-1]*(V3[int(ra/2)-1:ra-2,1-1:ra-2]+V3[int(ra/2)-1:ra-2,1+1:ra]+V3[int(ra/2)+1:ra,1-1:ra-2]+V3[int(ra/2)+1:ra,1+1:ra]-4.0*V3[int(ra/2):ra-1,1:ra-1])
		naux3[int(ra/2):ra-1,1:ra-1] = n_func(V3[int(ra/2):ra-1,1:ra-1],n3[int(ra/2):ra-1,1:ra-1],m3[int(ra/2):ra-1,1:ra-1]) 
		
		Vaux3[1:ra-1,int(2*ra/3):ra-1] = V_func_slow(V3[1:ra-1,int(2*ra/3):ra-1],n3[1:ra-1,int(2*ra/3):ra-1],m3[1:ra-1,int(2*ra/3):ra-1], c1[1:ra-1,int(2*ra/3):ra-1],I) + d4v1[1:ra-1,int(2*ra/3):ra-1]*(V3[1-1:ra-2,int(2*ra/3):ra-1]+V3[1+1:ra,int(2*ra/3):ra-1]+V3[1:ra-1,int(2*ra/3)-1:ra-2]+V3[1:ra-1,int(2*ra/3)+1:ra]+V4[1:ra-1,int(2*ra/3):ra-1]-5.0*V3[1:ra-1,int(2*ra/3):ra-1]) + d4v2[1:ra-1,int(2*ra/3):ra-1]*(V3[1-1:ra-2,int(2*ra/3)-1:ra-2]+V3[1-1:ra-2,int(2*ra/3)+1:ra]+V3[1+1:ra,int(2*ra/3)-1:ra-2]+V3[1+1:ra,int(2*ra/3)+1:ra]-4.0*V3[1:ra-1,int(2*ra/3):ra-1])
		naux3[1:ra-1,int(2*ra/3):ra-1] = n_func(V3[1:ra-1,int(2*ra/3):ra-1],n3[1:ra-1,int(2*ra/3):ra-1],m3[1:ra-1,int(2*ra/3):ra-1]) 
		
        #left wall
		Vaux3[0,1:ra-1] = V_func_slow(V3[0,1:ra-1],n3[0,1:ra-1],m3[0,1:ra-1], c1[0,1:ra-1],I) + d4v1[0,1:ra-1]*(V3[1,1:ra-1]+V3[0,0:ra-2]+V3[0,2:ra]+V3[0,ra-2:0:-1]+V4[0,1:ra-1]-5.0*V3[0,1:ra-1]) + d4v2[0,1:ra-1]*(V3[1,0:ra-2]+V3[1,2:ra]+V3[0,ra-1:1:-1]+V3[0,ra-3::-1]-4.0*V3[0,1:ra-1])
		naux3[0,1:ra-1] = n_func(V3[0,1:ra-1],n3[0,1:ra-1],m3[0,1:ra-1])
		
		Vaux3[0,1:int(2*ra/3)-1] = V_func_slow(V3[0,1:int(2*ra/3)-1],n3[0,1:int(2*ra/3)-1],m3[0,1:int(2*ra/3)-1], c1[0,1:int(2*ra/3)-1],I) + d4v1[0,1:int(2*ra/3)-1]*(V3[1,1:int(2*ra/3)-1]+V3[0,0:int(2*ra/3)-2]+V3[0,2:int(2*ra/3)]+V3[0,int(2*ra/3)-2:0:-1]-4.0*V3[0,1:int(2*ra/3)-1]) + d4v2[0,1:int(2*ra/3)-1]*(V3[1,0:int(2*ra/3)-2]+V3[1,2:int(2*ra/3)]+V3[0,int(2*ra/3)-1:1:-1]+V3[0,int(2*ra/3)-3::-1]-4.0*V3[0,1:int(2*ra/3)-1])
		naux3[0,1:int(2*ra/3)-1] = n_func(V3[0,1:int(2*ra/3)-1],n3[0,1:int(2*ra/3)-1],m3[0,1:int(2*ra/3)-1])
		
		#right wall
		Vaux3[ra-1,1:ra-1]= V_func_slow(V3[ra-1,1:ra-1],n3[ra-1,1:ra-1],m3[ra-1,1:ra-1], c1[ra-1,1:ra-1],I) + d4v1[ra-1,1:ra-1]*(V3[ra-2,1:ra-1]+V3[ra-1,0:ra-2]+V3[ra-1,2:ra]+V3[ra-1,ra-2:0:-1]+V4[ra-1,1:ra-1]-5.0*V3[ra-1,1:ra-1]) + d4v2[ra-1,1:ra-1]*(V3[ra-2,0:ra-2]+V3[ra-2,2:ra]+V3[ra-1,ra-1:1:-1]+V3[ra-1,ra-3::-1]-4.0*V3[ra-1,1:ra-1])
		naux3[ra-1,1:ra-1]= n_func(V3[ra-1,1:ra-1],n3[ra-1,1:ra-1],m3[ra-1,1:ra-1])
		
		#top wall
		
		Vaux3[1:ra-1,0]= V_func_slow(V3[1:ra-1,0],n3[1:ra-1,0],m3[1:ra-1,0], c1[1:ra-1,0],I) + d4v1[1:ra-1,0]*(V3[1:ra-1,1]+V3[0:ra-2,0]+V3[2:ra,0]+V3[ra-2:0:-1,0]+V4[1:ra-1,0]-5.0*V3[1:ra-1,0]) + d4v2[1:ra-1,0]*(V3[0:ra-2,1]+V3[2:ra,1]+V3[ra-1:1:-1,0]+V3[ra-3::-1,0]-4.0*V3[1:ra-1,0])
		naux3[1:ra-1,0]= n_func(V3[1:ra-1,0],n3[1:ra-1,0],m3[1:ra-1,0])
		
		Vaux3[1:int(ra/2-mv_lenght/2),0]= V_func_slow(V3[1:int(ra/2-mv_lenght/2),0],n3[1:int(ra/2-mv_lenght/2),0],m3[1:int(ra/2-mv_lenght/2),0], c1[1:int(ra/2-mv_lenght/2),0],I) + d4v1[1:int(ra/2-mv_lenght/2),0]*(V3[1:int(ra/2-mv_lenght/2),1]+V3[0:int(ra/2-mv_lenght/2)-1,0]+V3[2:int(ra/2-mv_lenght/2)+1,0]+V3[ra-2:int(ra/2+mv_lenght/2)-1:-1,0]-4.0*V3[1:int(ra/2-mv_lenght/2),0]) + d4v2[1:int(ra/2-mv_lenght/2),0]*(V3[0:int(ra/2-mv_lenght/2)-1,1]+V3[2:int(ra/2-mv_lenght/2)+1,1]+V3[ra-1:int(ra/2+mv_lenght/2):-1,0]+V3[ra-3:int(ra/2+mv_lenght/2)-2:-1,0]-4.0*V3[1:int(ra/2-mv_lenght/2),0])
		naux3[1:int(ra/2-mv_lenght/2),0]=      n_func(V3[1:int(ra/2-mv_lenght/2),0],n3[1:int(ra/2-mv_lenght/2),0],m3[1:int(ra/2-mv_lenght/2),0])
		




		#bottora wall
		
		Vaux3[1:ra-1,ra-1]= V_func_slow(V3[1:ra-1,ra-1],n3[1:ra-1,ra-1],m3[1:ra-1,ra-1], c1[1:ra-1,ra-1],I) + d4v1[1:ra-1,ra-1]*(V3[1:ra-1,ra-2]+V3[0:ra-2,ra-1]+V3[2:ra,ra-1]+V3[ra-2:0:-1,ra-1]+V4[1:ra-1,ra-1]-5.0*V3[1:ra-1,ra-1]) + d4v2[1:ra-1,ra-1]*(V3[0:ra-2,ra-2]+V3[2:ra,ra-2]+V3[ra-1:1:-1,ra-1]+V3[ra-3::-1,ra-1]-4.0*V3[1:ra-1,ra-1])
		naux3[1:ra-1,ra-1]= n_func(V3[1:ra-1,ra-1],n3[1:ra-1,ra-1],m3[1:ra-1,ra-1])
		
		Vaux3[0,0]= V_func_slow(V3[0,0],n3[0,0],m3[0,0], c1[0,0],I) + d4v1[0,0]*(V3[1,0]+V3[0,1]+V3[ra-1,0]+V3[0,ra-1]-4.0*V3[0,0]) + d4v2[0,0]*(V3[1,1]+V3[ra-1,ra-1]+V3[0,ra-2]+V3[ra-2,0]-4*V3[0,0])
		naux3[0,0]= n_func(V3[0,0],n3[0,0],m3[0,0])
		
		Vaux3[0,ra-1]= V_func_slow(V3[0,ra-1],n3[0,ra-1],m3[0,ra-1], c1[0,ra-1],I) + d4v1[0,ra-1]*(V3[1,ra-1]+V3[0,ra-2]+V3[0,0]+V3[ra-1,ra-1]+V4[0,ra-1]-5.0*V3[0,ra-1]) + d4v2[0,ra-1]*(V3[1,ra-2]+V3[1,0]+V3[0,1]+V3[ra-1,0]-4*V3[0,ra-1])
		naux3[0,ra-1]= n_func(V3[0,ra-1],n3[0,ra-1],m3[0,ra-1])
		
		Vaux3[ra-1,0]= V_func_slow(V3[ra-1,0],n3[ra-1,0],m3[ra-1,0], c1[ra-1,0],I) + d4v1[ra-1,0]*(V3[ra-1,1]+V3[ra-2,0]+V3[0,0]+V3[ra-1,ra-1]+V4[ra-1,0]-5.0*V3[ra-1,0]) + d4v2[ra-1,0]*(V3[ra-2,1]+V3[1,0]+V3[0,1]+V3[0,ra-1]-4*V3[ra-1,0])
		naux3[ra-1,0]= n_func(V3[ra-1,0],n3[ra-1,0],m3[ra-1,0])
		
		Vaux3[ra-1,ra-1]= V_func_slow(V3[ra-1,ra-1],n3[ra-1,ra-1],m3[ra-1,ra-1], c1[ra-1,ra-1],I) + d4v1[ra-1,ra-1]*(V3[ra-2,ra-1]+V3[ra-1,ra-2]+V3[ra-1,0]+V3[0,ra-1]+V4[ra-1,ra-1]-5.0*V3[ra-1,ra-1]) + d4v2[ra-1,ra-1]*(V3[ra-2,ra-2]+V3[1,ra-1]+V3[ra-1,1]+V3[0,0]-4*V3[ra-1,ra-1])
		naux3[ra-1,ra-1]= n_func(V3[ra-1,ra-1],n3[ra-1,ra-1],m3[ra-1,ra-1])
		
		Vaux3[int(ra/4),int(ra/4)]= V_func_slow(V3[int(ra/4),int(ra/4)],n3[int(ra/4),int(ra/4)],m3[int(ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-30]-V3[int(ra/4),int(ra/4)])
		naux3[int(ra/4),int(ra/4)]= n_func(V3[int(ra/4),int(ra/4)],n3[int(ra/4),int(ra/4)],m3[int(ra/4),int(ra/4)])


		##WOLF PARKINSON
		#Vaux3[int(3*ra/4),int(ra/4)]= V_func_slow(V3[int(3*ra/4),int(ra/4)],n3[int(3*ra/4),int(ra/4)],m3[int(3*ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-42]-V3[int(3*ra/4),int(ra/4)])
		#naux3[int(3*ra/4),int(ra/4)]= n_func(V3[int(3*ra/4),int(ra/4)],n3[int(3*ra/4),int(ra/4)],m3[int(3*ra/4),int(ra/4)])
	
	
		Vaux3[int(ra/2),ra-1]= V_func_slow(V3[int(ra/2),ra-1],n3[int(ra/2),ra-1],m3[int(ra/2),ra-1], const(I),I) +  1/6*(Vbundle[-10]-V3[int(ra/2),ra-1])#1/6*(V3[int(ra/4),int(ra/4)+7]-V3[int(ra/2),ra-1])
		naux3[int(ra/2),ra-1]= n_func(V3[int(ra/2),ra-1],n3[int(ra/2),ra-1],m3[int(ra/2),ra-1])
		
		Vaux3[int(3*ra/4),int(ra/4)]= V_func_slow(V3[int(3*ra/4),int(ra/4)],n3[int(3*ra/4),int(ra/4)],m1[int(3*ra/4),int(ra/4)], const(I),I) + 1/6*(Vbundle[-1]-V3[int(3*ra/4),int(ra/4)])#1/6*(V3[int(ra/4),int(ra/4)+7]-V3[int(3*ra/4),int(ra/4)])
		naux3[int(3*ra/4),int(ra/4)]= n_func(V3[int(3*ra/4),int(ra/4)],n3[int(3*ra/4),int(ra/4)],m1[int(3*ra/4),int(ra/4)])
		
		#mitral valve 
		Vaux3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0
		naux3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0

		Vaux3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]= V_func_slow(V3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],n3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],m3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],I) + d4v1[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]*(V3[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,mv_height]+V3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height+1]+V3[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,mv_height]-3.0*V3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]) + d4v2[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]*(V3[int(ra/2-mv_lenght/2)-1:int(ra/2+mv_lenght/2)-1,mv_height+1]+V3[int(ra/2-mv_lenght/2)+1:int(ra/2+mv_lenght/2)+1,mv_height+1]-2.0*V3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height])
		naux3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height]= n_func(V3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],n3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height],m3[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),mv_height])
		
		Vaux3[int(ra/2+mv_lenght/2),1:mv_height]= V_func_slow(V3[int(ra/2+mv_lenght/2),1:mv_height],n3[int(ra/2+mv_lenght/2),1:mv_height],m3[int(ra/2+mv_lenght/2),1:mv_height], c1[ra-1,0],I) + d4v1[int(ra/2+mv_lenght/2),1:mv_height]*(V3[int(ra/2+mv_lenght/2)+1,1:mv_height]+V3[int(ra/2+mv_lenght/2)+1,0:mv_height-1]+V3[int(ra/2+mv_lenght/2)+1,2:mv_height+1]-3.0*V3[int(ra/2+mv_lenght/2),1:mv_height]) + d4v2[int(ra/2+mv_lenght/2),1:mv_height]*(V3[int(ra/2+mv_lenght/2)+1,2:mv_height+1]+V3[int(ra/2+mv_lenght/2)+1,0:mv_height-1]-2*V3[int(ra/2+mv_lenght/2),1:mv_height])
		naux3[int(ra/2+mv_lenght/2),1:mv_height]=      n_func(V3[int(ra/2+mv_lenght/2),1:mv_height],n3[int(ra/2+mv_lenght/2),1:mv_height],m3[int(ra/2+mv_lenght/2),1:mv_height])
		
		Vaux3[int(ra/2-mv_lenght/2)-1,1:mv_height]= V_func_slow(V3[int(ra/2-mv_lenght/2)-1,1:mv_height],n3[int(ra/2-mv_lenght/2)-1,1:mv_height],m3[int(ra/2-mv_lenght/2)-1,1:mv_height], c1[ra-1,0],I) + d4v1[int(ra/2-mv_lenght/2)-1,1:mv_height]*(V3[int(ra/2-mv_lenght/2)-2,1:mv_height]+V3[int(ra/2-mv_lenght/2)-1,0:mv_height-1]+V3[int(ra/2-mv_lenght/2)-1,2:mv_height+1]-3.0*V3[int(ra/2-mv_lenght/2)-1,1:mv_height]) + d4v2[int(ra/2-mv_lenght/2)-1,1:mv_height]*(V3[int(ra/2-mv_lenght/2)-1-1,2:mv_height+1]+V3[int(ra/2-mv_lenght/2)-1-1,0:mv_height-1]-2*V3[int(ra/2-mv_lenght/2)-1,1:mv_height])
		naux3[int(ra/2-mv_lenght/2)-1,1:mv_height]=      n_func(V3[int(ra/2-mv_lenght/2)-1,1:mv_height],n3[int(ra/2-mv_lenght/2)-1,1:mv_height],m3[int(ra/2-mv_lenght/2)-1,1:mv_height])
		
		Vaux3[int(ra/2+mv_lenght/2),0]= V_func_slow(V3[int(ra/2+mv_lenght/2),0],n3[int(ra/2+mv_lenght/2),0],m3[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d4v1[int(ra/2+mv_lenght/2),0]*(V3[int(ra/2+mv_lenght/2),1]+V3[int(ra/2-mv_lenght/2)-1,0]+V3[int(ra/2+mv_lenght/2)+1,0]-3.0*V3[int(ra/2+mv_lenght/2),0]) + d4v2[int(ra/2+mv_lenght/2),0]*(V3[int(ra/2+mv_lenght/2)+1,1]+V3[int(ra/2-mv_lenght/2)-1,0]-2*V3[int(ra/2+mv_lenght/2),0])
		naux3[int(ra/2+mv_lenght/2),0]= n_func(V3[int(ra/2+mv_lenght/2),0],n3[int(ra/2+mv_lenght/2),0],m3[int(ra/2+mv_lenght/2),0])
		
		Vaux3[int(ra/2-mv_lenght/2)-1,0]= V_func_slow(V3[int(ra/2-mv_lenght/2)-1,0],n3[int(ra/2-mv_lenght/2)-1,0],m3[int(ra/2-mv_lenght/2)-1,0], c1[int(ra/2-mv_lenght/2)-1,0],I) + d4v1[int(ra/2-mv_lenght/2)-1,0]*(V3[int(ra/2-mv_lenght/2)-1,1]+V3[int(ra/2-mv_lenght/2)-1,0]+V3[int(ra/2+mv_lenght/2),0]-3.0*V3[int(ra/2-mv_lenght/2)-1,0]) + d4v2[int(ra/2-mv_lenght/2)-1,0]*(V3[int(ra/2+mv_lenght/2),1]+V3[int(ra/2-mv_lenght/2)-1,0]-2*V3[int(ra/2-mv_lenght/2)-1,0])
		naux3[int(ra/2-mv_lenght/2)-1,0]= n_func(V3[int(ra/2-mv_lenght/2)-1,0],n3[int(ra/2-mv_lenght/2)-1,0],m3[int(ra/2-mv_lenght/2)-1,0])
		

		Vaux3[0,1:int(ra/2)] = V_func_slow(V3[0,1:int(ra/2)],n3[0,1:int(ra/2)],m1[0,1:int(ra/2)], c1[0,1:int(ra/2)],I) + d1v1[0,1:int(ra/2)]*(V3[1,1:int(ra/2)]+V3[0,0:int(ra/2)-1]+V3[0,2:int(ra/2)+1]+V3[0,ra-1:int(ra/2):-1]-4.0*V3[0,1:int(ra/2)]) #+ d1v2[0,1:int(ra/2)]*(V3[1,0:int(ra/2)-1]+V3[1,2:int(ra/2)+1]+V3[0,int(ra/2):1:-1]+V3[0,ra-3::-1]-4.0*V3[0,1:int(ra/2)])
		naux3[0,1:int(ra/2)] = n_func(V3[0,1:int(ra/2)],n3[0,1:int(ra/2)],m1[0,1:int(ra/2)])

		Vaux4[int(ra/2):ra-1,1:ra-1] = V_func_fast(V4[int(ra/2):ra-1,1:ra-1],n4[int(ra/2):ra-1,1:ra-1],m4[int(ra/2):ra-1,1:ra-1], c1[int(ra/2):ra-1,1:ra-1],I) + d1v1[int(ra/2):ra-1,1:ra-1]*(V4[int(ra/2)-1:ra-2,1:ra-1]+V4[int(ra/2)+1:ra,1:ra-1]+V4[int(ra/2):ra-1,1-1:ra-2]+V4[int(ra/2):ra-1,1+1:ra]+V3[int(ra/2):ra-1,1:ra-1]-5.0*V4[int(ra/2):ra-1,1:ra-1]) + d1v2[int(ra/2):ra-1,1:ra-1]*(V4[int(ra/2)-1:ra-2,1-1:ra-2]+V4[int(ra/2)-1:ra-2,1+1:ra]+V4[int(ra/2)+1:ra,1-1:ra-2]+V4[int(ra/2)+1:ra,1+1:ra]-4.0*V4[int(ra/2):ra-1,1:ra-1])
		naux4[int(ra/2):ra-1,1:ra-1] = n_func_ventricle(V4[int(ra/2):ra-1,1:ra-1],n4[int(ra/2):ra-1,1:ra-1],m4[int(ra/2):ra-1,1:ra-1]) 
		
		Vaux4[1:int(ra/2)+1-1,int(2*ra/3):ra-1] = V_func_fast(V4[1:int(ra/2)+1-1,int(2*ra/3):ra-1],n4[1:int(ra/2)+1-1,int(2*ra/3):ra-1],m4[1:int(ra/2)+1-1,int(2*ra/3):ra-1], c1[1:int(ra/2)+1-1,int(2*ra/3):ra-1],I) + d1v1[1:int(ra/2)+1-1,int(2*ra/3):ra-1]*(V4[1-1:int(ra/2)+1-2,int(2*ra/3):ra-1]+V4[1+1:int(ra/2)+1,int(2*ra/3):ra-1]+V4[1:int(ra/2)+1-1,int(2*ra/3)-1:ra-2]+V4[1:int(ra/2)+1-1,int(2*ra/3)+1:ra]+V3[1:int(ra/2)+1-1,int(2*ra/3):ra-1]-5.0*V4[1:int(ra/2)+1-1,int(2*ra/3):ra-1]) + d1v2[1:int(ra/2)+1-1,int(2*ra/3):ra-1]*(V4[1-1:int(ra/2)+1-2,int(2*ra/3)-1:ra-2]+V4[1-1:int(ra/2)+1-2,int(2*ra/3)+1:ra]+V4[1+1:int(ra/2)+1,int(2*ra/3)-1:ra-2]+V4[1+1:int(ra/2)+1,int(2*ra/3)+1:ra]-4.0*V4[1:int(ra/2)+1-1,int(2*ra/3):ra-1])
		naux4[1:int(ra/2)+1-1,int(2*ra/3):ra-1] = n_func_ventricle(V4[1:int(ra/2)+1-1,int(2*ra/3):ra-1],n4[1:int(ra/2)+1-1,int(2*ra/3):ra-1],m4[1:int(ra/2)+1-1,int(2*ra/3):ra-1]) 
		
        #left wall
		Vaux4[0,int(2*ra/3):ra-1] = V_func_fast(V4[0,int(2*ra/3):ra-1],n4[0,int(2*ra/3):ra-1],m[0,int(2*ra/3):ra-1], const(I),I) + gx1[0,int(2*ra/3):ra-1]*(V4[1,int(2*ra/3):ra-1]+V4[0,int(2*ra/3)-1:ra-2]+V4[0,int(2*ra/3)+1:ra]+V4[0,ra-2:int(2*ra/3)-1:-1]+V3[0,int(2*ra/3):ra-1]-5.0*V4[0,int(2*ra/3):ra-1]) + gx2[0,int(2*ra/3):ra-1]*(V4[1,int(2*ra/3)-1:ra-2]+V4[1,int(2*ra/3)+1:ra]+V4[0,ra-1:int(2*ra/3):-1]+V4[0,ra-3:int(2*ra/3)-2:-1]-4.0*V4[0,int(2*ra/3):ra-1])
		naux4[0,int(2*ra/3):ra-1] = n_func_ventricle(V4[0,int(2*ra/3):ra-1],n4[0,int(2*ra/3):ra-1],m[0,int(2*ra/3):ra-1])
		
		#right wall
		Vaux4[ra-1,1:ra-1]= V_func_fast(V4[ra-1,1:ra-1],n4[ra-1,1:ra-1],m[ra-1,1:ra-1], const(I),I) + gx1[ra-1,1:ra-1]*(V4[ra-2,1:ra-1]+V4[ra-1,0:ra-2]+V4[ra-1,2:ra]+V4[ra-1,ra-2:0:-1]-4.0*V4[ra-1,1:ra-1]) + gx2[ra-1,1:ra-1]*(V4[ra-2,0:ra-2]+V4[ra-2,2:ra]+V4[ra-1,ra-1:1:-1]+V4[ra-1,ra-3::-1]-4.0*V4[ra-1,1:ra-1])
		naux4[ra-1,1:ra-1]= n_func_ventricle(V4[ra-1,1:ra-1],n4[ra-1,1:ra-1],m[ra-1,1:ra-1])
		
		#top wall
		Vaux4[int(ra/2):ra-1,0]= V_func_fast(V4[int(ra/2):ra-1,0],n4[int(ra/2):ra-1,0],m[int(ra/2):ra-1,0], const(I),I) + gx1[int(ra/2):ra-1,0]*(V4[int(ra/2):ra-1,1]+V4[int(ra/2)-1:ra-2,0]+V4[int(ra/2)+1:ra,0]-3.0*V4[int(ra/2):ra-1,0]) + gx2[int(ra/2):ra-1,0]*(V4[int(ra/2)-1:ra-2,1]+V4[int(ra/2)+1:ra,1]-2.0*V4[int(ra/2):ra-1,0])
		naux4[int(ra/2):ra-1,0]= n_func_ventricle(V4[int(ra/2):ra-1,0],n4[int(ra/2):ra-1,0],m[int(ra/2):ra-1,0])
		


		#bottora wall
		Vaux4[1:ra-1,ra-1]= V_func_fast(V4[1:ra-1,ra-1],n4[1:ra-1,ra-1],m[1:ra-1,ra-1], const(I),I) + gx1[1:ra-1,ra-1]*(V4[1:ra-1,ra-2]+V4[0:ra-2,ra-1]+V4[2:ra,ra-1]+V4[ra-2:0:-1,ra-1]-4.0*V4[1:ra-1,ra-1]) + gx2[1:ra-1,ra-1]*(V4[0:ra-2,ra-2]+V4[2:ra,ra-2]+V4[ra-1:1:-1,ra-1]+V4[ra-3::-1,ra-1]-4.0*V4[1:ra-1,ra-1])
		naux4[1:ra-1,ra-1]= n_func_ventricle(V4[1:ra-1,ra-1],n4[1:ra-1,ra-1],m[1:ra-1,ra-1])
		
		Vaux4[int(ra/2),0]= V_func_fast(V4[int(ra/2),0],n4[int(ra/2),0],m[int(ra/2),0], const(I),I) + gx1[int(ra/2),0]*(V4[int(ra/2)+1,0]+V4[int(ra/2),1]+V4[ra-int(ra/2),0]+V3[int(ra/2),0]-4.0*V4[int(ra/2),0]) + gx2[int(ra/2),0]*(V4[int(ra/2)+1,int(ra/2)+1]-V4[int(ra/2),0])
		naux4[int(ra/2),0]= n_func_ventricle(V4[int(ra/2),0],n4[int(ra/2),0],m[int(ra/2),0])
		
		Vaux4[0,ra-1]= V_func_fast(V4[0,ra-1],n4[0,ra-1],m[0,ra-1], const(I),I) + gx1[0,ra-1]*(V4[1,ra-1]+V4[0,ra-2]+V4[ra-1,ra-1]+V3[0,ra-1]-4.0*V4[0,ra-1]) + gx2[0,ra-1]*(V4[1,ra-2]+V4[ra-1,0]-2*V4[0,ra-1])
		naux4[0,ra-1]= n_func_ventricle(V4[0,ra-1],n4[0,ra-1],m[0,ra-1])
		
		Vaux4[ra-1,0]= V_func_fast(V4[ra-1,0],n4[ra-1,0],m[ra-1,0], const(I),I) + gx1[ra-1,0]*(V4[ra-1,1]+V4[ra-2,0]+V4[ra-1,ra-1]+V3[ra-1,0]-4.0*V4[ra-1,0]) + gx2[ra-1,0]*(V4[ra-2,1]+V4[0,ra-1]-2*V4[ra-1,0])
		naux4[ra-1,0]= n_func_ventricle(V4[ra-1,0],n4[ra-1,0],m[ra-1,0])
		
		Vaux4[ra-1,ra-1]= V_func_fast(V4[ra-1,ra-1],n4[ra-1,ra-1],m[ra-1,ra-1], const(I),I) + gx1[ra-1,ra-1]*(V4[ra-2,ra-1]+V4[ra-1,ra-2]+V4[ra-1,0]+V4[0,ra-1]+V3[ra-1,ra-1]-5.0*V4[ra-1,ra-1]) + gx2[ra-1,ra-1]*(V4[ra-2,ra-2]+V4[1,ra-1]+V4[ra-1,1]-3*V4[ra-1,ra-1])
		naux4[ra-1,ra-1]= n_func_ventricle(V4[ra-1,ra-1],n4[ra-1,ra-1],m[ra-1,ra-1])
		
		Vaux4[int(ra/2),1:int(2*ra/3)] =V_func_fast(V4[int(ra/2),1:int(2*ra/3)],n4[int(ra/2),1:int(2*ra/3)],m4[int(ra/2),1:int(2*ra/3)], c1[int(ra/2),1:int(2*ra/3)],I) + d1v1[int(ra/2),1:int(2*ra/3)]*(V4[int(ra/2)+1,1:int(2*ra/3)]+V4[int(ra/2),0:int(2*ra/3)-1]+V4[int(ra/2),1:int(2*ra/3)]-3.0*V4[int(ra/2),1:int(2*ra/3)]) + d1v2[int(ra/2),1:int(2*ra/3)]*(V4[int(ra/2)+1,0:int(2*ra/3)-1]+V4[int(ra/2)+1,1+1:int(2*ra/3)+1]-2.0*V4[int(ra/2),1:int(2*ra/3)])
		naux4[int(ra/2),1:int(2*ra/3)] =           n_func_ventricle(V4[int(ra/2),1:int(2*ra/3)],n4[int(ra/2),1:int(2*ra/3)],m4[int(ra/2),1:int(2*ra/3)]) 
		
		Vaux4[1:int(ra/2),int(2*ra/3)] = V_func_fast(V4[1:int(ra/2),int(2*ra/3)],n4[1:int(ra/2),int(2*ra/3)],m4[1:int(ra/2),int(2*ra/3)], c1[1:int(ra/2),int(2*ra/3)],I) + d1v1[1:int(ra/2),int(2*ra/3)]*(V4[0:int(ra/2)-1,int(2*ra/3)]+V4[2:int(ra/2)+1,int(2*ra/3)]+V4[1:int(ra/2),int(2*ra/3)+1]+V3[1:int(ra/2),int(2*ra/3)]-4.0*V4[1:int(ra/2),int(2*ra/3)]) + d1v2[1:int(ra/2),int(2*ra/3)]*(V4[0:int(ra/2)-1,int(2*ra/3)+1]+V4[1+1:int(ra/2)+1,int(2*ra/3)+1]-2.0*V4[1:int(ra/2),int(2*ra/3)])
		naux4[1:int(ra/2),int(2*ra/3)] = n_func_ventricle(V4[1:int(ra/2),int(2*ra/3)],n4[1:int(ra/2),int(2*ra/3)],m4[1:int(ra/2),int(2*ra/3)]) 
		
		
		Vaux4[int(ra/2):int(ra/2+mv_lenght/2),mv_height]= V_func_fast(V4[int(ra/2):int(ra/2+mv_lenght/2),mv_height],n4[int(ra/2):int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2):int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2):int(ra/2+mv_lenght/2),mv_height],I) + d1v1[int(ra/2):int(ra/2+mv_lenght/2),mv_height]*(V4[int(ra/2)+1:int(ra/2+mv_lenght/2)+1,mv_height]+V4[int(ra/2):int(ra/2+mv_lenght/2),mv_height+1]+V4[int(ra/2)-1:int(ra/2+mv_lenght/2)-1,mv_height]-3.0*V4[int(ra/2):int(ra/2+mv_lenght/2),mv_height]) + d1v2[int(ra/2):int(ra/2+mv_lenght/2),mv_height]*(V4[int(ra/2)-1:int(ra/2+mv_lenght/2)-1,mv_height+1]+V4[int(ra/2)+1:int(ra/2+mv_lenght/2)+1,mv_height+1]-2.0*V4[int(ra/2):int(ra/2+mv_lenght/2),mv_height])
		naux4[int(ra/2):int(ra/2+mv_lenght/2),mv_height]= n_func(V4[int(ra/2):int(ra/2+mv_lenght/2),mv_height],n4[int(ra/2):int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2):int(ra/2+mv_lenght/2),mv_height])
		
		Vaux4[int(ra/2+mv_lenght/2),1:mv_height]= V_func_fast(V4[int(ra/2+mv_lenght/2),1:mv_height],n4[int(ra/2+mv_lenght/2),1:mv_height],m4[int(ra/2+mv_lenght/2),1:mv_height], c1[ra-1,0],I) + d1v1[int(ra/2+mv_lenght/2),1:mv_height]*(V4[int(ra/2+mv_lenght/2)+1,1:mv_height]+V4[int(ra/2+mv_lenght/2)+1,0:mv_height-1]+V4[int(ra/2+mv_lenght/2)+1,2:mv_height+1]-3.0*V4[int(ra/2+mv_lenght/2),1:mv_height]) + d1v2[int(ra/2+mv_lenght/2),1:mv_height]*(V4[int(ra/2+mv_lenght/2)+1,2:mv_height+1]+V4[int(ra/2+mv_lenght/2)+1,0:mv_height-1]-2*V4[int(ra/2+mv_lenght/2),1:mv_height])
		naux4[int(ra/2+mv_lenght/2),1:mv_height]=      n_func(V4[int(ra/2+mv_lenght/2),1:mv_height],n4[int(ra/2+mv_lenght/2),1:mv_height],m4[int(ra/2+mv_lenght/2),1:mv_height])
		
		Vaux4[int(ra/2+mv_lenght/2),0]= V_func_fast(V4[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d1v1[int(ra/2+mv_lenght/2),0]*(V4[int(ra/2+mv_lenght/2),1]+V4[int(ra/2)-1-1,0]+V4[int(ra/2+mv_lenght/2)+1,0]-3.0*V4[int(ra/2+mv_lenght/2),0]) + d1v2[int(ra/2+mv_lenght/2),0]*(V4[int(ra/2+mv_lenght/2)+1,1]+V4[int(ra/2)-1-1,0]-2*V4[int(ra/2+mv_lenght/2),0])
		naux4[int(ra/2+mv_lenght/2),0]= n_func(V4[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0])
		
		Vaux4[int(ra/2),mv_height]=      V_func_fast(V4[int(ra/2),mv_height],n4[int(ra/2),mv_height],m[int(ra/2),mv_height], const(I),I) + d1v1[int(ra/2),mv_height]*(V4[int(ra/2)+1,mv_height]+V4[int(ra/2),mv_height+1]+V3[int(ra/2),mv_height]-3.0*V4[int(ra/2),mv_height]) + d1v2[int(ra/2)+1,mv_height+1]*(V4[int(ra/2)+1,mv_height+1]-V4[int(ra/2),mv_height])
		naux4[int(ra/2),mv_height]= n_func_ventricle(V4[int(ra/2),mv_height],n4[int(ra/2),mv_height],m[int(ra/2),mv_height])
		
		Vaux4[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0
		naux4[int(ra/2-mv_lenght/2):int(ra/2+mv_lenght/2),0:mv_height] = 0

		Vaux4[int(ra/2),int(2*ra/3)] = V_func_fast(V4[int(ra/2),int(2*ra/3)],n4[int(ra/2),int(2*ra/3)],m4[int(ra/2),int(2*ra/3)], c1[int(ra/2),int(2*ra/3)],I) + d1v1[int(ra/2),int(2*ra/3)]*(V4[int(ra/2),int(2*ra/3)+1]+V4[int(ra/2)+1,int(2*ra/3)]+V4[int(ra/2)-1,int(2*ra/3)]+V4[int(ra/2),int(2*ra/3)-1]+V3[int(ra/2),int(2*ra/3)]-5.0*V4[int(ra/2),int(2*ra/3)]) + d1v2[int(ra/2),int(2*ra/3)]*(V4[int(ra/2)+1,int(2*ra/3)+1]+V4[int(ra/2)-1,int(2*ra/3)+1]+V4[int(ra/2)-1,int(2*ra/3)-1]-3.0*V4[int(ra/2),int(2*ra/3)])
		naux4[int(ra/2),int(2*ra/3)] = n_func_ventricle(V4[int(ra/2),int(2*ra/3)],n4[int(ra/2),int(2*ra/3)],m4[int(ra/2),int(2*ra/3)]) 


		Vaux4[0,int(2*ra/3)] = V_func_fast(V4[0,int(2*ra/3)],n4[0,int(2*ra/3)],m4[0,int(2*ra/3)], c1[0,int(2*ra/3)],I) + d1v1[0,int(2*ra/3)]*(V4[0,int(2*ra/3)+1]+V4[0+1,int(2*ra/3)]+V3[0,int(2*ra/3)]-3.0*V4[0,int(2*ra/3)]) + d1v2[0,int(2*ra/3)]*(V4[1,int(2*ra/3)+1]-V4[0,int(2*ra/3)])
		naux4[0,int(2*ra/3)] = n_func_ventricle(V4[0,int(2*ra/3)],n4[0,int(2*ra/3)],m4[0,int(2*ra/3)]) 

		Vaux4[int(ra/2+mv_lenght/2),mv_height] = V_func_fast(V4[int(ra/2+mv_lenght/2),mv_height],n4[int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2+mv_lenght/2),mv_height], c1[int(ra/2+mv_lenght/2),mv_height],I) + d1v1[int(ra/2+mv_lenght/2),mv_height]*(V4[int(ra/2+mv_lenght/2),mv_height+1]+V4[int(ra/2+mv_lenght/2)+1,mv_height]+V4[int(ra/2+mv_lenght/2)-1,mv_height]+V4[int(ra/2+mv_lenght/2),mv_height-1]+V3[int(ra/2+mv_lenght/2),mv_height]-5.0*V4[int(ra/2+mv_lenght/2),mv_height]) + d1v2[int(ra/2+mv_lenght/2),mv_height]*(V4[int(ra/2+mv_lenght/2)+1,mv_height+1]+V4[int(ra/2+mv_lenght/2)-1,mv_height+1]+V4[int(ra/2+mv_lenght/2)-1,mv_height-1]-3.0*V4[int(ra/2+mv_lenght/2),mv_height])
		naux4[int(ra/2+mv_lenght/2),mv_height] = n_func_ventricle(V4[int(ra/2+mv_lenght/2),mv_height],n4[int(ra/2+mv_lenght/2),mv_height],m4[int(ra/2+mv_lenght/2),mv_height]) 

		Vaux4[int(ra/2+mv_lenght/2),0] = V_func_fast(V4[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0], c1[int(ra/2+mv_lenght/2),0],I) + d1v1[int(ra/2+mv_lenght/2),0]*(V4[int(ra/2+mv_lenght/2),0+1]+V4[int(ra/2+mv_lenght/2)+1,0]-2.0*V4[int(ra/2+mv_lenght/2),0]) + d1v2[int(ra/2+mv_lenght/2),0]*(V4[int(ra/2+mv_lenght/2)+1,1]-V4[int(ra/2+mv_lenght/2),0])
		naux4[int(ra/2+mv_lenght/2),0] = n_func_ventricle(V4[int(ra/2+mv_lenght/2),0],n4[int(ra/2+mv_lenght/2),0],m4[int(ra/2+mv_lenght/2),0]) 



		#COMPUTING THE 3-LEAD ECG FOR EACH TIME-STEP
		soma7 = 0 #LEAD I
		soma8 = 0 #LEAD II
		soma9 = 0 #LEAD III

		soma1 = 0
		V1s = np.flip(Vaux1,0)
		V2s = np.flip(Vaux2,0)

		soma7 = np.sum(V5_coeff[1:ra-1,1:ra-1]*(Vaux5[2:ra,2:ra]-Vaux5[1:ra-1,1:ra-1])) + np.sum(V6_coeff[1:ra-1,1:ra-1]*(Vaux6[2:ra,2:ra]-Vaux6[1:ra-1,1:ra-1])) + 5*np.sum(V1_coeff[1:ra-1,1:ra-1]*(V1s[2:ra,2:ra]-V1s[1:ra-1,1:ra-1])) + 15*np.sum(V3_coeff[1:ra-1,1:ra-1]*(Vaux3[2:ra,2:ra]-Vaux3[1:ra-1,1:ra-1]))
		soma8 = np.sum(V5_coeff[1:ra-1,1:ra-1]*(Vaux5[1:ra-1,2:ra]-Vaux5[1:ra-1,1:ra-1])) + np.sum(V6_coeff[1:ra-1,1:ra-1]*(Vaux6[1:ra-1,2:ra]-Vaux6[1:ra-1,1:ra-1])) + 5*np.sum(V1_coeff[1:ra-1,1:ra-1]*(V1s[1:ra-1,2:ra]-V1s[1:ra-1,1:ra-1])) + 15*np.sum(V3_coeff[1:ra-1,1:ra-1]*(Vaux3[1:ra-1,2:ra]-Vaux3[1:ra-1,1:ra-1]))
		soma9 = np.sum(V5_coeff[1:ra-1,1:ra-1]*(Vaux5[2:ra,0:ra-2]-Vaux5[1:ra-1,1:ra-1])) + np.sum(V6_coeff[1:ra-1,1:ra-1]*(Vaux6[2:ra,0:ra-2]-Vaux6[1:ra-1,1:ra-1])) + 5*np.sum(V1_coeff[1:ra-1,1:ra-1]*(V1s[2:ra,0:ra-2]-V1s[1:ra-1,1:ra-1])) + 15*np.sum(V3_coeff[1:ra-1,1:ra-1]*(Vaux3[2:ra,0:ra-2]-Vaux3[1:ra-1,1:ra-1]))

		
		total = 60088 
		lista_ecgI.append(-(soma7/total))
		lista_ecgII.append(-(soma8/total)-3)
		lista_ecgIII.append((soma9/total)-6)
		list_time.append(t*dt*50)
		
		
		#PLOTTING APs for specific cells
		V1middle.append(V5[30,30])
		V2middle.append(V6[30,30])
		V3middle.append(V1[30,30])
		V4middle.append(V3[30,30])
		V1[:,:], n1[:,:], m1[:,:] = Vaux1, naux1, maux1
		V2[:,:], n2[:,:], m2[:,:] = Vaux2, naux2, maux2 
		V3[:,:], n3[:,:], m3[:,:] = Vaux3, naux3, maux3
		V4[:,:], n4[:,:], m4[:,:] = Vaux4, naux4, maux4
		V5[:,:], n5[:,:], m5[:,:] = Vaux5, naux5, maux5
		V6[:,:], n6[:,:], m6[:,:] = Vaux6, naux6, maux6
		Vbundle[:], nbundle[:], mbundle[:] = Vbundle_aux, nbundle_aux, mbundle_aux


		SAN_list.append(V5[19,19])
		bach1_list.append(V6[19,19]-150) 
		bundle1_list.append(Vbundle[20]-450)
		avn_list.append(V5[int(3*ra/4),int(3*ra/4)]-300)
		ventricle1.append(V3[int(ra/4)+3,int(ra/4)+3]-600)
		epi1.append(V4[int(ra/2),ra-1]-750)
		ventricle2.append(V3[int(ra/2),int(ra/2)]-900)
		ventricle3.append(V3[int(3*ra/4),int(3*ra/4)]-1050)

		V1f = np.flip(V1,0)
		V2f = np.flip(V2,0)
		
		V_full[0:ra,0:ra] = V5[:,:]
		V_full[ra+10:,0:ra] = V6[:,:]
		V_full[0:rv:,rv+10:2*rv+10] = V1f[:,:]
		V_full[lv+10:2*lv+10,lv+10:2*lv+10] = V3[:,:]
		V_full[ra+5,40:40+bndl] = Vbundle[:]

	
import sys

np.set_printoptions(threshold=sys.maxsize)



#side_lenght = int(np.sqrt(2*M**2)/2)


#ANIMATION FUNCTION
def update_fig(i):   #update_fig(i)

	global Vaux1, naux1, maux1, V1, n1, m1, T, Tint, dt,V2,V3#,lista_ecg,list_time
	update(i)
	im1.set_array(rotate(V_full, angle=90))
	return  

anim=animation.FuncAnimation(fig,update_fig,frames=T+1,interval=0.01,blit=False,repeat=False)

plt.show()



#ECG PLOT
fig5 = plt.figure()
ax = fig5.add_subplot(1, 1, 1)
fig5.set_size_inches(8, 8)

major_ticks = np.arange(0, 15000, 400)
minor_ticks = np.arange(0, 15000, 40)

minor_ticksy = np.arange(-20, 20, 0.25)
major_ticksy = np.arange(-20, 20, 1)

ax.set_yticks(major_ticksy)
ax.set_yticks(minor_ticksy, minor=True)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)

ax.plot(list_time, lista_ecgI,color = "black",label="Model at 50bpm")
ax.plot(list_time, lista_ecgII,color = "black")
ax.plot(list_time, lista_ecgIII,color = "black")
ax.spines[['right', 'top']].set_visible(False)

ax.yaxis.set_ticklabels([])
ax.text(100,1,"I",size="18")
ax.text(100,-1,"II",size="18")
ax.text(100,-3,"III",size="18")
ax.grid(which='both')

ax.set_xlabel("t (ms)")
ax.set_ylabel(r"$<V_m>$ (mV)")

ax.grid(which='minor', alpha=0.2,color="red")
ax.grid(which='major', alpha=0.5,color="red")
plt.show()




#OUTPUTTING ECG DATA TO CSV
with open("../outputs/ecg_data_ischemia.csv", 'w', newline='') as file: #Change output file name
			writer = csv.writer(file)
			for i in range(len(lista_ecgI)):
				writer.writerow([list_time[i],lista_ecgI[i], lista_ecgII[i],lista_ecgIII[i]])
