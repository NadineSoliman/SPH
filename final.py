## Author: Nadine Soliman
# Computational Physics Final Project


import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
plt.ion()

#number of particles in the simulation
N =1000


## arrays that store the positions, velocities, densities and pressure values of each ith particle
x = np.array(x, float)
y = np.array(y, float)
vx = np.array(vx, float)
vy = np.array(vy, float)
rho = [0] * N
p = [0] * N
p = np.array(p, float)

## array that stores whether the particle is accreted or not 
accreted = [False] * N 


# mass of the central object (BH)
mbh = 1  ## in Msol
dt = 0.001  # time step in years
T  = 1000   # total time the simulation is setup to run
G = 39.5 ## gravitational constant in AU^3 Msol^-1 yr^-2

c = 632.849  ## speed of light in AU yr^-1

r_disk = 1 ## radius of the disk in AU
H_rdisk = 0.035/r_disk ## the H/r value of the disk 
v_k = np.sqrt(G * mbh/r_disk) ##keplerian velocity
cs = [H_rdisk * v_k] * N      # speed of sound in AU yr^-1

mdisk = 0.085				# total mass of particles in the disk
particle_mass = 0.085/N
m = [particle_mass] * N


## artificial viscosity terms 
alpha = 1
beta = 2


## softening parameter
eps = 0.1
eps_sq = eps**2

## kernel's h paramter
h = r_disk / np.sqrt(N)
hsqr = h**2

# radius within which particles would be be accreted into the blackhole
r_sch = 2*h

## the kernel's normalization factor
W =1/(h**2*np.pi)

## setting up the initial distrubution of the particles 
def initialize():
	for i in range(N):
		theta = np.random.normal() * np.pi
		r = np.abs(np.random.uniform()) + (2*h)
		i = r *math.cos(theta)
		j = r  * math.sin(theta)
		x.append(i)
		y.append(j)	
		r_= np.sqrt(i**2 + j**2 + eps_sq) 
		v =  np.sqrt(G * mbh/r_)
		vx.append(v * -j/r_)
		vy.append(v * i/r_)

def compute_pressure():
	for i in range(N):
		if(not accreted[i]):
			# reset all density and pressure values 
			rho[i] =0
			p[i] = 0
			for j in range(N):
				if(not accreted[j] and j!=i):
					r_sq = math.pow(x[i] - x[j], 2) + math.pow(y[i]- y[j], 2)
					if(9* hsqr>=r_sq):
						rho[i] += m[j]*math.exp(-r_sq/hsqr)
			p[i] = (cs[0])**2 * (rho[i])

def compute_force(dt, flag, mbh):
	for i in range(N):
		if(not accreted[i]):
			## reset all values to zero
			apress = 0
			a_press_x = 0
			a_press_y = 0
			art_visc = 0
			a_gradx = 0
			a_grady = 0
			a_visc_x = 0
			a_visc_y =0
			art_visc_x=0
			art_visc_y=0
			# for self gravitation not yet implemented
			# a_sgx =0
			# a_sgy = 0
			for j in range(N):
				if (not accreted[j]):
					if ( i!=j and rho[i] !=0 and rho[j]!= 0 ):
						dx = float(x[i] - x[j])
						dy = float(y[i] - y[j])
						r_sq = math.pow(dx, 2) + math.pow(dy, 2)
						# check if particles are within the radius of the kernel (3h)
						if(9*hsqr>= r_sq):
							r = np.sqrt(r_sq + eps_sq)
							q = r/h
							apress += -m[j]* (p[i] + p[j])/(2*rho[j])
							vijx = vx[i] - vx[j]
							vijy = vy[i] - vy[j]
							dotproduct = (vijx * dx) + (vijy * dy)
							#calculate artificial visocity terms
							a_visc_x +=  m[j]/(rho[i]*rho[j]) * vijx * nu_visc * W* (-2 * math.exp(-2 * q**2)  + -2*q**2 *math.exp(-2 * q**2) )
							a_visc_y += m[j]/(rho[i]*rho[j])* vijy * nu_visc * W* (-2 * math.exp(-2 * q**2)  + -2*q**2 *math.exp(-2 * q**2) )
							if dotproduct<0:
								uij = h* dotproduct/(r_sq + 0.01*hsqr)
								pi_ij = ((-alpha * uij * (cs[i] + cs[j]))+  beta * math.pow(uij, 2)) / (0.5 * (rho[i] + rho[j]))
								art_visc = -m[j] * pi_ij * W * (-2 * q)*math.exp(-2 * q**2) *dx/(r)
							# calculate pressure terms
							a_press_x += (apress/rho[i] ) * W * (-2 * q)*math.exp(-2 * q**2) *dx/(r)
							a_press_y += (apress/rho[i] ) *  W * (-2 * q)*math.exp(-2 * q**2) *dy/(r)
							# calculate artificial visocity terms 
							art_visc_x += 0.5 * art_visc * dx/r
							art_visc_y += 0.5 * art_visc *dy/r
							# a_sgx += G*m[j]/(r**2) *  W * (-2 * q)*math.exp(-2 * q**2) *dx/(r) 
							# a_sgy += G*m[j]/(r**2) *  W * (-2 * q)*math.exp(-2 * q**2) *dy/(r) 
			# calculate central gravitational force
			r_sq = (x[i]**2 + y[i]**2)
			r = np.sqrt(r_sq + eps_sq)
			a_grav_x = -G*mbh/(r_sq +eps_sq) * x[i]/(r )
			a_grav_y = -G*mbh/(r_sq +eps_sq)* y[i]/(r )
 

 			# if this is the first time step, begin leap frog integration
			if flag:
				vx[i] += (a_press_x + a_grav_x  + a_visc_x +art_visc_x ) * dt/2
				vy[i] += (a_press_y + a_grav_y  + a_visc_y + art_visc_y ) * dt/2

			else:
				vx[i] += (a_press_x+a_grav_x+ a_visc_x + art_visc_x ) * dt
				vy[i] += (a_press_y+a_grav_y+ a_visc_y + art_visc_y ) * dt
				x[i] += vx[i] * dt
				y[i] += vy[i] * dt
				# if the paritcle is within the "Schwarzschild" radius, it 
				#is considered accreted within th central object, place them in the center
				## add their mass to the central object 
				if ((x[i] **2  + y[i]**2) <= r_sch):
					accreted[i] = True
					mbh += m[i]
					x[i] = 0
					y[i] = 0
	return mbh

def update(dt, flag, mbh):
	compute_pressure()
	mbh = compute_force(dt, flag, mbh)
	return mbh

f,(ax1, ax2) = plt.subplots(1, 2)
def plot():
	ax1.scatter(x, y)
	ax2.plot(time, mass_mbh)

	plt.pause(0.000001)
	ax1.set_xlabel('x [AU]')
	ax2.set_xlabel('Time [yr]')
	ax1.set_ylabel('y [AU]')
	ax2.set_ylabel('Mass of Star ' + r'$[M_\odot]$')
	ax1.clear()
	ax2.clear()

	## the lines commented out result in a smoothened representation of the particles in log scale
	# h, xe, ye, img = plt.hist2d(x, y, bins = 50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.jet)
	# extent = [xe[0], xe[-1], ye[0], ye[-1]]
	# plt.imshow(h, interpolation = "xwgaussian",  extent=extent)
	# plt.xlim(-1.5, 1.5)
	# plt.ylim(-1.5, 1.5)
	plt.show()


## begin the simulation by setting up the particles 
t = 0
flag = True
initialize()

## alpha - viscosity terms
alpha_visc = 0.1
H = r_disk * 0.035
# coefficient of friction
nu_visc = alpha_visc * H * cs[0]

counter = 0
time = []
mass_mbh = []


while t <T:
	# dynamically calculated time-step
	v = np.sqrt(vx**2 + vy**2)
	dt = 0.3 *h/max(cs[0] + v)
	time.append(t)
	counter +=1

	mass_mbh.append(mbh)
	mbh = update(dt, flag, mbh)
	if flag:
		flag = False

	# plot every 5 time steps 
	if (counter %5==0):
		plot()
	t+= dt

# save the data to plot data faster to make movies
data = {'x': x, 'y': y, 'time': time, 'mbh': mass_mbh, 'mass': mbh}
pickle.dump( data, open( str(H_rdisk) + ".p", "wb" ) )

