import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def basis_func(t):
    t2 = t*t
    t3 = t2*t
    h00 = 2.0*t3-3.0*t2+1.0
    h10 = t3-2.0*t2+t
    h01 = -2.0*t3+3.0*t2
    h11 = t3 - t2
    return h00,h10,h01,h11

def limit_tangent(dk1, dk2, s1, s2):
   if dk1*s1 <= 0.0:
       t1 = 0.0
   else:
       alpha = s1 / dk1           
       if alpha > 3.0:
           t1 = 3.0 * dk1
       else:
           t1 = s1
   if dk2*s2 <= 0.0:
       t2 = 0.0
   else:
       beta  = s2 / dk2    
       if beta > 3.0:
           t2 = 3.0 * dk2               
       else:
           t2 = s2
   return t1, t2
   
def limit_tangent_hyman(dk1, dk2, s):
   if dk1*dk2 <= 0.0:
       t = 0.0
   else:
       if s >= 0.0:
           if np.abs(dk1) < np.abs(dk2):
               t = 3.0*np.abs(dk1)
           else:
               t = 3.0*np.abs(dk2)
           if s < t:
               t = s
       else:
           if np.abs(dk1) < np.abs(dk2):
               t = -3.0*np.abs(dk1)
           else:
               t = -3.0*np.abs(dk2)
           if s > t:
               t = s
   return t   
   

def interp_1d(t, i, xi, yi, mono=False):
    if i-1 < 0:
        s1=(yi[i+1]-yi[i])/(xi[i+1]-xi[i])
    else:
        h1 = xi[i]-xi[i-1]
        h2 = xi[i+1]-xi[i]         
        s1=((yi[i+1]-yi[i])/h2*h1 + (yi[i]-yi[i-1])/h1*h2)/(h1+h2)

    if i == xi.shape[0]-2:
        s2 = (yi[i+1]-yi[i])/(xi[i+1]-xi[i])
    else:
        h1 = xi[i+1]-xi[i]
        h2 = xi[i+2]-xi[i+1]            
        s2=((yi[i+2]-yi[i+1])/h2*h1+(yi[i+1]-yi[i])/h1*h2)/(h1+h2)

    if mono:
       if i-1 < 0:
            dk1=(yi[i+1]-yi[i])/(xi[i+1]-xi[i])
       else:            
            dk1 = (yi[i]-yi[i-1])/(xi[i]-xi[i-1])
       dk2 = (yi[i+1]-yi[i])/(xi[i+1]-xi[i])
       t1=limit_tangent_hyman(dk1, dk2, s1)    
       dk1= (yi[i+1]-yi[i])/(xi[i+1]-xi[i])
       if i == xi.shape[0]-2:
           dk2 = (yi[i+1]-yi[i])/(xi[i+1]-xi[i])
       else:
           dk2 = (yi[i+2]-yi[i+1])/(xi[i+2]-xi[i+1])

       t2=limit_tangent_hyman(dk1, dk2, s2)    
    else:
       t1 = s1
       t2 = s2
    h00,h10,h01,h11=basis_func(t)
    ynew = h00*yi[i] + h10*(xi[i+1]-xi[i])*t1 + h01*yi[i+1] + h11*(xi[i+1]-xi[i])*t2

    return ynew    

def bicubic_mono(xi, yi, zi, xnew, ynew, mono=False):
   assert len(xnew.shape) == 2
   assert len(ynew.shape) == 2
   assert xnew.shape==ynew.shape   
   znew=np.zeros(xnew.shape)
   for n in range(0, xnew.shape[1]):       
       for m in range(0, ynew.shape[0]):
           i = np.searchsorted(xi, xnew[m,n])-1
           j = np.searchsorted(yi, ynew[m,n])-1
           t = (xnew[m,n]-xi[i])/(xi[i+1]-xi[i])
           tmpj = interp_1d(t, i, xi, zi[j,:], mono)
           tmpjp1 = interp_1d(t, i, xi, zi[j+1,:], mono)
           if j-1 >= 0:
               tmpjm1 = interp_1d(t, i, xi, zi[j-1,:], mono)
           if j+2 < yi.shape[0]:
               tmpjp2 = interp_1d(t, i, xi, zi[j+2,:], mono)
           r = (ynew[m,n]-yi[j])/(yi[j+1]-yi[j])
           if j-1 < 0:           
               s1=(tmpjp1-tmpj)/(yi[j+1]-yi[j])
           else:  
               h1 = yi[j]-yi[j-1]
               h2 = yi[j+1]-yi[j]
               s1=((tmpjp1-tmpj)/h2*h1+(tmpj-tmpjm1)/h1*h2)/(h1+h2)
           if j == yi.shape[0]-2:           
               s2 = (tmpjp1-tmpj)/(yi[j+1]-yi[j])
           else:           
               h1 = yi[j+1]-yi[j]
               h2 = yi[j+2]-yi[j+1]
               s2=((tmpjp2-tmpjp1)/h2*h1+ (tmpjp1-tmpj)/h1*h2)/(h1+h2)
           if mono:               
               if j-1 < 0:           
                   dk1 = (tmpjp1-tmpj)/(yi[j+1]-yi[j])
               else:
                   dk1 = (tmpj-tmpjm1)/(yi[j]-yi[j-1])
               dk2 = (tmpjp1-tmpj)/(yi[j+1]-yi[j])    
               t1=limit_tangent_hyman(dk1, dk2, s1)                   
               dk1 = (tmpjp1-tmpj)/(yi[j+1]-yi[j])
               if j == yi.shape[0]-2:           
                   dk2 = (tmpjp1-tmpj)/(yi[j+1]-yi[j])
               else:
                   dk2 = (tmpjp2-tmpjp1)/(yi[j+2]-yi[j+1])
               t2=limit_tangent_hyman(dk1, dk2, s2)                   
           
           else:
               t1 = s1
               t2 = s2
           h00,h10,h01,h11=basis_func(r)           
           znew[m,n]= h00*tmpj + h10*(yi[j+1]-yi[j])*t1 + h01*tmpjp1 + h11*(yi[j+1]-yi[j])*t2                          

   return znew

# Define the interval and the function
interval_start = 0
interval_end = np.pi
num_samples_x = 10
num_samples_y = 10
num_interp_points_x = 100
num_interp_points_y = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples_x)
yi = np.linspace(interval_start, interval_end, num_samples_y)

# Generate a 2D grid of sample points
x, y = np.meshgrid(xi, yi)

# Compute the function values at these sample points
zi = np.sin(x) * np.cos(y)

# Generate a 2D grid for interpolation
xin = np.linspace(interval_start, interval_end, num_interp_points_x)
yin = np.linspace(interval_start, interval_end, num_interp_points_y)
xn, yn = np.meshgrid(xin, yin)

# Perform bicubic interpolation
z_interpolated = bicubic_mono(xi, yi, zi, xn, yn, mono=True)

# Plot the original function and the interpolated values
plt.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Original Function')
plt.scatter(xi, yi, c='red', marker='o', label='Sample Points')
plt.contour(xn, yn, z_interpolated, levels=10, colors='white', linestyles='dashed', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bicubic Interpolation of sin(x) * cos(y)')
plt.legend()
plt.show()