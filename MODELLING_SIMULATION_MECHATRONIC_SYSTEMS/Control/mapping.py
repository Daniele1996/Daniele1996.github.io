from sympy import *
import numpy as np

# defining all the terms

vx_dot, vy_dot, psi_doubledot = symbols('vx_dot vy_dot psi_doubledot') 
x, y, psi, vx, vy, psi_dot, ffz, frz = symbols('x y psi vx vy psi_dot ffz frz') #states
mu_fx, mu_fy, mu_rx, mu_ry = symbols('mu_fx mu_fy mu_rx mu_ry') #inputs
a, b, c, cxbeta, cysigma, d, e, g, h, m, Ixz, Izz, r1, r2, tf, tr = symbols('a b c cxbeta cysigma d e g h m Ixz Izz r1 r2 tf tr') #constants
bfx, cfx, dfx, efx, r1fx, r2fx = symbols('bfx cfx dfx efx r1fx r2fx')
bfy, cfy, dfy, efy, r1fy, r2fy = symbols('bfy cfy dfy efy r1fy r2fy')
brx, crx, drx, erx, r1rx, r2rx = symbols('brx crx drx erx r1rx r2rx')
bry, cry, dry, ery, r1ry, r2ry = symbols('bry cry dry ery r1ry r2ry')
beta_f, beta_r, sigma_f, sigma_r, delta = symbols('beta_f beta_r sigma_f sigma_r delta') # pacejka's constants
dt, x_p, y_p, psi_p, vx_p, vy_p, psi_dot_p = symbols('dt x_p y_p psi_p vx_p vy_p psi_dot_p') # euler's constants
v_contact_front, v_contact_rear = symbols('v_contact_front v_contact_rear')

# Pacejka tire model

# psi_dot_skew = Matrix([[0, -psi_dot, 0], [psi_dot, 0, 0], [0, 0, 0]])
# v_contact_front = Matrix([[vx], [vy], [0]]) + psi_dot_skew*Matrix([[a+b], [0], [0]])
# v_contact_rear = Matrix([[vx], [vy], [0]])

# rotation_matrix = Matrix([[cos(delta), sin(delta), 0], [-sin(delta), cos(delta), 0], [0, 0, 1]])
# v_contact_front = rotation_matrix*v_contact_front

# beta_f = atan2(v_contact_front[1], v_contact_front[0])
# beta_f = - beta_f + delta

# beta_r = atan2(v_contact_rear[1], v_contact_rear[0])

mu_fx = dfx*sin(cfx*atan(bfx*sigma_f-efx*(bfx*sigma_f-atan(bfx*sigma_f)))) * cos(cxbeta*atan(beta_f*r1fx/(1+(r2fx*sigma_f)**2)))
mu_rx = drx*sin(crx*atan(brx*sigma_r-erx*(brx*sigma_r-atan(brx*sigma_r)))) * cos(cxbeta*atan(beta_r*r1rx/(1+(r2rx*sigma_r)**2)))
mu_fy = dfy*sin(cfy*atan(bfy*beta_f-efy*(bfy*beta_f-atan(bfy*beta_f)))) * cos(cysigma*atan(sigma_f*r1fy/(1+(r2fy*beta_f)**2)))
mu_ry = dry*sin(cry*atan(bry*beta_r-ery*(bry*beta_r-atan(bry*beta_r)))) * cos(cysigma*atan(sigma_r*r1ry/(1+(r2ry*beta_r)**2)))

mu_fx = mu_fx*cos(delta) - mu_fy*sin(delta)
mu_rx = mu_rx
mu_fy = mu_fx*sin(delta) + mu_fy*cos(delta)
mu_ry = mu_ry

deriv_mu_fx = Derivative(mu_fx, sigma_f)
print(deriv_mu_fx)
quit()
# writing the system 

M = Matrix([[m, 0, 0, mu_fx, mu_rx],
            [0, m, m*b, mu_fy, mu_ry],
            [0, m*b, Izz+m*b**2, (a+b)*mu_fy, 0],
            [0,0,0,-1,-1],
            [-m*h, 0, 0, a+b, 0]])

K = Matrix([[-m*b*psi_dot**2 - m*vy*psi_dot],
            [m*vx*psi_dot],
            [m*b*vx*psi_dot],
            [-m*g],
            [(Ixz + m*h*b)*psi_dot**2 + m*h*vy*psi_dot+m*g*b]])

# eqs = [m*vx_dot + mu_fx*ffz + mu_rx*frz - m*b*psi_dot**2-m*vy*psi_dot,
#        m*vy_dot + m*b*psi_doubledot + mu_fy*ffz + mu_ry*frz + m*vx*psi_dot,
#        m*b*vy_dot + (Izz + m*b**2)*psi_doubledot + (a+b)*mu_fy*ffz + m*b*vx*psi_dot,
#        -ffz-frz-m*g,
#        -m*h*vx_dot + (a+b)*ffz + (Ixz + m*h*b)*psi_dot**2 + m*h*vy*psi_dot + m*g*b]

# solving the system

out = -M.inv()*K
out= simplify(out)

# result = simplify(solve(eqs, [vx_dot, vy_dot, psi_doubledot, ffz, frz]))

# printing the result in a text file 

# with open('system_solved.txt', 'w') as file:
#     for elem in [vx_dot, vy_dot, psi_doubledot, ffz, frz]:
#        file.write(str(elem)+ ' = \n\n\t\t\t' + str(result[elem]) + '\n\n')

names = ['Vx_dot', 'Vy_dot','Psi_double_dot','Ffz','Frz']
with open('system_solved.txt', 'w') as file:
    for i in range(5):
       file.write(names[i]+'=\t'+str(out[i])+ '\n\n')
