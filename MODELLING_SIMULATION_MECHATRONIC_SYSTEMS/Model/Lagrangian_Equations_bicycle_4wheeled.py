from sympy import *

t = symbols('t')
x = Function('x')(t)
y = Function('y')(t)
z = Function('z')(t)
psi = Function('psi')(t)
theta = Function('theta')(t)
fi = Function('fi')(t)

xdot = Function('xdot')(t)
ydot = Function('ydot')(t)
psidot = Function('psidot')(t)
zdot = Function('zdot')(t)
thetadot = Function('thetadot')(t)
fidot = Function('fidot')(t)

x_doubledot, y_doubledot, psi_doubledot, z_doubledot, theta_doubledot, fi_doubledot, Ixx, Ixz, Izz, Iyy, m, b, g, h, d, a, tf, tr = symbols('x_doubledot y_doubledot psi_doubledot z_doubledot theta_doubledot fi_doubledot Ixx Ixz Izz Iyy m b g h d a tf tr') 
vx_dot, vy_dot, vx, vy = symbols('vx_dot vy_dot vx vy') 
c_front_right, c_front_left, c_rear_right, c_rear_left = symbols('c_front_right c_front_left c_rear_right c_rear_left') 

show_bi_cycle = True
show_4_wheeled = True

if show_bi_cycle:
    # BI-CYCLE MODEL ##############################################################################
    Ry = Matrix([[cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)]])

    Rz = Matrix([[cos(psi), -sin(psi), 0],
            [sin(psi), cos(psi), 0],
            [0, 0, 1]]) 

    R = Rz*Ry # rotation matrix from body-ref-frame to abs-ref-frame

    OG_abs = Matrix([[x],[y],[z]]) + R*Matrix([[b],[0], [-h]]) # (CG - O) in abs-ref-frame 

    v_linear_abs = diff(OG_abs, t) # velocity of the baricenter in the abs-ref-frame
    v_linear_abs = v_linear_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)

    I_body = Matrix([[Ixx, 0, Ixz],
                    [0, Iyy, 0],
                    [Ixz, 0, Izz]]) # inertia matrix of the car WRT the ref-frame centered in the baricenter

    w_matrix_abs = diff(R, t)*R.T # skew-symm mat associated to the angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the abs-ref-frame
    w_matrix_abs = w_matrix_abs.subs(Derivative(psi, t), psidot).subs(Derivative(theta, t), thetadot)
    w_matrix_abs = simplify(w_matrix_abs)
    w_matrix_body = simplify(R.T*w_matrix_abs*R) # skew-symm mat associated to the angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the body-ref-frame
    w_matrix_body = w_matrix_body.subs(Derivative(psi, t), psidot).subs(Derivative(theta, t), thetadot)
    w_body = simplify(Matrix([[w_matrix_body[2,1]],
                        [w_matrix_body[0,2]],
                        [w_matrix_body[1,0]]])) # angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the body-ref-frame

    # LAGRANGIAN FUNCTION
    kinetic_linear = simplify(0.5*v_linear_abs.T*m*v_linear_abs)
    kinetic_linear = kinetic_linear[0]

    kinetic_rotational = simplify(0.5*w_body.T*I_body*w_body)
    kinetic_rotational = kinetic_rotational[0]

    potential = - m*g*OG_abs[2]

    L = kinetic_linear + kinetic_rotational - potential

    # GENERALIZED FORCES
    v_rear_b = R.T*Matrix([[xdot], [ydot], [zdot]]) # velocity of the rear wheel expressed in the body-ref-frame
    v_rear_b = simplify(v_rear_b)

    OF_abs = Matrix([[x],[y],[z]]) + R*Matrix([[a+b],[0], [0]]) # position vector of the front wheel expressed in the abs-ref-frame
    v_front_abs = diff(OF_abs, t) 
    v_front_b = R.T*v_front_abs # velocity of the front wheel expressed in the body-ref-frame
    v_front_b = simplify(v_front_b)

    J = Matrix([[cos(psi)*cos(theta), sin(psi)*cos(theta), 0, -sin(theta),0],
                [-sin(psi), cos(psi), (a+b)*cos(theta),0,0],
                [cos(psi)*cos(theta), sin(psi)*cos(theta), 0, -sin(theta), 0],
                [-sin(psi), cos(psi), 0, 0, 0]]) # manually compute the jacobian from previous relations

    Ffx_body, Ffy_body, Frx_body, Fry_body, Ffz_body, Frz_body, mu_fx_body, mu_fy_body, mu_rx_body, mu_ry_body = symbols('Ffx_body Ffy_body Frx_body Fry_body Ffz_body Frz_body mu_fx_body mu_fy_body mu_rx_body mu_ry_body ') 
    Gen_forces_abs = J.T*Matrix([[Ffx_body], [Ffy_body], [Frx_body], [Fry_body]])

    print("\n////// BI_CYCLE MODEL \\\\\\\\\\\\\n\n")
    print(f"- - - Generalized forces for Lagrangian equations - - -")
    print(f"Generalized forces, 1st_eq : \n{Gen_forces_abs[0]}\n")
    print(f"Generalized forces, 2nd_eq : \n{Gen_forces_abs[1]}\n")
    print(f"Generalized forces, 3rd_eq : \n{Gen_forces_abs[2]}\n")
    print(f"Generalized forces, 4th_eq : \n{Gen_forces_abs[3]}\n")
    print(f"Generalized forces, 5th_eq : \n{Gen_forces_abs[4]}\n\n")

    # CONSTRAINED FORCES
    A = Matrix([[cos(psi)*sin(theta), sin(psi)*sin(theta), 0, cos(theta), -(a+b)],
                [cos(psi)*sin(theta), sin(psi)*sin(theta), 0, cos(theta), 0]])

    Constrained_forces_abs = A.T*Matrix([[-Ffz_body], [-Frz_body]])

    print(f"- - - Constrained forces for Lagrangian equations - - -")
    print(f"Constrained forces, 1st_eq : \n{Constrained_forces_abs[0]}\n")
    print(f"Constrained forces, 2nd_eq : \n{Constrained_forces_abs[1]}\n")
    print(f"Constrained forces, 3rd_eq : \n{Constrained_forces_abs[2]}\n")
    print(f"Constrained forces, 4th_eq : \n{Constrained_forces_abs[3]}\n")
    print(f"Constrained forces, 5th_eq : \n{Constrained_forces_abs[4]}\n\n")

    # LAGRANGIAN EQUATIONS
    print(f"- - - Lagrangian expressed in the absolute frame - - -\nL = {L}\n")

    dL_dx = diff(L, x)
    dL_dxdot = diff(L, xdot)
    Dt_dL_dxdot = diff(dL_dxdot, t)
    _1st_eq = Dt_dL_dxdot - dL_dx

    _1st_eq_abs = _1st_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)
    _1st_eq_abs= _1st_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot)

    _1st_eq_abs = _1st_eq_abs - Gen_forces_abs[0] + Constrained_forces_abs[0]
    _1st_eq_abs = simplify(_1st_eq_abs)
    _1st_eq_abs = _1st_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0)

    print(f"1st Lagrangian eq, absolute frame : \n{_1st_eq_abs} = 0\n")

    dL_dy = diff(L, y)
    dL_dydot = diff(L, ydot)
    Dt_dL_dydot = diff(dL_dydot, t)
    _2nd_eq = Dt_dL_dydot - dL_dy

    _2nd_eq_abs = _2nd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)
    _2nd_eq_abs= _2nd_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot)

    _2nd_eq_abs = _2nd_eq_abs - Gen_forces_abs[1] + Constrained_forces_abs[1]
    _2nd_eq_abs = simplify(_2nd_eq_abs)
    _2nd_eq_abs = _2nd_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0)

    print(f"2nd Lagrangian eq, absolute frame : \n{_2nd_eq_abs} = 0\n")

    dL_dpsi = diff(L, psi)
    dL_dpsidot = diff(L, psidot)
    Dt_dL_dpsidot = diff(dL_dpsidot, t)
    _3rd_eq = Dt_dL_dpsidot - dL_dpsi

    _3rd_eq_abs = _3rd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)
    _3rd_eq_abs= _3rd_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot)

    _3rd_eq_abs = _3rd_eq_abs - Gen_forces_abs[2] + Constrained_forces_abs[2]
    _3rd_eq_abs = simplify(_3rd_eq_abs)
    _3rd_eq_abs = _3rd_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0)

    print(f"3rd Lagrangian eq, absolute_frame : \n{_3rd_eq_abs} = 0\n")

    dL_z = diff(L, z)
    dL_zdot = diff(L, zdot)
    Dt_dL_zdot = diff(dL_zdot, t)
    _4th_eq_abs = Dt_dL_zdot - dL_z

    _4th_eq_abs = _4th_eq_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)
    _4th_eq_abs= _4th_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot)

    _4th_eq_abs = _4th_eq_abs - Gen_forces_abs[3] + Constrained_forces_abs[3]
    _4th_eq_abs = simplify(_4th_eq_abs)
    _4th_eq_abs = _4th_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0)

    print(f"4th Lagrangian eq, absolute_frame : \n{_4th_eq_abs} = 0\n")

    dL_theta = diff(L, theta)
    dL_thetadot = diff(L, thetadot)
    Dt_dL_thetadot = diff(dL_thetadot, t)
    _5th_eq_abs = Dt_dL_thetadot - dL_theta

    _5th_eq_abs = _5th_eq_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot)
    _5th_eq_abs= _5th_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot)

    _5th_eq_abs = _5th_eq_abs - Gen_forces_abs[4] + Constrained_forces_abs[4]
    _5th_eq_abs = simplify(_5th_eq_abs)
    _5th_eq_abs = _5th_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0)
    
    print(f"5th Lagrangian eq, absolute_frame : \n{_5th_eq_abs} = 0\n\n")

    # CAST EQUATIONS IN THE BODY-REF-FRAME
    x_doubledot_proj = vx_dot*cos(psi) - vx*sin(psi)*psidot - vy_dot*sin(psi) - vy*cos(psi)*psidot
    y_doubledot_proj = vx_dot*sin(psi) + vx*cos(psi)*psidot + vy_dot*cos(psi) - vy*sin(psi)*psidot

    print(f"- - - Substitution - - -\nx_doubledot = vx_dot*cos(psi) - vx*sin(psi)*psidot - vy_dot*sin(psi) - vy*cos(psi)*psidot\ny_doubledot_proj = vx_dot*sin(psi) + vx*cos(psi)*psidot + vy_dot*cos(psi) - vy*sin(psi)*psidot\n")

    _1st_eq_abs = simplify(_1st_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"1st Lagrangian eq, absolute frame, after subs : \n{_1st_eq_abs} = 0\n")

    _2nd_eq_abs = simplify(_2nd_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"2nd Lagrangian eq, absolute frame, after subs : \n{_2nd_eq_abs} = 0\n")

    _3rd_eq_abs = simplify(_3rd_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"3rd Lagrangian eq, absolute frame, after subs : \n{_3rd_eq_abs} = 0\n")

    _4th_eq_abs = simplify(_4th_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"4th Lagrangian eq, absolute frame, after subs : \n{_4th_eq_abs} = 0\n")

    _5th_eq_abs = simplify(_5th_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"5th Lagrangian eq, absolute frame, after subs : \n{_5th_eq_abs} = 0\n\n")

    print(f"- - - Apply rotation to cast the 1st-2nd-3rd eq in the body-ref-frame - - -")
    
    equations_rel = Rz.T*Matrix([[_1st_eq_abs], [_2nd_eq_abs], [_3rd_eq_abs]])
    equations_rel = simplify(equations_rel.subs(Ffx_body, -Ffz_body*mu_fx_body).subs(Ffy_body, -Ffz_body*mu_fy_body).subs(Frx_body, -Frz_body*mu_rx_body).subs(Fry_body, -Frz_body*mu_ry_body), mul=True)

    _1st_eq_rel = equations_rel[0]
    print(f"1st Lagrangian eq, body-ref-frame: \n{_1st_eq_rel} = 0\n")

    _2nd_eq_rel = equations_rel[1]
    print(f"2nd Lagrangian eq, body-ref-frame: \n{_2nd_eq_rel} = 0\n")

    _3rd_eq_rel = equations_rel[2]
    print(f"3rd Lagrangian eq, body-ref-frame: \n{_3rd_eq_rel} = 0\n")

    _4th_eq_rel = _4th_eq_abs
    print(f"4th Lagrangian eq, body-ref-frame: \n{_4th_eq_rel} = 0\n")

    _5th_eq_rel = _5th_eq_abs
    print(f"5th Lagrangian eq, body-ref-frame: \n{_5th_eq_rel} = 0\n\n")


if show_4_wheeled:
    # 4-WHEELED MODEL ##############################################################################
    Rx = Matrix([[1, 0, 0],
            [0, cos(fi), -sin(fi)],
            [0, sin(fi), cos(fi)]])

    Ry = Matrix([[cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)]])

    Rz = Matrix([[cos(psi), -sin(psi), 0],
            [sin(psi), cos(psi), 0],
            [0, 0, 1]]) 
    
    R = Rz*Ry*Rx # rotation matrix from body-ref-frame to abs-ref-frame

    OG_abs = Matrix([[x],[y],[z]]) + R*Matrix([[b],[-d], [-h]]) # (CG - O) in abs-ref-frame 

    v_linear_abs = diff(OG_abs, t) # velocity of the baricenter in the abs-ref-frame
    v_linear_abs = v_linear_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)

    I_body = Matrix([[Ixx, 0, Ixz],
                    [0, Iyy, 0],
                    [Ixz, 0, Izz]]) # inertia matrix of the car WRT the ref-frame centered in the baricenter
    
    w_matrix_abs = diff(R, t)*R.T # skew-symm mat associated to the angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the abs-ref-frame
    w_matrix_abs = w_matrix_abs.subs(Derivative(psi, t), psidot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    w_matrix_abs = simplify(w_matrix_abs)
    w_matrix_body = simplify(R.T*w_matrix_abs*R) # skew-symm mat associated to the angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the body-ref-frame
    w_matrix_body = w_matrix_body.subs(Derivative(psi, t), psidot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    w_body = simplify(Matrix([[w_matrix_body[2,1]],
                        [w_matrix_body[0,2]],
                        [w_matrix_body[1,0]]])) # angular velocity of the body-ref-frame WRT the abs-ref-frame, expressed in the body-ref-frame
    
    # LAGRANGIAN FUNCTION
    kinetic_linear = simplify(0.5*v_linear_abs.T*m*v_linear_abs)
    kinetic_linear = kinetic_linear[0]

    kinetic_rotational = simplify(0.5*w_body.T*I_body*w_body)
    kinetic_rotational = kinetic_rotational[0]

    potential = - m*g*OG_abs[2]

    L = kinetic_linear + kinetic_rotational - potential

    # GENERALIZED FORCES
    ORear_right_abs = Matrix([[x], [y], [z]]) + R*Matrix([[0], [tr], [0]])
    ORear_left_abs = Matrix([[x], [y], [z]]) + R*Matrix([[0], [-tr], [0]])
    OFront_right_abs = Matrix([[x], [y], [z]]) + R*Matrix([[a+b], [tf], [0]])
    OFront_left_abs = Matrix([[x], [y], [z]]) + R*Matrix([[a+b], [-tf], [0]])

    v_rear_right_abs = diff(ORear_right_abs, t) 
    v_rear_right_b = R.T*v_rear_right_abs # velocity of the front wheel expressed in the body-ref-frame
    v_rear_right_b = simplify(v_rear_right_b)

    v_rear_left_abs = diff(ORear_left_abs, t) 
    v_rear_left_b = R.T*v_rear_left_abs # velocity of the front wheel expressed in the body-ref-frame
    v_rear_left_b = simplify(v_rear_left_b)

    v_front_right_abs = diff(OFront_right_abs, t) 
    v_front_right_b = R.T*v_front_right_abs # velocity of the front wheel expressed in the body-ref-frame
    v_front_right_b = simplify(v_front_right_b)

    v_front_left_abs = diff(OFront_left_abs, t) 
    v_front_left_b = R.T*v_front_left_abs # velocity of the front wheel expressed in the body-ref-frame
    v_front_left_b = simplify(v_front_left_b)

    J = Matrix([
        [cos(psi) * cos(theta), sin(psi) * cos(theta), -tf * cos(theta) * cos(fi), -sin(theta), 0, tf * sin(fi)],
        [cos(psi) * sin(theta) * sin(fi) - sin(psi) * cos(fi), sin(psi) * sin(theta) * sin(fi) + cos(psi) * cos(fi), (a+b) * cos(theta) * cos(fi), cos(theta) * sin(fi), 0, -(a+b) * sin(fi)],
        [cos(psi) * cos(theta), sin(psi) * cos(theta), tf * cos(theta) * cos(fi), -sin(theta), 0, -tf * sin(fi)],
        [cos(psi) * sin(theta) * sin(fi) - sin(psi) * cos(fi), sin(psi) * sin(theta) * sin(fi) + cos(psi) * cos(fi), (a+b) * cos(theta) * cos(fi), cos(theta) * sin(fi), 0, (a+b) * sin(fi)],
        [cos(psi) * cos(theta), sin(psi) * cos(theta), -tr * cos(theta) * cos(fi), -sin(theta), 0, tr * sin(fi)],
        [cos(psi) * sin(theta) * sin(fi) - sin(psi) * cos(fi), sin(psi) * sin(theta) * sin(fi) + cos(psi) * cos(fi), 0, cos(theta) * sin(fi), 0, 0],
        [cos(theta) * cos(fi), sin(psi) * cos(theta), tr * cos(theta) * cos(fi), -sin(theta), 0, tr * sin(fi)],
        [cos(psi) * sin(theta) * sin(fi) - sin(psi) * cos(fi), sin(psi) * sin(theta) * sin(fi) + cos(psi) * cos(fi), 0, cos(theta) * sin(fi), 0, 0]
    ]) # manually compute the jacobian from previous relations

    Ffx_right_body, Ffx_left_body, Ffy_right_body, Ffy_left_body, Frx_right_body, Frx_left_body, Fry_right_body, Fry_left_body, Ffz_right_body, Ffz_left_body, Frz_right_body, Frz_left_body, mu_fx_right_body, mu_fx_left_body, mu_fy_right_body, mu_fy_left_body, mu_rx_body, mu_rx_right_body, mu_rx_left_body,  mu_ry_right_body, mu_ry_left_body = symbols('Ffx_right_body Ffx_left_body Ffy_right_body Ffy_left_body Frx_right_body Frx_left_body Fry_right_body Fry_left_body Ffz_right_body Ffz_left_body Frz_right_body Frz_left_body mu_fx_right_body mu_fx_left_body mu_fy_right_body mu_fy_left_body mu_rx_body mu_rx_right_body mu_rx_left_body  mu_ry_right_body mu_ry_left_body') 
    Gen_forces_abs = J.T*Matrix([[Ffx_right_body], [Ffy_right_body], [Ffx_left_body], [Ffy_left_body], [Frx_right_body], [Fry_right_body], [Frx_left_body], [Fry_left_body]])

    print("\n////// 4-WHEELED MODEL \\\\\\\\\\\\\n\n")
    print(f"- - - Generalized forces for Lagrangian equations - - -")
    print(f"Generalized forces, 1st_eq : \n{Gen_forces_abs[0]}\n")
    print(f"Generalized forces, 2nd_eq : \n{Gen_forces_abs[1]}\n")
    print(f"Generalized forces, 3rd_eq : \n{Gen_forces_abs[2]}\n")
    print(f"Generalized forces, 4th_eq : \n{Gen_forces_abs[3]}\n")
    print(f"Generalized forces, 5th_eq : \n{Gen_forces_abs[4]}\n")
    print(f"Generalized forces, 6th_eq : \n{Gen_forces_abs[5]}\n\n")

    # CONSTRAINED FORCES
    A = Matrix([
        [cos(psi)*sin(theta)*cos(fi) + sin(psi)*sin(fi), sin(psi)*sin(theta)*cos(fi) - cos(psi)*sin(fi), -tf*sin(theta) - (a+b)*cos(theta)*sin(fi), cos(theta)*cos(fi), tf, -(a+b)*cos(fi)],
        [cos(psi)*sin(theta)*cos(fi) + sin(psi)*sin(fi), sin(psi)*sin(theta)*cos(fi) - cos(psi)*sin(fi), tf*sin(theta) - (a+b)*cos(theta)*sin(fi), cos(theta)*cos(fi), -tf, -(a+b)*cos(fi)],
        [cos(psi)*sin(theta)*cos(fi) + sin(psi)*sin(fi), sin(psi)*sin(theta)*cos(fi) - cos(psi)*sin(fi), -tr*sin(theta), cos(theta)*cos(fi), tr, 0],
        [cos(psi)*sin(theta)*cos(fi) + sin(psi)*sin(fi), sin(psi)*sin(theta)*cos(fi) - cos(psi)*sin(fi), tr*sin(theta), cos(theta)*cos(fi), -tr, 0]
    ])

    Constrained_forces_abs = A.T*Matrix([[-Ffz_right_body], [-Ffz_left_body], [-Frz_right_body], [-Frz_left_body]])

    print(f"- - - Constrained forces for Lagrangian equations - - -")
    print(f"Constrained forces, 1st_eq : \n{Constrained_forces_abs[0]}\n")
    print(f"Constrained forces, 2nd_eq : \n{Constrained_forces_abs[1]}\n")
    print(f"Constrained forces, 3rd_eq : \n{Constrained_forces_abs[2]}\n")
    print(f"Constrained forces, 4th_eq : \n{Constrained_forces_abs[3]}\n")
    print(f"Constrained forces, 5th_eq : \n{Constrained_forces_abs[4]}\n")
    print(f"Constrained forces, 6th_eq : \n{Constrained_forces_abs[5]}\n\n")

    print(f"- - - Lagrangian expressed in the absolute frame - - -\nL = {L}\n")

    # LAGRANGIAN EQUATIONS
    dL_dx = diff(L, x)
    dL_dxdot = diff(L, xdot)
    Dt_dL_dxdot = diff(dL_dxdot, t)
    _1st_eq = Dt_dL_dxdot - dL_dx

    _1st_eq_abs = _1st_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _1st_eq_abs= _1st_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _1st_eq_abs = _1st_eq_abs - Gen_forces_abs[0] + Constrained_forces_abs[0]
    _1st_eq_abs = simplify(_1st_eq_abs)
    _1st_eq_abs = _1st_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)

    print(f"1st Lagrangian eq, absolute frame : \n{_1st_eq_abs} = 0\n")

    dL_dy = diff(L, y)
    dL_dydot = diff(L, ydot)
    Dt_dL_dydot = diff(dL_dydot, t)
    _2nd_eq = Dt_dL_dydot - dL_dy

    _2nd_eq_abs = _2nd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _2nd_eq_abs= _2nd_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _2nd_eq_abs = _2nd_eq_abs - Gen_forces_abs[1] + Constrained_forces_abs[1]
    _2nd_eq_abs = simplify(_2nd_eq_abs)
    _2nd_eq_abs = _2nd_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)

    print(f"2nd Lagrangian eq, absolute frame : \n{_2nd_eq_abs} = 0\n")

    dL_dpsi = diff(L, psi)
    dL_dpsidot = diff(L, psidot)
    Dt_dL_dpsidot = diff(dL_dpsidot, t)
    _3rd_eq = Dt_dL_dpsidot - dL_dpsi

    _3rd_eq_abs = _3rd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _3rd_eq_abs= _3rd_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _3rd_eq_abs = _3rd_eq_abs - Gen_forces_abs[2] + Constrained_forces_abs[2]
    _3rd_eq_abs = simplify(_3rd_eq_abs)
    _3rd_eq_abs = _3rd_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)

    print(f"3rd Lagrangian eq, absolute_frame : \n{_3rd_eq_abs} = 0\n")

    dL_z = diff(L, z)
    dL_zdot = diff(L, zdot)
    Dt_dL_zdot = diff(dL_zdot, t)
    _4th_eq_abs = Dt_dL_zdot - dL_z

    _4th_eq_abs = _4th_eq_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _4th_eq_abs= _4th_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _4th_eq_abs = _4th_eq_abs - Gen_forces_abs[3] + Constrained_forces_abs[3]
    _4th_eq_abs = simplify(_4th_eq_abs)
    _4th_eq_abs = _4th_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)

    print(f"4th Lagrangian eq, absolute_frame : \n{_4th_eq_abs} = 0\n")

    dL_theta = diff(L, theta)
    dL_thetadot = diff(L, thetadot)
    Dt_dL_thetadot = diff(dL_thetadot, t)
    _5th_eq_abs = Dt_dL_thetadot - dL_theta

    _5th_eq_abs = _5th_eq_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _5th_eq_abs= _5th_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _5th_eq_abs = _5th_eq_abs - Gen_forces_abs[4] + Constrained_forces_abs[4]
    _5th_eq_abs = simplify(_5th_eq_abs)
    _5th_eq_abs = _5th_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)
    
    print(f"5th Lagrangian eq, absolute_frame : \n{_5th_eq_abs} = 0\n")

    dL_fi = diff(L, fi)
    dL_fidot = diff(L, fidot)
    Dt_dL_fidot = diff(dL_fidot, t)
    _6th_eq_abs = Dt_dL_fidot - dL_fi

    _6th_eq_abs = _6th_eq_abs.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(z, t), zdot).subs(Derivative(theta, t), thetadot).subs(Derivative(fi, t), fidot)
    _6th_eq_abs= _6th_eq_abs.subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot).subs(Derivative(zdot, t), z_doubledot).subs(Derivative(thetadot, t), theta_doubledot).subs(Derivative(fidot, t), fi_doubledot)

    _6th_eq_abs = _6th_eq_abs - Gen_forces_abs[5] + Constrained_forces_abs[5]
    _6th_eq_abs = simplify(_6th_eq_abs)
    _6th_eq_abs = _6th_eq_abs.subs(z, 0).subs(theta, 0).subs(zdot, 0).subs(thetadot, 0).subs(z_doubledot, 0).subs(theta_doubledot, 0).subs(fi, 0).subs(fidot, 0).subs(fi_doubledot, 0)
    
    print(f"6th Lagrangian eq, absolute_frame : \n{_6th_eq_abs} = 0\n\n")

    # CAST EQUATIONS IN BODY-REF_FRAME
    x_doubledot_proj = vx_dot*cos(psi) - vx*sin(psi)*psidot - vy_dot*sin(psi) - vy*cos(psi)*psidot
    y_doubledot_proj = vx_dot*sin(psi) + vx*cos(psi)*psidot + vy_dot*cos(psi) - vy*sin(psi)*psidot

    print(f"- - - Substitution - - -\nx_doubledot = vx_dot*cos(psi) - vx*sin(psi)*psidot - vy_dot*sin(psi) - vy*cos(psi)*psidot\ny_doubledot_proj = vx_dot*sin(psi) + vx*cos(psi)*psidot + vy_dot*cos(psi) - vy*sin(psi)*psidot\n")

    _1st_eq_abs = simplify(_1st_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"1st Lagrangian eq, absolute frame, after subs : \n{_1st_eq_abs} = 0\n")

    _2nd_eq_abs = simplify(_2nd_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"2nd Lagrangian eq, absolute frame, after subs : \n{_2nd_eq_abs} = 0\n")

    _3rd_eq_abs = simplify(_3rd_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"3rd Lagrangian eq, absolute frame, after subs : \n{_3rd_eq_abs} = 0\n")

    _4th_eq_abs = simplify(_4th_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"4th Lagrangian eq, absolute frame, after subs : \n{_4th_eq_abs} = 0\n")

    _5th_eq_abs = simplify(_5th_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"5th Lagrangian eq, absolute frame, after subs : \n{_5th_eq_abs} = 0\n")

    _6th_eq_abs = simplify(_6th_eq_abs.subs(x_doubledot, x_doubledot_proj).subs(y_doubledot, y_doubledot_proj))
    print(f"6th Lagrangian eq, absolute frame, after subs : \n{_6th_eq_abs} = 0\n\n")

    print(f"- - - Apply rotation to cast the 1st-2nd-3rd eq in the body-ref-frame - - -")
    equations_rel = Rz.T*Matrix([[_1st_eq_abs], [_2nd_eq_abs], [_3rd_eq_abs]])
    equations_rel = simplify(equations_rel.subs(Ffx_body, -Ffz_body*mu_fx_body).subs(Ffy_body, -Ffz_body*mu_fy_body).subs(Frx_body, -Frz_body*mu_rx_body).subs(Fry_body, -Frz_body*mu_ry_body), mul=True)

    _1st_eq_rel = equations_rel[0]
    print(f"1st Lagrangian eq, body-ref-frame: \n{_1st_eq_rel} = 0\n")

    _2nd_eq_rel = equations_rel[1]
    print(f"2nd Lagrangian eq, body-ref-frame: \n{_2nd_eq_rel} = 0\n")

    _3rd_eq_rel = equations_rel[2]
    print(f"3rd Lagrangian eq, body-ref-frame: \n{_3rd_eq_rel} = 0\n")

    _4th_eq_rel = _4th_eq_abs
    print(f"4th Lagrangian eq, body-ref-frame: \n{_4th_eq_rel} = 0\n")

    _5th_eq_rel = _5th_eq_abs
    print(f"5th Lagrangian eq, body-ref-frame: \n{_5th_eq_rel} = 0\n")

    _6th_eq_rel = _6th_eq_abs
    print(f"6th Lagrangian eq, body-ref-frame: \n{_6th_eq_rel} = 0\n")

    _7th_eq_rel = -c_front_right*tr*Ffz_right_body + c_front_left*tr*Ffz_left_body + c_rear_right*tf*Frz_right_body - c_rear_left*tf*Frz_left_body
    print(f"7th Lagrangian eq, body-ref-frame: \n{_7th_eq_rel} = 0\n")























# q_doubledot = Matrix([[x_doubledot], [y_doubledot], [psi_doubledot], [Ffz_body], [Frz_body]])

# dynamics_equations_abs = M*q_doubledot + C + G - Gen_forces_abs





# M = Matrix([[m, 0, -m*b*sin(psi)],
#             [0, m, m*b*cos(psi)],
#             [-m*b*sin(psi), m*b*cos(psi), Izz + m*b**2]])

# C = Matrix([[-m*b*cos(psi)*psidot**2], [-m*b*sin(psi)*psidot**2], [0]])

# G = Matrix([[0], [0], [0]])



# subs = R.T*Matrix([[vx_dot - vy*psidot], [vy_dot + vx*psidot], [psi_doubledot]])

# q_doubledot = Matrix([[subs[0]], [subs[1]], [subs[2]], [Ffz_body], [Frz_body]])

# dynamics_equations_rel = simplify(M*q_doubledot + C + G - Gen_forces_abs.subs(Ffx_body, -Ffz_body*mu_fx_body).subs(Ffy_body, -Ffz_body*mu_fy_body).subs(Frx_body, -Frz_body*mu_rx_body).subs(Fry_body, -Frz_body*mu_ry_body))

# RR = Matrix([[R[0], R[1], R[2], 0, 0], [R[3], R[4], R[5], 0, 0], [R[6], R[7], R[8], 0, 0], [R[0], R[1], R[2], 0, 0], [R[3], R[4], R[5], 0, 0]])
# print(RR)
# dynamics_equations_rel = simplify(RR*dynamics_equations_rel)

# print("1_\n\n ", dynamics_equations_rel[0])
# print("2_\n\n ", dynamics_equations_rel[1])
# print("3_\n\n ", dynamics_equations_rel[2])
# print("4_\n\n ", dynamics_equations_rel[3])
# print("5_\n\n ", dynamics_equations_rel[4])

# Gen_forces_body = Matrix([[Ffx_body], [Ffy_body], [Frx_body], [Fry_body]])

# _1st_eq_body = simplify(_1st_eq_body - (Ffx_body + Frx_body))
# _1st_eq_body = _1st_eq_body.subs(Ffx_body, -Ffz_body*mu_fx_body).subs(Ffy_body, -Ffz_body*mu_fy_body).subs(Frx_body, -Frz_body*mu_rx_body).subs(Fry_body, -Frz_body*mu_ry_body)
# _2nd_eq_body = simplify(_2nd_eq_body - (Ffy_body + Fry_body))
# _2nd_eq_body = _2nd_eq_body.subs(Ffx_body, -Ffz_body*mu_fx_body).subs(Ffy_body, -Ffz_body*mu_fy_body).subs(Frx_body, -Frz_body*mu_rx_body).subs(Fry_body, -Frz_body*mu_ry_body)
# _3rd_eq_body = simplify(_3rd_eq_body)

# print(f"1st_eq, body frame : {_1st_eq_body}\n\n")
# print(f"2nd_eq, body frame : {_2nd_eq_body}\n\n")
# print(f"3rd_eq, body frame : {_3rd_eq_body}\n\n")

# 4-WHEELED MODEL ##############################################################################
# I_body = Matrix([[Ixx, 0, Ixz], [0, Iyy, 0], [Ixz, 0, Izz]])
# R = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
# w = Matrix([[0], [0], [psidot]])

# kinetic_linear = 0.5*m*((xdot - b*sin(psi)*psidot - d*cos(psi)*psidot)**2 + (ydot + b*cos(psi)*psidot - d*sin(psi)*psidot)**2)
# kinetic_rotational = 0.5*w.T*R.T*I_body*R*w
# kinetic_rotational = kinetic_rotational[0]
# potential = -m*g*h
# L = kinetic_linear + kinetic_rotational - potential

# print("\n/// 4-WHEELED MODEL \\\\\\\n")
# print(f"Lagrangian expressed in the absolute frame: {L}\n")

# dL_dx = diff(L, x)
# dL_dxdot = diff(L, xdot)
# Dt_dL_dxdot = diff(dL_dxdot, t)
# _1st_eq = Dt_dL_dxdot - dL_dx
# _1st_eq = _1st_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot)
# _1st_eq = simplify(_1st_eq)
# print(f"1st_eq : {_1st_eq}\n\n")

# dL_dy = diff(L, y)
# dL_dydot = diff(L, ydot)
# Dt_dL_dydot = diff(dL_dydot, t)
# _2nd_eq = Dt_dL_dydot - dL_dy
# _2nd_eq = _2nd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot)
# _2nd_eq = simplify(_2nd_eq)
# print(f"2nd_eq : {_2nd_eq}\n\n")

# dL_dpsi = diff(L, psi)
# dL_dpsidot = diff(L, psidot)
# Dt_dL_dpsidot = diff(dL_dpsidot, t)
# _3rd_eq = Dt_dL_dpsidot - dL_dpsi
# _3rd_eq = _3rd_eq.subs(Derivative(x, t), xdot).subs(Derivative(y, t), ydot).subs(Derivative(psi, t), psidot).subs(Derivative(xdot, t), x_doubledot).subs(Derivative(ydot, t), y_doubledot).subs(Derivative(psidot, t), psi_doubledot)
# _3rd_eq = simplify(_3rd_eq)
# print(f"3rd_eq : {_3rd_eq}\n\n")

