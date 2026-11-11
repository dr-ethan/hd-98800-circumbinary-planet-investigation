#body problem
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import math
import random
import multiprocessing


#-------------------------------Functions-----------------------------------

def InitialConditions(x,y,z,bodies):
    '''
    Function that generates a dictionary with each list inside being the vector [x,y,z] components of the body
    Any vector can be inserted as long as all components have the same length 
    '''
    if len(x) != bodies or len(y) != bodies or len(z) != bodies:
        raise ValueError("Vector count mismatch! Make sure there is an equal amount of vectors for each body!")
    body_information = {}
    for vector in range(bodies):
        body_information[f'body_{vector}'] = np.array([x[vector],y[vector],z[vector]])
    return body_information

def Grav_Accel_eq(i_pos,j_pos,mass_j): #the attractive force of body i and j. it is the RHS of the equation.
    '''
    finds gravitational acceleration towards j for object i
    ---
    returns:
        acceleration towards body j. 
            format: vector [x,y,z]  
    '''
    seperation = j_pos-i_pos
    distance = np.linalg.norm(seperation)
    unit_vector_direction = seperation/distance
    acceleration = Grav*(mass_j)*unit_vector_direction / (distance**2)
    return acceleration

def ivp_function(t,bodies,state,masses):
    '''
    ivp_function for finding differential equation with the acceleration,velocity,positions (i.e. 1st ODE)

    ---
    vars
        state - list containing all position and velocity information
            ex: [x1,y1,z1,x2,y2,z2,Vx1,Vy1,Vz1,Vx2,Vy2,Vz2]
        t - int, time step no.
        bodies - int, number of bodies in the problem
        masses - list of all masses in the system 
    '''
    vector_positions = np.array(state[:bodies*3]).flatten() #of form [r1,r2,r3] for 3 bodies with r = x,y,z
    vector_velocities = np.array(state[3*bodies:]).flatten() #of form [v1,v2,v3] for 3 bodies with v = vx,vy,vz
    acceleration_of_body = np.zeros((bodies,3))#bodiesx3 shape matrix
    for i in range(bodies):
        for j in range(bodies):
            if i != j: #performing for loops in this way compares each object to all others once without repeating 
                j_pos = vector_positions[j*3:j*3+3] #sets j = x,y,z
                i_pos = vector_positions[i*3:i*3+3] #sets i = x,y,z
                acceleration_of_body[i] += Grav_Accel_eq(i_pos,j_pos,masses[j])
    #print(acceleration_of_body)
    differential_state = np.hstack([vector_velocities,acceleration_of_body.flatten()])
    return differential_state

def all_collision_events(t,state,radii,bodies):
    '''
    this function is used in the solve_IVP function to signal when either 
        a. one of the objects collides into eachother
        b. the planet travels outside its orbital range (> 6.5 AU)
    
    t - time slice
    state - vector array containing all position and velocity data
    radii - list of radii of each body
    bodies - int, amount of bodies in the problem
    '''
    vector_positions = state[:bodies*3].reshape(bodies,3)
    for i in range(bodies):
        for j in range(bodies):
            if i != j:
                if np.linalg.norm(vector_positions[j]-vector_positions[i]) < radii[i]+radii[j]:
                    return 0
                elif i == 4 and j != 0 and j != 1: #if planet and j is not system A:
                    if np.linalg.norm(vector_positions[j]-vector_positions[i]) > 6.5*AU: #if the planet's separation between Ba or Bb is greater than 8 AU:
                        return 0 #it is considered lost and terminate the iteration.
    return 1


def vis_viva_eq(Total_Mass,r,a):
    '''
    finds velocity at point r for desired body in M_1 ~= M_2 systems
    needs to be scaled with COM.
    ---
    parameters:
        M - mass of body the desired object orbits
        r - distance from COM
        a - semi-major axis
    '''
    return math.sqrt(Grav*(Total_Mass)*(2/r - 1/a))

def normal_dist(mean_list,error_list):
    '''
    function that uses normal distribution to randomise initial conditions.
    ---
    vars
        mean_list - list of values
        error_list - list of error for those values
    '''
    if len(mean_list) != len(error_list):
        raise ValueError("Lengths of lists do not match")
    values =[]
    for element in range(len(mean_list)):
        values.append(np.random.normal(loc=mean_list[element],scale=error_list[element]))
    return values

def semi_major_calc(T,M):
    '''calculate semi-major axis from time period and mass'''
    return np.cbrt(T**2 * Grav * M /(4*math.pi**2))

def semi_major_for_smaller_bodies(body_mass,orbiting_body_mass,total_semi_major):
    '''computes the semi major for an object with respect to its orbital barycenter'''
    return orbiting_body_mass/(body_mass+orbiting_body_mass)*total_semi_major

def separation_from_COM(semi_major_body_1,eccentricity,theta):
    '''position of body_1 in orbit of binary based on a_1,e and theta_1
    we assume theta to be 180 or 0 to make initial conditions easier to compute
    '''
    return semi_major_body_1*(1-eccentricity**2)/(1+eccentricity*np.cos(theta))

def orbital_vel_formula(M,r):
    '''v^2 = GM/r. computes velocity for a satelite object'''
    return np.sqrt(Grav*M/r)

def separation_of_2_bodies_over_runtime(body_1,body_2):
    '''
    compute the separation of two bodies over the entire simulation's runtime and plots it
    body_1, body_2 of form [[x],[y],[z]]
    '''
    body_1 = np.array(body_1)
    body_2 = np.array(body_2)
    sep_x,sep_y,sep_z = [(body_1[j]-body_2[j]).flatten() for j in range(3)]
    sep =[]
    for j in range(len(sep_x)):
        sep.append(np.linalg.norm([sep_x[j],sep_y[j],sep_z[j]]))
    return sep
#---------Global Constants------
global Grav #Gravitational Constant
Grav = 6.67430e-11
global AU #AU in meters
AU = 147.1e9
global parsec #parsec in meters
parsec = 648000*AU/math.pi

global plotting
global saving
global true_statement
global multiple_iteration
global save_current_run
global get_previous_run
global from_t0

global hour
global day
global year
year = 365*24*3600 #seconds in a year
hour = 60**2
day = hour*24


#-------------Loop Settings-------------

plotting = True #plot the results
saving = False #save the initial seed
multiple_iteration = False #do more than one loop
#For multiple iterations, I warn you, make sure each iteartion is a short period of time. This program takes up a lot of RAM and will quickly run out if each iteration takes 1~ GiB of RAM
#In summary, Only perform multiple iteration for short-lived setup (~10years)
from_t0 = True

#-------------Save and Load Data Settings-------------
save_current_run = False
save_directory = ""
get_previous_run = False
load_directory = "SAVES.npz"



dt = 200 #step size for IVP solver. Usually ok to keep 50-500 seconds for rough searching of systems
Stop_Time_Global = year*130 #Stop simulation after this much time passes (in seconds)
skip_frames = hour*5 #only important if plotting - it reduces the amount of data points for matplotlib to load easier - this value only takes every X value from final position list
Initial_Seed = None #set to None for no error, set to number for specific seed, set to "Random" for random seed
Stop_Time_Per_Iteration = None #set to None if running one sim

true_statement = True #for the while loop - don't change
iteration = 1 #keeps track of iteration - don't change

if year < Stop_Time_Global/dt: print("\nWarning! Data amount will take up a lot of virtual memory! Ensure your device has suitable hardware for this computation!") 
#Good timestep and stop_time relation has a stop_time/dt value of about a year's worth of datapoints (~31.6 Million datapoints)
#31.6 mil time datapoints is suitable for a device with ~8GiB RAM
#my device has 32GiB and about 24GiB allocated for Python for which i have a Stop_Time/dt ratio of 3.2 years worth of data (~100M time points)- which is the max my device can handle

while true_statement: #we create a while loop to iterate through different seeds for different orbital configurations (slight mass variations, planet separation, time periods. this is because of the uncertainty in their readings)
     
    #Setting datatype equal to np.float32 instead of np.float64
    #--------------------------------------- Constants -------------------------------------
    # For quick referencing during setup to save time
    #grav constant
    Earth = 5.972e24
    Sun = 1.989e30
    Earth_Radius = 6378e3
    Sun_Radius = 696340e3
    Jupiter_Radius = 11.2*Earth_Radius
    Jupiter_Mass = 317.8*Earth


    #----------Masses & Error------------
    #Sources are cited in the report
    Mass_Aa = 0.93*Sun
    Mass_Ab = 0.29*Sun
    Mass_Aa_error,Mass_Ab_error = 0.09,0.02
    Mass_Ba = 0.77*Sun
    Mass_Bb = 0.62*Sun
    Mass_Ba_error,Mass_Bb_error = 0.04,0.02
    Planet_Mass = Jupiter_Mass#assuming the planet has a mass similar to Jupiter to clear its orbit of 2-5 AU
    Planet_Radii_error,Planet_Mass_error = 0,0 

    #------------Semi Major Axes & Error--------------
    #Sources are cited in the report
    Ba_Bb_semi_major = 1.01*AU
    Ba_Bb_semi_major_error = 0.01*AU
    Aa_Ab_semi_major = 0.86*AU
    Aa_Ab_semi_major_error = 0.02*AU

    Ba_Bb_eccent = 0.805
    Ba_Bb_eccent_error = 0.0005
    Aa_Ab_eccent = 0.4808
    Aa_Ab_eccent_error = 0.0008
    A_B_eccent = 0.5

    #-----------List Assigning------------------------
    #Sources are cited in the report

    Solar_Radii = [Jupiter_Radius, #Aa
                1.75*Sun_Radius, #Ab
                1.09*Sun_Radius, #Ba
                0.85*Sun_Radius, #Bb
                Jupiter_Radius #Planet
                ] 
    #The lists have a specific format of: [Aa,Ab,Ba,Bb,planet]
    #There is no specific purpose for this type of layout - its just how I coded it in. Other ways definetly work but it doesn't really matter
    Solar_Radii_error = [0, #Aa
                        0, #Ab
                        0.14*Sun_Radius, #Ba
                        0.11*Sun_Radius, #Bb
                        Planet_Radii_error #planet
                        ]

    Solar_Masses = [Mass_Aa, 
                    Mass_Ab,
                    Mass_Ba,
                    Mass_Bb,
                    Planet_Mass
                    ]

    Solar_Masses_error = [Mass_Aa_error,
                    Mass_Ab_error,
                    Mass_Ba_error,
                    Mass_Bb_error,
                    Planet_Mass_error
                    ]

    if Initial_Seed == "Random":
        Initial_Seed = random.randrange(0,1000000) #randomly generate a value
        random.seed(Initial_Seed)
    elif Initial_Seed == type(int):
        random.seed(Initial_Seed) #use the randomly generated value as the seed
    #good seed list quick reference:
    '''9945, 239001'''
    #bad seeds
    '''3198, 987004'''

    #------------Calculating Normally Distributed Values for variation in orbital parameters--------------------------------

    if Initial_Seed != None: 
        body_radi = normal_dist(Solar_Radii,Solar_Radii_error) #these will be checked to make sure no collisions during calculations
        Mass_Aa,Mass_Ab,Mass_Ba,Mass_Bb,Planet_Mass = normal_dist(Solar_Masses,Solar_Masses_error) #vary the masses slightly 
    else:
        body_radi = Solar_Radii
    masses = [Mass_Aa,Mass_Ab,Mass_Ba,Mass_Bb,Planet_Mass] #these will be called when performing force calculations
    bodies = 5
    #--------Calculating A B orbital parameters from period, mass and eccentricity - references in report------------#
    
    #B is the companion of A, A has velocity in +V_A and B has velocity -V_B
    #In B, Bb is the companion of Ba, Ba has velocity +V_Ba and Bb has velocity -V_Bb, which also has velocity -V_B for B and A orbit
    #In A, Ab is the companion of Aa, Aa has velocity +V_Aa and Ab has velocity -V_Ab, which also has velocity V_A for A and B orbit
    #We also make the assumption that all bodies are at their apastron and on coordinate axes for easy IC setup so velocity and acceleration are perpendicular.
    #Inserting the planet is easy as we assume B's COM is just a single body and use v^2=GM/r
    
    #--------------------------------Load data code -------------------------------------------
    if get_previous_run and not from_t0: #Get data and not randomly generate starting conditions
        if load_directory == None or load_directory == "": from_t0= True
        else:
            load_data = np.load(load_directory, allow_pickle=True)
            positions = load_data["positions"].item()
            velocities = load_data["velocities"].item()
            x_values =[]
            y_values =[]
            z_values =[]
            vx_values =[]
            vy_values = []
            vz_values =[]
                #the structure of loaded list is {"Body_1_pos":[[BODY_1X],[BODY_1Y],[BODY_1Z]], "Body_2_pos": [...]}
            for identifier in positions:
                x_values.append(positions[identifier][0][0][-1]) #take last reading to use as the initial condition
                y_values.append(positions[identifier][1][0][-1])
                z_values.append(positions[identifier][2][0][-1])
            for identifier in velocities:
                vx_values.append(velocities[identifier][0][0][-1]) #take last reading to use as the initial condition
                vy_values.append(velocities[identifier][1][0][-1])
                vz_values.append(velocities[identifier][2][0][-1])

            

    if from_t0: #if start from t=0 is True and get previous run is True:
        from_t0 = False #Make start from t=0 False
        #and begin generating the inital conditions
        Mass_A,Mass_B = Mass_Aa+Mass_Ab,Mass_Ba+Mass_Bb
        if Initial_Seed != None: A_B_orbit_time = np.random.normal(loc=246*year,scale=10*year) #Gaussian distribute around 246 plus minus 10 years, source in report
        else: A_B_orbit_time = 246*year
        A_B_semi_major = semi_major_calc(A_B_orbit_time,np.sum(masses[:-1])) #exclude planet mass it is negligible #calculating semi_major axis
        A_semi_major = Mass_A/(Mass_A+Mass_B)*A_B_semi_major
        B_semi_major = Mass_B/(Mass_A+Mass_B)*A_B_semi_major


        A_pos = separation_from_COM(A_semi_major,A_B_eccent,theta=180) #theta = 180 for max distance from COM - velocity is perpendicular at this point in orbit and acceleration is perp to velocity going to COM
        B_pos = -separation_from_COM(B_semi_major,A_B_eccent,theta=180)

        AB_speed_orbit = vis_viva_eq(Total_Mass=Mass_A+Mass_B,r=(A_pos+abs(B_pos)),a=A_B_semi_major)
        A_speed = Mass_B*AB_speed_orbit/(Mass_A+Mass_B)
        B_speed = -Mass_A*AB_speed_orbit/(Mass_A+Mass_B)

        #----------------Ba Bb parameter setup-----------------------
        if Initial_Seed != None:
            Ba_Bb_eccent = np.random.normal(loc=Ba_Bb_eccent,scale=Ba_Bb_eccent_error)
            Ba_Bb_semi_major = np.random.normal(loc=Ba_Bb_semi_major,scale=Ba_Bb_semi_major_error)

        Ba_semi_major = semi_major_for_smaller_bodies(body_mass=Mass_Ba, orbiting_body_mass=Mass_Bb,total_semi_major=Ba_Bb_semi_major)
        Bb_semi_major = semi_major_for_smaller_bodies(body_mass=Mass_Bb, orbiting_body_mass=Mass_Ba,total_semi_major=Ba_Bb_semi_major)


        Ba_pos = separation_from_COM(Ba_semi_major,Ba_Bb_eccent,theta=180)
        Bb_pos = -separation_from_COM(Bb_semi_major,Ba_Bb_eccent,theta=180)

        Ba_Bb_speed_orbit = vis_viva_eq(Total_Mass=(Mass_Ba+Mass_Bb),r=(Ba_pos+abs(Bb_pos)),a=Ba_Bb_semi_major)
        Ba_speed = Mass_Bb*Ba_Bb_speed_orbit/(Mass_Ba+Mass_Bb)
        Bb_speed = -Mass_Ba*Ba_Bb_speed_orbit/(Mass_Ba+Mass_Bb)
        #-----------------Aa Ab parameter setup-------------------
        if Initial_Seed != None:
            Aa_Ab_eccent = np.random.normal(loc=Aa_Ab_eccent,scale=Aa_Ab_eccent_error)
            Aa_Ab_semi_major = np.random.normal(loc=Aa_Ab_semi_major,scale=Aa_Ab_semi_major_error)

        Aa_semi_major = semi_major_for_smaller_bodies(body_mass=Mass_Aa, orbiting_body_mass=Mass_Ab,total_semi_major=Aa_Ab_semi_major)
        Ab_semi_major = semi_major_for_smaller_bodies(body_mass=Mass_Ab, orbiting_body_mass=Mass_Aa,total_semi_major=Aa_Ab_semi_major)

        Aa_pos = separation_from_COM(Aa_semi_major,Aa_Ab_eccent,theta=180)
        Ab_pos = -separation_from_COM(Ab_semi_major,Aa_Ab_eccent,theta=180)

        Aa_Ab_speed_orbit = vis_viva_eq(Total_Mass=(Mass_Aa+Mass_Ab),r=(Aa_pos+abs(Ab_pos)),a=Aa_Ab_semi_major)
        Aa_speed = Mass_Ab*Aa_Ab_speed_orbit/(Mass_Aa+Mass_Ab)
        Ab_speed = -Mass_Aa*Aa_Ab_speed_orbit/(Mass_Aa+Mass_Ab)

        #---------Final Calculation of Parameters----------------
        #We now need to combine all values together
        #let A COM and B COM be aligned on the X axis initially
        #let Aa and Ab be aligned on the Y axis initially
        #let Ba and Bb be aligned on the Y axis initially
        #let the planet be aligned to the COM of B on the Z axis initially
        #Add velocity of A and B orbit to components of A and B so they also flow
        Aa_speed_y_axis = A_speed
        Ab_speed_y_axis = A_speed
        Ba_speed_y_axis = B_speed
        Bb_speed_y_axis = B_speed

        #note that negative values have already been applied in earlier calculations (making the COM of A & B be 0,0,0 means they travel in different directions)
        #Positions already assigned to Aa,Ab,Ba,Bb are all on the Y axis under the variables "I_pos" with I being the body. example: Ab_pos
        Planet_position_x_axis = B_pos
        if Initial_Seed != None: planet_pos = random.randrange(2,6)*AU
        else: planet_pos = 4*AU
        planet_speed = orbital_vel_formula(M=Mass_B,r=planet_pos)
        #planet orbit direction can either be in X or Y axis initially, I chose X axis so the orbit of A and B does not influence it's IC
        planet_speed_y_axis = B_speed

        # ------------------Body Problem Customisation-----------------------
        # Change body count and their initial conditions 






        bodies = 5
        #[Aa,Ab,Ba,Bb,Planet]
        #index of coordinate value correlates to the body number we are giving that positon. example: Aa is first so x_values[0] is the x value of Aa's initial starting value.
        x_values = [A_pos,
                    A_pos,
                    B_pos,
                    B_pos,
                    Planet_position_x_axis
                    ]

        y_values = [Aa_pos,
                    Ab_pos,
                    Ba_pos,
                    Bb_pos,
                    0
                    ]

        z_values = [0,
                    0,
                    0,
                    0,
                    planet_pos,
                    ] 

        vx_values = [Aa_speed,
                    Ab_speed,
                    Ba_speed,
                    Bb_speed,
                    planet_speed]

        vy_values = [Aa_speed_y_axis,
                    Ab_speed_y_axis,
                    Ba_speed_y_axis,
                    Bb_speed_y_axis,
                    planet_speed_y_axis
                    ]

        vz_values = [0,
                    0,
                    0,
                    0,
                    0]



    body_positions = InitialConditions(x_values,y_values,z_values,bodies)
    body_velocities = InitialConditions(vx_values,vy_values,vz_values,bodies)
    #IC gives the positions and velocities in the order of [Aa,Ab,Ba,Bb,Planet]

    # --------Intermediate Calcs for IVP solver-----------
    # Do not change this code it runs the IVP
    if int(skip_frames/bodies) != 0: skip_frames = int(skip_frames/bodies)

    if Stop_Time_Per_Iteration != None: Stop_Time = Stop_Time_Per_Iteration
    else: Stop_Time = Stop_Time_Global

    time_span = (0,Stop_Time)
    time_points = np.linspace(0,Stop_Time,int(Stop_Time/dt)+1)

    initial_state = [] #for IVP later
    vector_position =[]
    vector_velocity =[]

    for identifier,vector in body_positions.items():
        initial_state.append(vector) #takes the vectors [x,y,z] for the respective body from the dictionary for later
        vector_position.append(vector)

    for identifier,vector in body_velocities.items():
        initial_state.append(vector)
        vector_velocity.append(vector)

    initial_state = np.concatenate(initial_state,dtype=np.float32)
    state = np.concatenate([vector_position,vector_velocity],dtype=np.float32) 

    all_collision_events.terminal = True
    sols = solve_ivp(
        fun=lambda t, state: ivp_function(t,bodies,state,masses),
        t_span=time_span,
        y0=initial_state,
        method="RK23",
        t_eval=time_points,
        events=lambda t,state: all_collision_events(t,state,body_radi,bodies),
        dense_output=False
    )

    if len(sols.t_events[0]) > 0: print(sols.t_events[0][0]/(year))
    all_vector_positions = []
    all_vector_velocities = []
    for i in range(len(sols.y)): #len of sols.y = 6*bodies: always even so this seperation will work.
        if i >= int(len(sols.y)/2): all_vector_velocities.append(sols.y[i])
        else: all_vector_positions.append(sols.y[i])
        #sols.y comes in form [Body 1X, Body 1Y, Body 1Z, Body 2X, ..., Body NZ, Body 1 Vx, Body 1 Vy, ...]
        #separate positions from speeds
        #output body number depends on input order of body information

    #---------------------------- Unpacking sols.y -------------------------
    positions_body_based = {f"Body_{i+1}_pos": [[],[],[]] for i in range(bodies)}
    velocities_body_based = {f"Body_{i+1}_vel": [[],[],[]] for i in range(bodies)}
    positions_body_based_skipped = {f"Body_{i+1}_pos": [[],[],[]] for i in range(bodies)}
    velocities_body_based_skipped = {f"Body_{i+1}_vel": [[],[],[]] for i in range(bodies)}
    for i in range(bodies):
        for j in range(3):  # For x, y, z
                positions_body_based[f"Body_{i+1}_pos"][j].append(all_vector_positions[i*3+ j])
                velocities_body_based[f"Body_{i+1}_vel"][j].append(all_vector_velocities[i*3 + j])
                positions_body_based_skipped[f"Body_{i+1}_pos"][j].append(all_vector_positions[i*3+ j][::skip_frames])
                velocities_body_based_skipped[f"Body_{i+1}_vel"][j].append(all_vector_velocities[i*3+j][::skip_frames])

    if plotting: #if plotting is enabled
        #-----------------------Randomly generate colours for 3D plot
        def color_creator(rs,bs,gs,rf,bf,gf,steps):
            #rs,rf,bs,bf,gs,gf = rs/255,rf/255,bs/255,bf/255,gs/255,gf/255
            colors = [
            (rs + (rf - rs) * i / (steps - 1), bs + (bf - bs) * i / (steps - 1), gs + (gf - gs) * i / (steps - 1))
            for i in range(steps)]
            return colors

        colours = {f"Color_{i+1}": [] for i in range(bodies)}
        for i in range(bodies):
            colours[f"Color_{i+1}"] = color_creator(random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),len(positions_body_based_skipped[f"Body_{i+1}_pos"][0][0]))
        # --------------------- Plot 3D data to visualise orbit------------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Axis/m')
        ax.set_ylabel('Y Axis/m')
        ax.set_zlabel('Z Axis/m')
        ax.set_title("3D Plot of Trajectories of Objects in HD 98800")
        for i in range(bodies):
            ax.scatter(positions_body_based_skipped[f"Body_{i+1}_pos"][0],positions_body_based_skipped[f"Body_{i+1}_pos"][1],positions_body_based_skipped[f"Body_{i+1}_pos"][2], c = colours[f"Color_{i+1}"], label = f"Body {i+1}",s=2)
        plt.legend()
        plt.show()
        
        #------------------------Plot Separation between Planet and Ba, Bb ------------------------
        separation_Ba_Planet = separation_of_2_bodies_over_runtime(positions_body_based_skipped["Body_3_pos"],positions_body_based_skipped["Body_5_pos"])
        separation_Bb_Planet = separation_of_2_bodies_over_runtime(positions_body_based_skipped["Body_4_pos"],positions_body_based_skipped["Body_5_pos"])
        
        time_on_axis = year
        separation_on_axis = AU
        plt.plot(np.array(time_points[::skip_frames])/time_on_axis,np.array(separation_Ba_Planet)/separation_on_axis,label="Ba")
        plt.plot(np.array(time_points[::skip_frames])/time_on_axis,np.array(separation_Bb_Planet)/separation_on_axis,label="Bb")
        plt.title(f"HD98800 Circumbinary Planet initial separation: Ba {separation_Ba_Planet[0]/AU:.2f} AU, Bb {separation_Bb_Planet[0]/AU:.2f} AU")
        plt.xlabel("Time in years")
        plt.ylabel("Separation in AU")
        plt.legend()
        plt.show()
        #------------------------ Plot Separation of Aa, Ab and Ba, Bb-------------------------------
        separation_Ba_Bb = separation_of_2_bodies_over_runtime(positions_body_based_skipped["Body_3_pos"],positions_body_based_skipped["Body_4_pos"])
        separation_Aa_Ab = separation_of_2_bodies_over_runtime(positions_body_based_skipped["Body_1_pos"],positions_body_based_skipped["Body_2_pos"])

        plt.plot(np.array(time_points[::skip_frames])/time_on_axis,np.array(separation_Ba_Bb)/separation_on_axis,label="Ba Bb")
        plt.title(f"HD98800 Ba and Bb")
        plt.ylabel("Separation in AU")
        plt.xlabel("Time in years")
        plt.legend()
        plt.show()
        plt.plot(np.array(time_points[::skip_frames])/time_on_axis,np.array(separation_Aa_Ab)/separation_on_axis,label="Aa Ab")
        plt.title(f"HD98800 Aa and Ab")
        plt.ylabel("Separation in AU")
        plt.xlabel("Time in years")
        plt.legend()
        plt.show()

    if save_current_run:
        if save_directory == None or save_directory == "": print("Save directory is not stated, Data is not saved")
        else: np.savez(save_directory, positions=positions_body_based_skipped, velocities=velocities_body_based_skipped)

    if len(sols.t_events[0]) > 0: #if an event occured during runtime (collision or separation too large means this seed is not a viable system):
        print("Bad seed!", Initial_Seed)
        if saving:
            with open("bad_seed.txt", "a") as file: #open file for bad seeds in mode append
                file.write(f"{Initial_Seed}\n") #write initial seed
                file.close() #close file
    else:
        print("Good seed!", Initial_Seed)
        if saving:
            with open("good_seed.txt", "a") as file: #open file for good seeds in mode append
                file.write(f"{Initial_Seed}\n") #write initial seed
                file.close() #close file
    #I iterate through seeds to find which systems are stable for one A-B orbital period (~265 years) at ~50-100 seconds per time step. Leave this running overnight to get a bunch of good seeds
    #I can then look into the details of these systems later and inspect which orbital parameters cause the system to be unstable and then write about this in the report
                
    sols.clear() #clear sols to not take up any more ram for next iteration (if running one)
    if Stop_Time_Per_Iteration != None:
        if iteration*Stop_Time_Per_Iteration > Stop_Time_Global: #if global stop time has been reached
            true_statement = False #terminate
    iteration += 1
    if not multiple_iteration: #if multiple iterations of the code is false
         true_statement = False #break the while loop
    print(iteration)