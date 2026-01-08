from astropy import units as u
from astropy import time
import astropy
import numpy as np

from poliastro import iod
from poliastro.bodies import Body,Mars, Earth, Venus, Jupiter, Saturn, Uranus, Neptune, Sun, Europa, Ganymede, Callisto, Io, Titan
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.frames import Planes
from poliastro.frames.fixed import JupiterFixed
from poliastro.frames.equatorial import JupiterICRS
from astroquery.jplhorizons import Horizons
from copy import deepcopy
from collections import defaultdict

from scipy import ndimage
from scipy.spatial.transform import Rotation as scipyRot

from poliastro.plotting import OrbitPlotter3D, StaticOrbitPlotter
import math
import matplotlib.pyplot as plt
# More info: https://plotly.com/python/renderers/
import plotly.io as pio
from poliastro.util import norm, time_range
pio.renderers.default = "plotly_mimetype+notebook_connected"
import weakref
from astropy.coordinates import solar_system_ephemeris
from collections.abc import Mapping

solar_system_ephemeris.set("jpl")

def match_astro_query_num(body):
    if body==Io:
        return 501
    if body==Europa:
        return 502
    if body==Ganymede:
        return 503
    if body==Callisto:
        return 504
    else:
        return False
    
def get_galilean_ephemerides(start_date, end_date, num_periods=50):
    """
    Generate ephemeris data for all Galilean moons.
    
    Parameters:
    -----------
    start_date : astropy.time.Time
        Start date for ephemeris
    end_date : astropy.time.Time
        End date for ephemeris
    num_periods : int
        Number of time steps for ephemeris
        
    Returns:
    --------
    dict : Dictionary mapping Body objects to their Ephem objects
    """
    epochs = time_range(start=start_date, end=end_date, periods=num_periods)
    
    # Dictionary mapping Body objects to their Horizons ID numbers
    galilean_ids = {
        Io: 501,
        Europa: 502,
        Ganymede: 503,
        Callisto: 504
    }
    
    # Generate ephemerides for each moon
    ephem_dict = {}
    for body, horizons_id in galilean_ids.items():
        ephem = Ephem.from_horizons(
            horizons_id,
            epochs=epochs,
            attractor=Jupiter,
            plane=Planes.EARTH_ECLIPTIC
        )
        ephem_dict[body] = ephem
    
    return ephem_dict

def get_galilean_orbs(start_date,end_date):
    # Generate ephemerides for each moon
    ephemerides = get_galilean_ephemerides(start_date, end_date)
    #Generate orbit data for all Galilean moons.
    orbit_dict = {}
    for body in [Io,Europa,Ganymede,Callisto]:
        orbit=Orbit.from_ephem(attractor=Jupiter,ephem=ephemerides[body],epoch=start_date)
        orbit_dict[body] = orbit
    
    return orbit_dict
    
def ecliptic_slingshot(spacecraft_orb,body_orb,body,h_p,sign):
    r_p=h_p+body.R
    attractor=spacecraft_orb.attractor
    flyby_dist=np.linalg.norm(spacecraft_orb.r-body_orb.r)
    if(flyby_dist)<2*body.R:
        rel_vel=spacecraft_orb.rv()[1]-body_orb.rv()[1]
        rel_speed=np.linalg.norm(rel_vel)
        rot_ang = 2 * np.arcsin(1 / ((r_p * rel_speed**2 / body.k) + 1 * u.one))
        if(r_p<body.R):
            print("Warning: spacecraft crashes into the body, ignoring that for now.")
        # Rotation axis in ecliptic plane
        axis = np.array([0, 0, 1.0])
        axis = axis / np.linalg.norm(axis)
        # Rotate the relative velocity
        rot = scipyRot.from_rotvec(sign * axis * rot_ang.value)
        rel_vel_out = rot.apply(rel_vel.value) * u.km / u.s
        
        # Calculate post-assist velocity in attractor frame
        post_assist_vel = rel_vel_out + body_orb.rv()[1]
        
        # Create orbit from post-assist state
        test_orb = Orbit.from_vectors(
            attractor, 
            body_orb.r, 
            post_assist_vel, 
            spacecraft_orb.epoch,
            plane=Planes.EARTH_ECLIPTIC
        )

        return (post_assist_vel,test_orb)
                
                
    else:
        print(f"Trajectory does not fly past {body} closely enough - misses by {flyby_dist.to(u.km):.1f}")
        return None

def slingshot_to_another_body(spacecraft_orb, flyby_body,target_body,date_range, max_tof=50*u.day,tof_step=0.1*u.day):
    galilean_orbs = get_galilean_orbs(date_range[0],date_range[-1])
    # print(galilean_orbs)
    target_body_orb_initial=galilean_orbs[target_body]
    flyby_body_orb=galilean_orbs[flyby_body].propagate(spacecraft_orb.epoch)
    initial_dist=np.linalg.norm(spacecraft_orb.r-flyby_body_orb.r)
    initial_sped=np.linalg.norm(spacecraft_orb.v-flyby_body_orb.v)
    lowest_dv=100000*u.m/u.s
    best_orb=None
    target_arrival_orb=None
    print(flyby_body_orb)
    print(target_body_orb_initial)
    arrival_v=spacecraft_orb.v
    if initial_dist>5*flyby_body.R:
        raise ValueError(f"Misses {flyby_body} by {initial_dist:.2f} km.")
    for i in range(int(max_tof.to(u.day).value/tof_step.to(u.day).value)):
        arrival_date=spacecraft_orb.epoch+i*tof_step
        target_body_orb=target_body_orb_initial.propagate(arrival_date+1*u.s)
        try:
            lambert=Maneuver.lambert(spacecraft_orb,target_body_orb)
            burn=lambert[0]
            dv=np.linalg.norm(burn[1])
            new_spacecraft_orb,dummy=spacecraft_orb.apply_maneuver(lambert,intermediate=True)
            speed_def=np.linalg.norm(new_spacecraft_orb.v.to(u.km/u.s)-arrival_v.to(u.km/u.s))
            angle_dif=np.arccos(np.dot(new_spacecraft_orb.v.to(u.km/u.s).value,arrival_v.to(u.km/u.s).value)/(norm(new_spacecraft_orb.v).to(u.km/u.s).value*norm(arrival_v).to(u.km/u.s).value))*u.rad
            r_p=((math.sin(angle_dif.value/2)**(-1))-1)/(initial_sped**2)*flyby_body.k
            if speed_def<500*u.m/u.s:
                lowest_dv=dv
                best_orb=new_spacecraft_orb
                target_arrival_orb=target_body_orb
                print(f"Found transfer with {angle_dif}, peripasis of {(r_p-flyby_body.R).to(u.km):.2f}.")
        except Exception as e:
            continue
    print(f"Min dv {lowest_dv:.1f}")
    return best_orb, target_arrival_orb


#tries to find the lowest DV burn at periapsis of the initial orbit that 
#leads to encounter with target body
def get_single_burn_elliptical_hohmann_generate_ephem(target, initial_orb, min_tof=0.1*u.day, max_tof=50*u.day, tof_step=0.1*u.day, max_revs=0):
    attractor=initial_orb.attractor
    time_till_pe=initial_orb.period-initial_orb.t_p
    periapsis_epoch=initial_orb.epoch+time_till_pe

    sim_range=time_range(start=periapsis_epoch,end=periapsis_epoch+max_tof,periods=50)
    min_dv=50000*u.m/u.s
    final_orb=None
    final_date=None
    final_burn=None
    final_targ_orb=None
    periapsis_orb=initial_orb.propagate(periapsis_epoch)
    body_num=match_astro_query_num(target)
    body_ephem=Ephem.from_horizons(body_num,epochs=sim_range, attractor=attractor, plane=Planes.EARTH_ECLIPTIC)
    body_orb_placeholder = Orbit.from_ephem(attractor, body_ephem, periapsis_epoch)
    tof_range=np.arange(min_tof.to(u.day).value,max_tof.to(u.day).value,tof_step.to(u.day).value)*u.day
    print(f"Getting ephems from {periapsis_epoch}.")
    for i in range(len(tof_range)):
        arrival_date=periapsis_epoch+tof_range[i]
        body_orb=body_orb_placeholder.propagate(arrival_date+1*u.s)
        # print(f"{initial_orb} and {body_orb}")
        for j in range(max_revs+1):
            try:
                lambert=Maneuver.lambert(periapsis_orb,body_orb,M=j)
                burn=lambert[0]
                dv=np.linalg.norm(burn[1])
                if dv<min_dv:
                    min_dv=dv
                    final_orb,dummy=periapsis_orb.apply_maneuver(lambert,intermediate=True)
                    final_date=arrival_date
                    final_burn=burn
                    final_targ_orb=body_orb
            except Exception as e:
                # print(e)
                continue

    return [min_dv,final_orb,final_targ_orb,final_date,final_burn]


##Takes in target orbit, rather than generating from ephem to avoid Jovian ephem problems
def get_single_burn_elliptical_hohmann(target_orb, initial_orb, min_tof=0.1*u.day, max_tof=50*u.day, tof_step=0.1*u.day, max_revs=0):
    time_till_pe=initial_orb.period-initial_orb.t_p
    periapsis_epoch=initial_orb.epoch+time_till_pe

    min_dv=50000*u.m/u.s
    final_orb=None
    final_date=None
    final_burn=None
    final_targ_orb=None
    periapsis_orb=initial_orb.propagate(periapsis_epoch)
    body_orb_placeholder = target_orb.propagate(periapsis_epoch)
    tof_range=np.arange(min_tof.to(u.day).value,max_tof.to(u.day).value,tof_step.to(u.day).value)*u.day
    for i in range(len(tof_range)):
        arrival_date=periapsis_epoch+tof_range[i]
        body_orb=body_orb_placeholder.propagate(arrival_date+1*u.s)
        # print(f"{initial_orb} and {body_orb}")
        for j in range(max_revs+1):
            try:
                lambert=Maneuver.lambert(periapsis_orb,body_orb,M=j)
                burn=lambert[0]
                dv=np.linalg.norm(burn[1])
                if dv<min_dv:
                    min_dv=dv
                    final_orb,dummy=periapsis_orb.apply_maneuver(lambert,intermediate=True)
                    final_date=arrival_date
                    final_burn=burn
                    final_targ_orb=body_orb
            except Exception as e:
                # print(e)
                continue

    return [min_dv,final_orb,final_targ_orb,final_date,final_burn]






def assist_possible_periods(spacecraft_orb, body_orb, body, r_p_min=10*u.km, r_p_max=5000*u.km, num_samples=500):#finds possible periods of orbit around attractor given flyby
    periods=np.zeros((num_samples,2))
    periaps=np.zeros((num_samples))
    attractor=spacecraft_orb.attractor
    flyby_dist=np.linalg.norm(spacecraft_orb.r-body_orb.r)
    if(flyby_dist)<4*body.R:
        rel_vel=spacecraft_orb.rv()[1]-body_orb.rv()[1]
        rel_speed=np.linalg.norm(rel_vel)
        r_ps = body.R + np.linspace(r_p_min.to(u.km).value, r_p_max.to(u.km).value, num=num_samples) * u.km
        rot_angs = 2 * np.arcsin(1 / ((r_ps * rel_speed**2 / body.k) + 1 * u.one))
        # Rotation axis in ecliptic plane
        axis = np.array([0, 0, 1.0])
        axis = axis / np.linalg.norm(axis)
        # Try both rotation directions (leading/trailing side flybys)
        for j,sign in enumerate([1,-1]):
            for i, rot_ang in enumerate(rot_angs):
                # Rotate the relative velocity
                rot = scipyRot.from_rotvec(sign * axis * rot_ang.value)
                rel_vel_out = rot.apply(rel_vel.value) * u.km / u.s
                
                # Calculate post-assist velocity in attractor frame
                post_assist_vel = rel_vel_out + body_orb.rv()[1]
                
                # Create orbit from post-assist state
                test_orb = Orbit.from_vectors(
                    attractor, 
                    body_orb.r, 
                    post_assist_vel, 
                    spacecraft_orb.epoch,
                    plane=Planes.EARTH_ECLIPTIC
                )
                periods[i][j]=test_orb.period.to(u.day).value
                periaps[i]=r_ps[i].to(u.km).value
        return (periods,periaps)
                
                
    else:
        print(f"Trajectory does not fly past {body} closely enough - misses by {flyby_dist.to(u.km):.1f}")
        return None

def resonance_search(spacecraft_orb,body,r_p_min=10*u.km, r_p_max=5000*u.km, lower=True, numerator_prioritize=True, max_numerator=9,sim_start_date=time.Time("2037-06-11 00:01", scale="utc").tdb):
    used_orbit=deepcopy(spacecraft_orb)
    attractor=spacecraft_orb.attractor
    galilean_orbs=get_galilean_orbs(sim_start_date,sim_start_date+10*u.day)
    prelim_body_orb= galilean_orbs[body].propagate(used_orbit.epoch)
    postflyby_data=assist_possible_periods(used_orbit, prelim_body_orb, body,r_p_min=r_p_min, r_p_max=r_p_max,num_samples=5000)
    possible_resonances=[]
    if postflyby_data is not None:
        ##Check which side gives lower periods
        trailing_mean_period = np.mean(postflyby_data[:][0])
        leading_mean_period = np.mean(postflyby_data[0][:, 1])
        if trailing_mean_period < leading_mean_period:
            lower_side = 0  # Trailing side (sign=1)
            # print("Trailing side flybys give lower periods")
        else:
            lower_side = 1  # Leading side (sign=-1)
            # print("Leading side flybys give lower periods")

        side=lower_side if lower else 1-lower_side

        sign=int((-1)**(side))

        
        #generate allowed resonances      
        pairs = np.array([(i, j)
                  for i in range(1, max_numerator + 1)
                  for j in range(1, max_numerator + 1)
                  if math.gcd(i, j) == 1], dtype=int)
        # optional: the fraction values
        vals = pairs[:, 0] / pairs[:, 1]
        
        possible_resonant_periods = vals*prelim_body_orb.period.to(u.day)
        for i,period in enumerate(postflyby_data[0][:,side]):
            for j,test_period in enumerate(possible_resonant_periods):
                t_ratio=(((period*u.day)/test_period).to(u.one)).value
                if abs(1-t_ratio)<0.001:
                    # print(f"At {test_period:.1f} ie {pairs[j]} with flyby at {postflyby_data[1][i]*u.km-body.R.to(u.km)}")
                    possible_resonances.append((pairs[j], postflyby_data[1][i]*u.km-body.R.to(u.km)))
        # print(possible_resonances)
        
        groups = defaultdict(list)

        for frac_arr, rp in possible_resonances:
            key = tuple(frac_arr.tolist())      # e.g. (5, 7)
            groups[key].append(rp.to(u.km))     # keep as Quantity
        # one averaged periapsis per resonance fraction
        averaged = []
        for key, rps in groups.items():
            rp_mean = np.mean(u.Quantity(rps))  # Quantity mean, stays in km
            averaged.append((np.array(key, dtype=int), rp_mean, len(rps)))         
        
        if numerator_prioritize:#pick the resonance with the lowest numerator
            averaged.sort(key=lambda x: (x[0][0], x[1].value))  # Sort by numerator, then by periapsis
        h_p=averaged[0][1]
        chosen_resonance=averaged[0][0]
        
        tof=prelim_body_orb.period * (chosen_resonance[0])
        # print(f"Time of flight is {tof.to(u.day):.1f}, which is {chosen_resonance[0]} times the period of the moon.")
        arrival_date=used_orbit.epoch + tof
        target_orb=prelim_body_orb.propagate(arrival_date)
        print(f"Going for {chosen_resonance} resonance with periapsis at {h_p}, time till next encounter is {tof.to(u.day):.2f}")
        print(f"Flyby v inf speed is {np.linalg.norm(used_orbit.v-prelim_body_orb.v):.2f}")
        simple_ecliptic_vector,dummy2=ecliptic_slingshot(used_orbit,prelim_body_orb,body,h_p,1*sign)
        
        #find actual transfer
        print(f"Spacecraft does {chosen_resonance[1]-1} complete orbits before arriving, {body} does {chosen_resonance[0]-1}.")
        # print(f"Period of spacecraft orbit should be {(tof/chosen_resonance[1]).to(u.day):.1f}, actually is {dummy2.period.to(u.day):.1f}")
        # plotter=StaticOrbitPlotter(plane=Planes.EARTH_ECLIPTIC)       
        # plotter.plot(spacecraft_orb,label='initial_orb')
        # plotter.plot(target_orb,label='destination')
        # plotter.plot(dummy2.propagate(target_orb.epoch),label='ecliptic screwup')
        print(f"Arrives with distance of {np.linalg.norm(dummy2.propagate(target_orb.epoch).r-target_orb.r):.2f}")
        if dummy2.r_p.to(u.km)<1.05*attractor.R.to(u.km):
            print(f"Orbit crashes into {attractor}")
        return dummy2,tof

    else:
        print("Ending resonance search.")
        return None



def multiple_resonances(spacecraft_orb, body, num_flybys, lower=False):
    active_orb=deepcopy(spacecraft_orb)
    orbits_so_far=[]
    for f in range(num_flybys):
        print(f"Flyby {f+1}")
        orbits_so_far.append(active_orb)
        try:
            dorb,time=resonance_search(active_orb,body,lower=lower)
            active_orb=dorb.propagate(time)
            continue
        except Exception as e:
            print(f"Failed to find resonance: {e}")
    orbits_so_far.append(active_orb)
    return orbits_so_far





####DEPRICATED

##find lowest possible period orbit that is still a multiple of ganymede's period, this can continue until period approaches Ganymedes
def search_for_resonant_orbit(body_orb, body, inc_orb_vel, r_p_min, r_p_max, num_samples=150, max_resonance_ratio=10,lower=True):
    """
    Find the gravity assist that produces the lowest period orbit that is still a resonant multiple of the body's period.
    
    Parameters:
    body_orb: Orbit of the body we're slingshotting around
    body: The body itself (e.g., Ganymede)
    inc_orb_vel: Incoming velocity vector of spacecraft relative to Jupiter
    r_p_min, r_p_max: Min/max periapsis distances for flyby
    num_samples: Number of periapsis values to scan
    max_resonance_ratio: Maximum numerator or denominator to check (default 10)
    
    Returns:
    best_orbit: The resonant orbit with lowest period
    best_resonance: Tuple (n_sc, n_body) representing the resonance ratio
    best_rp: Periapsis distance that achieves this
    """
    
    slingshot_epoch = body_orb.epoch
    body_period = body_orb.period
    
    # Relative velocity & speed to body we are slingshotting around
    rel_vel = inc_orb_vel - body_orb.rv()[1]
    rel_speed = np.linalg.norm(rel_vel)
    
    # Sample periapsis distances
    r_ps = body.R + np.linspace(r_p_min.value, r_p_max.value, num=num_samples) * u.km
    
    # Calculate deflection angles for each periapsis
    rot_angs = 2 * np.arcsin(1 / ((r_ps * rel_speed**2 / body.k) + 1 * u.one))
    
    # Rotation axis in ecliptic plane
    axis = np.array([0, 0, 1.0])
    axis = axis / np.linalg.norm(axis)
    
    best_orbit = None
    best_resonance = None
    best_rp = None
    if lower:
        best_period = np.inf * u.day
    else:
        best_period = 0 * u.day
    
    # Try both rotation directions (leading/trailing side flybys)
    for sign in [1,-1]:
        for i, rot_ang in enumerate(rot_angs):
            # Rotate the relative velocity
            rot = scipyRot.from_rotvec(sign * axis * rot_ang.value)
            rel_vel_out = rot.apply(rel_vel.value) * u.km / u.s
            
            # Calculate post-assist velocity in Jupiter frame
            post_assist_vel = rel_vel_out + body_orb.rv()[1]
            
            # Create orbit from post-assist state
            test_orb = Orbit.from_vectors(
                Jupiter, 
                body_orb.r, 
                post_assist_vel, 
                slingshot_epoch,
                plane=Planes.EARTH_ECLIPTIC
            )
            
            #only consider orbits that don't crash into the planet
            if test_orb.r_p < Jupiter.R:
                continue
            
            # Only consider bound orbits
            if test_orb.ecc >= 1:
                continue
            
            # Check if orbit crosses body's orbital radius (necessary for resonance)
            body_radius = body_orb.a
            orbit_crosses = (test_orb.r_p < body_radius < test_orb.r_a)
            
            if not orbit_crosses:
                continue
            
            # Calculate the period ratio
            period_ratio = test_orb.period / body_period
            # print(period_ratio)
            # Check for integer resonances by testing simple fractions
            found_resonance = False
            for n_body in range(1, max_resonance_ratio + 1):
                for n_sc in range(1, max_resonance_ratio + 1):
                    expected_ratio = n_body / n_sc
                    ratio_error = abs(period_ratio.value - expected_ratio) / expected_ratio
                    
                    if ratio_error < 0.03 :
                        # Check if this is better than current best
                        is_better = (test_orb.period < best_period) if lower else (test_orb.period > best_period)
                        if is_better:
                            # Additional check for lower=False: only accept if period is LONGER than body period
                            if not lower and test_orb.period <= body_period:
                                continue
                            
                            best_period = test_orb.period
                            best_orbit = test_orb
                            best_resonance = (n_sc, n_body)
                            best_rp = r_ps[i]
                            print(f"Found {n_sc}:{n_body} resonance: period={test_orb.period.to(u.day):.2f}, r_p={r_ps[i]:.0f}")
                            found_resonance = True
                            break
                if found_resonance:
                    break
    
    if best_orbit is None:
        print("No resonant orbit found!")
        return None, None, None
    
    print(f"\nBest resonance: {best_resonance[0]}:{best_resonance[1]}")
    print(f"Period: {best_period.to(u.day):.2f} (target body: {body_period.to(u.day):.2f})")
    print(f"Periapsis: {best_rp:.0f}")
    
    return best_orbit, best_resonance, best_rp
    
    
    
def match_orbit_plane(source_orbit, target_orbit):
    """
    Create a new orbit with the same orbital parameters as source_orbit
    but with the inclination and RAAN from target_orbit (making it coplanar).
    
    Parameters
    ----------
    source_orbit : Orbit
        Orbit whose parameters (a, ecc, argp, nu) will be used
    target_orbit : Orbit
        Orbit whose plane (inc, raan) will be used
        
    Returns
    -------
    Orbit
        New orbit with source parameters in target's plane
    """
    
    # Get orbital elements from source
    a = source_orbit.a
    ecc = source_orbit.ecc
    argp = source_orbit.argp
    nu = source_orbit.nu
    
    # Get plane orientation from target
    inc = target_orbit.inc
    raan = target_orbit.raan
    
    # Create new orbit with combined parameters
    new_orbit = Orbit.from_classical(
        attractor=source_orbit.attractor,
        a=a,
        ecc=ecc,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu,
        epoch=source_orbit.epoch,
        plane=source_orbit.plane
    )
    
    return new_orbit
