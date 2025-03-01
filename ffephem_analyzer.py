import numpy as np
from coe2roe_jit import calculate_ROE
from PS_check_jit import analytical_passive_safety
from time import perf_counter_ns
import time

assigned_frequency = '1/352'
num_of_orbits = 4.25
states_per_orbit = 352
kov = np.array([0.08, 0.72, 0.13])

per_evaluation_freq = round(eval(assigned_frequency) * states_per_orbit)
total_states = round(num_of_orbits * states_per_orbit)
chief_state_idx = 0
deputy_state_idx = 0
ephemeris_data_chief = [[],] * (total_states + 1)
ephemeris_data_deputy = [[],] * (total_states + 1)
ephemeris_data_roe = [[],] * (total_states + 1)
ephemeris_data_roe_dn = [[],] * (total_states + 1)

num_of_evals = round(total_states / per_evaluation_freq) + 1
PS_correct_counter = 0
PS_runtime = np.array([0.,] * num_of_evals)

# Chief states
with open('nom_traj_chief_ephem.FFephem', 'r') as NOM_TRIAL:
    for EPHEM_LINE in NOM_TRIAL:
        line_contents = EPHEM_LINE.split("#")
        if "Jan" in line_contents[0] and len(line_contents) >= 16:
            ephemeris_data_chief[chief_state_idx] = np.array([float(line) for line in line_contents[10:]])
            ephemeris_data_chief[chief_state_idx][2] = np.radians(ephemeris_data_chief[chief_state_idx][2])
            ephemeris_data_chief[chief_state_idx][3] = np.radians(ephemeris_data_chief[chief_state_idx][3])
            ephemeris_data_chief[chief_state_idx][4] = np.radians(ephemeris_data_chief[chief_state_idx][4])
            ephemeris_data_chief[chief_state_idx][5] = np.radians(ephemeris_data_chief[chief_state_idx][5])
            chief_state_idx += 1

# Deputy states
with open('nom_traj_deputy_ephem.FFephem', 'r') as NOM_TRIAL:
    for EPHEM_LINE in NOM_TRIAL:
        line_contents = EPHEM_LINE.split("#")
        if "Jan" in line_contents[0] and len(line_contents) >= 16:
            ephemeris_data_deputy[deputy_state_idx] = np.array([float(line) for line in line_contents[16:22]])
            ephemeris_data_deputy[deputy_state_idx][2] = np.radians(ephemeris_data_deputy[deputy_state_idx][2])
            ephemeris_data_deputy[deputy_state_idx][3] = np.radians(ephemeris_data_deputy[deputy_state_idx][3])
            ephemeris_data_deputy[deputy_state_idx][4] = np.radians(ephemeris_data_deputy[deputy_state_idx][4])
            ephemeris_data_deputy[deputy_state_idx][5] = np.radians(ephemeris_data_deputy[deputy_state_idx][5])
            deputy_state_idx += 1

# Compute PS
counter = 0
for k in range(len(ephemeris_data_chief)):
    oe = ephemeris_data_chief[k]
    oed = ephemeris_data_deputy[k]
    # warm up JIT
    if k == 0:
        for x in range(100):
            analytical_passive_safety(AS=kov[2], BS=kov[0], SMA=ephemeris_data_chief[k][0], 
                                    OE1=oe[0], OE2=oe[1], OE3=oe[2], OE4=oe[3], OE5=oe[4], OE6=oe[5], 
                                    OED1=oed[0], OED2=oed[1], OED3=oed[2], OED4=oed[3], OED5=oed[4], OED6=oed[5])
    if k % per_evaluation_freq == 0:
        t0 = perf_counter_ns()
        result = analytical_passive_safety(AS=kov[2], BS=kov[0], SMA=ephemeris_data_chief[k][0], 
                                           OE1=oe[0], OE2=oe[1], OE3=oe[2], OE4=oe[3], OE5=oe[4], OE6=oe[5], 
                                           OED1=oed[0], OED2=oed[1], OED3=oed[2], OED4=oed[3], OED5=oed[4], OED6=oed[5])
        tf = perf_counter_ns()
        PS_runtime[counter] = np.float128(tf - t0) * 1e-3
        PS_correct_counter += 1 if bool(result) else 0
        counter += 1

# Output
print(len(PS_runtime), PS_correct_counter, num_of_evals)
print(assigned_frequency, num_of_evals, PS_correct_counter/num_of_evals * 100, np.mean(PS_runtime), np.std(PS_runtime), np.sum(PS_runtime))