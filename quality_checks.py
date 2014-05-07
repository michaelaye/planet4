from P4_sandbox import helper_functions as hf

def check_for_outside_blotches(data):
    """Currently simply takes maximum radius of blotch and adds/subtracts from x/y."""
    # pixel coordinate maximum in x:
    right_side = 840
    cols = 'radius_1 radius_2'.split()
    #define length of all data
    no_all = data.shape[0]
    no_off_right = ((blotches.x - blotches[cols].max(axis=1)) > 840).value_counts()[True]
    return no_off_right/float(no_all)