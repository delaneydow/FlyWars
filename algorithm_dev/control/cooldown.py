# cooldown.py

# cpu thermal adaptability 
def get_cpu_temp(): 
    try: 
        with open("sys/class/thermal/thermal_zone0/temp") as f: 
            return float(f.read()) / 1000.0
    except Exception: 
        return None
    
def adaptive_cooldown(temp_c): # input temp in celsius
    # adjust waiting time based on board temperature. Will need to tune threshold

    if temp_c is None: 
        return 0.0
    
    if temp_c > 85: 
        return 0.25 # heaviest cooldown, 85 exceeds specs 
    elif temp_c > 75: 
        return 0.1
    elif temp_c > 65: #TODO pull up what mean operating/idle temp is and this value shouldn't be below that
        return 0.02
    else: 
        return 0.0