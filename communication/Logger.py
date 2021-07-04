
debugLogs = False
debug_level = 0
def set_log_level(level):
    global debug_level
    debug_level = level

def get_log_level(level):
    return debug_level

def debug_print(str):
    if(debug_level >= 2):
        print(str)

def event_print(str):
    if(debug_level >= 1):
        print(str)

def milestone_print(str):
    if(debug_level >= 0):
        print(str)
