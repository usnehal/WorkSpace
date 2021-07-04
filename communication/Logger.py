class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
        print(bcolors.OKCYAN + str)

def milestone_print(str):
    if(debug_level >= 0):
        print(bcolors.OKGREEN + str)
