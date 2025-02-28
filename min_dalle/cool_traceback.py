import pdb
import sys
import traceback

class b:
    BLUE = '\033[34m'
    WEIRD = '\033[41m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_border():
    print(b.WEIRD + "- - - - - - - - - " + b.ENDC)

def print_file_line(stringa, stringb):
    parts = stringa.split('"')

    # if len(parts) < 3:
    #     return

    stringa_fancy = '' + parts[0] + b.ENDC \
        + b.FAIL + "\\".join(parts[1].split("\\")[-3:]) + b.ENDC \
        + '' + parts[2] + b.ENDC

    print(stringa_fancy)

    stringb_fancy = stringb

    print(stringb_fancy)

    
def print_info_line(string):
    print(b.BLUE + string + "       " + b.ENDC)

def print_error_line(string):
    parts = string.split(':')

    print(b.FAIL + parts[0] + b.ENDC + ': ')

def cool_traceback(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        data = traceback.format_exc().splitlines()

        print_border()

        for i, a in enumerate(data[:-1]):
            if "min-dalle" in data[i] and "site-packages" not in data[i]:
                print_file_line(data[i], data[i+1])

        print(flush=True)
        print_error_line(data[-1])
        print(e, flush=True)

        print_border()

        # sys.last_traceback = traceback
        tb = sys.exc_info()[-1]
        import pdb; pdb.post_mortem(tb)
