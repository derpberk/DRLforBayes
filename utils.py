
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def message_log(msg, msg_type='WARNING'):

    head = getattr(BColors, msg_type)
    return head + msg + BColors.ENDC

def print_log(msg, msg_type='WARNING'):

    head = getattr(BColors, msg_type)
    print(head + msg + BColors.ENDC)

