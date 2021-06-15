# python3 -m solvers.hs.remoterun.progresses.progress_cmd
from time import sleep


class ProgressCmd():
    def __init__(self, STEPS, prefix="progress"):
        self.step_progress = 0
        self.steps_total = STEPS
        self.set_prefix(prefix)
        
    def succ(self, value):
        self.step_progress = value
        progress_cmd(value, self.steps_total, is_sleep=False,
                     prefix=self.prefix)

    def set_steps(self, STEPS):
        self.steps_total = STEPS

    def get_steps(self):
        return(self.steps_total)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_prefix(self):
        return(self.prefix)

    def print_end(self):
        print("\ndone")

        
def progress_cmd(value, total, win_size=10, prefix="progress",
                 is_sleep=True):
    
    procent_value = int(value*100 / (total))
    p_value = int(value * win_size / (total))
    progress = "█"*p_value
    prefix = prefix + ": "
    print(prefix+progress+(" %d" % procent_value)+" %", end='\r')
    if value >= total:
        print()
    if is_sleep:
        sleep(0.1)


def test_cmd(interval=1):
    print("test for: ")

    values = [i*interval for i in range(100)]
    print(values)

    for value in values:
        progress_cmd(value, values[-1])


if __name__ == '__main__':
    test_cmd(1)
    test_cmd(7)
    test_cmd(3)
