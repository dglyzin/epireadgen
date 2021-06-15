# python3 -m solvers.hs.remoterun.progresses.progress_main
import re
from hybriddomain.solvers.hs.remoterun.progresses.progress_cmd import ProgressCmd


class StdoutProgresses():

    '''Progress for parsing stdout.
    Can make output to cmd or given notebook progress.'''

    def __init__(self, re_pattern="(?P<val>[\d]+)\s?%",
                 STEPS=100, notebook=None):
        self.re_pattern = re_pattern
        self.prefix = "solving"
        self.steps = STEPS

        self.progresses = []

        if notebook is not None:
            # from hybriddomain.solvers.hs.remoterun.progresses.progress_notebook import ProgressNotebook
            self.progresses.append(notebook)
            # self.progresses.append(ProgressNotebook(STEPS,
            #                                         prefix=self.prefix))
        else:
            self.progresses.append(ProgressCmd(STEPS,
                                               prefix=self.prefix))

    def set_prefix(self, prefix):
        self.prefix = prefix
        for progress in self.progresses:
            progress.set_prefix(prefix)

    def show_stdout_progresses(self, line):
        res = re.search(self.re_pattern, line)
        if res is not None:
            res_str = res.group("val")
            value = int(res_str)
            # res_str = res.group()
            # value = int(res_str[:-1])
            if value > self.steps:
                raise(BaseException("progress value more then total"))
            for progress in self.progresses:
                progress.succ(value)
    

def test():
    lines = ["there is 11% completed"]
    lines.append("there is 1 % completed")
    lines.append("there is 100 % completed")
    
    progress = StdoutProgresses()
    for line in lines:
        progress.show_stdout_progresses(line)


if __name__ == '__main__':
    test()
