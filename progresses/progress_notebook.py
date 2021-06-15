import ipywidgets as widgets


class ProgressNotebook():
    def __init__(self, STEPS, prefix='progress'):

        self.step_progress = 0
        self.progress = widgets.IntProgress(
            value=self.step_progress,
            min=0,
            max=STEPS-1,
            step=1,
            description=prefix+': ',
            bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        self.set_prefix(prefix)

    def succ(self, val):
        self.progress.value = val

    def set_steps(self, STEPS):
        # self.progress.min = 0
        # print("max-min:")
        # print(self.progress.max - self.progress.min)
        # print("Steps:")
        # print(STEPS)
        self.progress.max = STEPS-1

    def get_steps(self):
        return(self.progress.max+1)

    def set_prefix(self, prefix):
        self.prefix = prefix
        self.progress.description = prefix + ": "

    def get_prefix(self):
        return(self.prefix)

    def print_end(self):
        print("\ndone")


def test_notebook(interval=1):
    print("test for: ")

    values = [i*interval for i in range(100)]
    print(values)
    progress = ProgressNotebook(values[-1])

    for value in values:
        progress.succ(value)
