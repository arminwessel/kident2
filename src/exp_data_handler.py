import os
from pathlib import Path
import datetime
import pickle
from da_test_suite_functions import latex_to_pdf
class ExperimentDataHandler:
    figures = {}
    dataframes = {}
    notes = {}
    marker_locations = {}

    def add_figure(self, figure, name):
        self.figures[name] = figure

    def add_df(self, df, name):
        self.dataframes[name] = df

    def add_note(self, note_body, note_name):
        self.notes[note_name] = note_body

    def add_marker_location(self, marker_locations, name):
        self.marker_locations[name] = marker_locations

    def save_figure(self, figure, savepath, name):
        figure.canvas.draw()
        figure.savefig(savepath.joinpath(name + '.pdf'),  format='pdf', bbox_inches='tight')
        with open(savepath.joinpath(name + '.p'), 'wb') as f_handle:
            pickle.dump(figure, f_handle)

    def save_experiment(self, path_string):
        # prepare folder for saving data
        path = Path(path_string)
        timestamp = 'exp_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        savepath = path.joinpath(timestamp)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # save figures
        for name in self.figures:
            self.save_figure(self.figures[name], savepath, name)

        # save notes
        for name in self.notes:
            with open(savepath.joinpath(name + '.txt'), 'w') as notefile:
                notefile.write(self.notes[name])

        # save data
        for name in self.dataframes:
            # for latex
            cols = ['tex_names', 'errors_mm_deg', 'results_mm_deg', 'identification_accuracy']
            headers = ['parameter', 'simulated', 'identified', 'difference']
            texstr = self.dataframes[name].to_latex(escape=False, columns=cols, header=headers)
            latex_to_pdf(savepath, 'data', texstr)

            # as csv
            self.dataframes[name].to_csv(savepath.joinpath('data.csv'))

        # save marker locations
        for mlocation in self.marker_locations:
            with open(savepath.joinpath('marker_locations.p'), 'wb') as f:  # open a text file
                pickle.dump(self.marker_locations[mlocation], f)  # serialize the list

