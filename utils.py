import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import os
import datetime

import sys
# print_rewards(current_scores=scores, scores_hist=scores_hist, steps=step_count, ave_len=100)
def print_rewards(current_scores, scores_hist, steps, total_steps_count, epsilon, ave_len=100):
    if(len(scores_hist)>1):
        x = np.arange(0, len(scores_hist[-ave_len:]))
        y = np.nan_to_num(scores_hist[-ave_len:])  # getting occasional NAN.  TODO: why?
        z = np.polyfit(x, y, 1)  # outputs [a, b] as in ax+b I think
        slope = z[0]
        print ('episode:{}, steps:{}, total steps:{}, current:[{:+.3f}, {:+.3f}], score:{:5.3f} slope:{:.2e} epsilon:{:.2f}' \
               .format(len(scores_hist),
                       steps,
                       total_steps_count,
                       current_scores[0],
                       current_scores[1],
                       np.mean(scores_hist[-ave_len:]),
                       slope,
                       epsilon,
                       ))

def print_misc(base_name):
    print('time :' + str(datetime.datetime.now()))
    print('main file: ' + base_name)
    print('platform = ' + platform.system())




class Plot_Scores:
    def __init__(self):
        plt.clf()
        matplotlib.use('tkagg')  # needed to run on AWS wiht X11 forwarding
        self.line, = plt.plot(np.array(0), np.array(0),'b:')
        self.line1, = plt.plot(np.array(0,),np.array(0),'r-')
        self.axes = plt.gca()
        plt.ion()
        plt.xlabel = 'Episode'
        plt.ylabel = 'Mean score'


    def plot(self, scores_hist):

        if len(scores_hist) > 2:
            ma = []
            for i in range(len(scores_hist)):
                ma.append(np.average(scores_hist[i-100:i]))
            self.line.set_xdata(np.arange(0, len(scores_hist)))
            self.line1.set_xdata(np.arange(0, len(ma)))
            self.line.set_ydata(scores_hist)
            self.line1.set_ydata(ma)
            self.axes.set_xlim(max(0, len(scores_hist) - 2000), len(scores_hist))
            y_min = np.min(scores_hist)
            y_max = np.max(scores_hist)
            self.axes.set_ylim(y_min - np.abs(.05*y_min), y_max + np.abs(.05*y_max))
            plt.draw()
            plt.pause(.1)

    def save(self, suffix):
        plt.savefig('plot_image trial' + suffix + '.jpeg', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


class Logger(object):
    def __init__(self, f_name):
        self.terminal = sys.stdout
        self.log = open(f_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


