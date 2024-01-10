import datetime, csv, os, numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from .misc import gimme_save_string


"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path):
        self.save_path = save_path
        self.written         = []   # written group recorder
        self.n_written_lines = {}   # written lines for each group

    def log(self, group, segments, content):
        if group not in self.n_written_lines.keys():
            self.n_written_lines[group] = 0

        with open(self.save_path+'_'+group+'.csv', "a") as csv_file:  # append mode, allow group file to be opened repeatedly
            writer = csv.writer(csv_file, delimiter=",")
            if group not in self.written: writer.writerow(segments)  # if the first time to write group, write segments
            for line in content:   # write lines
                writer.writerow(line)
                self.n_written_lines[group] += 1

        self.written.append(group)


################## PLOT SUMMARY IMAGE #####################
# plot multiple lists and save
class InfoPlotter:
    def __init__(self, save_path, title='Training Log', figsize=(25, 19)):
        self.save_path = save_path  # figure save path
        self.title     = title  # figure title
        self.figsize   = figsize  # figure size
        self.colors    = ['r','g','b','y','m','c','orange','darkgreen','lightblue']  # colors

    # sub_plots: plotted array name list
    # sub_plots_data: plotted matrix, one row for one plotted list
    def make_plot(self, base_title, title_append, sub_plots, sub_plots_data):
        sub_plots = list(sub_plots)
        if 'epochs' not in sub_plots:
            x_data = range(len(sub_plots_data[0]))  # the first line as x data
        else:
            x_data = range(sub_plots_data[np.where(np.array(sub_plots) == 'epochs')[0][0]][-1]+1)  # generate x data by epoch data

        # construct sub_plot titles
        # [(plot_name, plot_data), ...]
        self.ov_title = [(sub_plot, sub_plot_data) for sub_plot, sub_plot_data in zip(sub_plots, sub_plots_data) if sub_plot not in ['epoch', 'epochs', 'time']]
        # [(plot_name, max(plot_data)/min(plot_data), ...]
        self.ov_title = [(x[0], np.max(x[1])) if 'loss' not in x[0] else (x[0], np.min(x[1])) for x in self.ov_title]
        # title_append: plot_name0: max(plot_data0)/min(plot_data0) | ...
        self.ov_title = title_append + ': ' + '  |  '.join('{0}: {1:.4f}'.format(x[0], x[1]) for x in self.ov_title)

        # nonsense
        sub_plots_data = [x for x, y in zip(sub_plots_data, sub_plots)]
        sub_plots      = [x for x in sub_plots]

        # grid box style for saving as vector diagram
        plt.style.use('ggplot')
        f, ax = plt.subplots(1)  # get fig and sub-ax
        ax.set_title(self.ov_title, fontsize=22)  # set title
        for i, (data, title) in enumerate(zip(sub_plots_data, sub_plots)):
            ax.plot(x_data, data, '-{}'.format(self.colors[i]), linewidth=1.7, label=base_title+' '+title)  # plot each sub_plot
        ax.tick_params(axis='both', which='major', labelsize=18)  # add major ticks
        ax.tick_params(axis='both', which='minor', labelsize=18)  # add minor ticks
        ax.legend(loc=2, prop={'size': 16})  # set legend
        f.set_size_inches(self.figsize[0], self.figsize[1])  # set figure size
        f.savefig(self.save_path+'_'+title_append+'.svg')  # save vector diagram
        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
# generate save path & save options
def set_logging(opt):
    # construct save folder
    checkfolder = opt.save_path + '/' + opt.save_name
    if opt.save_name == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = opt.save_path+'/{}_{}_'.format(opt.dataset.upper(), opt.arch.upper())+time_string
        # folder name: dataset_arch_year-month-day-hour-minute-second

    # if the folder exists, add counter
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path +'/' + opt.save_name + '_' + str(counter)
        counter += 1

    # create the folder
    os.makedirs(checkfolder)
    opt.save_path = checkfolder

    if 'experiment' in vars(opt):  # filter experiment key
        import argparse
        save_opt = {key: item for key, item in vars(opt).items() if key != 'experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    # save options as string
    with open(save_opt.save_path+'/Parameter_Info.txt', 'w') as f:
        f.write(gimme_save_string(save_opt))
    # save options as object
    # pkl.dump(save_opt, open(save_opt.save_path+"/hypa.pkl", "wb"))


# generate data structure for saving
# self.groups = {
#   group_0:{  # Recall NMI, mAP
#       segment_0: {  # Recall@1, Recall@3, ...
#           'content': [c0, ...],
#           'saved_idx': 0,
#       }
#   }
#   ...
# }
class Progress_Saver():
    def __init__(self):
        self.groups = {}

    # e.g. group: Recall, segments: R@1, R@3
    def log(self, segment, content, group=None):
        if group is None:
            group = segment
        if group not in self.groups.keys():
            self.groups[group] = {}

        if segment not in self.groups[group].keys():
            self.groups[group][segment] = {'content':[], 'saved_idx':0}

        self.groups[group][segment]['content'].append(content)


class LOGGER:
    # sub_loggers = ["train", "test", "val"]
    def __init__(self, opt, sub_loggers=[], prefix=None, start_new=True, log_online=False):
        """
        LOGGER Internal Structure:

        self.progress_saver: Contains multiple Progress_Saver instances to log metrics for main metric subsets (e.g. "Train" for training metrics)
            ['main_subset_name']: Name of each main subset (-> e.g. "Train")
                .groups: Dictionary of subsets belonging to one of the main subsets, e.g. ["Recall", "NMI", ...]
                    ['specific_metric_name']: Specific name of the metric of interest, e.g. Recall@1.
        """
        self.prop        = opt
        self.prefix      = '{}_'.format(prefix) if prefix is not None else ''
        self.sub_loggers = sub_loggers

        # Make Logging Directories
        if start_new:
            set_logging(opt)

        # Set Graph and CSV writer
        self.csv_writer, self.graph_writer, self.progress_saver = {}, {}, {}
        for sub_logger in sub_loggers:
            csv_savepath = opt.save_path + '/CSV_Logs'
            if not os.path.exists(csv_savepath):
                os.makedirs(csv_savepath)
            self.csv_writer[sub_logger]     = CSV_Writer(csv_savepath+'/Data_{}{}'.format(self.prefix, sub_logger))

            prgs_savepath = opt.save_path + '/Progression_Plots'
            if not os.path.exists(prgs_savepath):
                os.makedirs(prgs_savepath)
            self.graph_writer[sub_logger]   = InfoPlotter(prgs_savepath + '/Graph_{}{}'.format(self.prefix, sub_logger))
            self.progress_saver[sub_logger] = Progress_Saver()


        ### WandB Init
        self.save_path   = opt.save_path
        self.log_online  = log_online

    def update(self, *sub_loggers, all=False):
        online_content = []

        if all: sub_loggers = self.sub_loggers

        # sub_logger: train test evaluation, group: NMI, mAP, Recall, segment: Recall@1, Recall@3, ...
        for sub_logger in list(sub_loggers):
            for group in self.progress_saver[sub_logger].groups.keys():
                pgs      = self.progress_saver[sub_logger].groups[group]
                segments = pgs.keys()
                # [start_index_0, ...]
                per_seg_saved_idxs   = [pgs[segment]['saved_idx'] for segment in segments]  # start index list for each group
                # [[c_start, ], ...]
                per_seg_contents     = [pgs[segment]['content'][idx:] for segment, idx in zip(segments, per_seg_saved_idxs)]  # content_list_from_start_index list for each group
                # [[c0, c1, ...], ...]
                per_seg_contents_all = [pgs[segment]['content'] for segment,idx in zip(segments, per_seg_saved_idxs)]  # content_list list for each group

                # Adjust indexes, marking: previous content elements have been handled
                for content, segment in zip(per_seg_contents, segments):
                    self.progress_saver[sub_logger].groups[group][segment]['saved_idx'] += len(content)

                # [
                #    [s0_start, s1_start, s2_start, ...],
                #    [s0_start+1, s1_start+1, s2_start+1, ...],
                #    ...
                # ]
                tupled_seg_content = [list(seg_content_slice) for seg_content_slice in zip(*per_seg_contents)]

                # e.g. group: Recall, segments: R@1, R@3
                # R@1, R@3
                # s0_start, s1_start
                # s0_start+1, s1_start+1
                self.csv_writer[sub_logger].log(group, segments, tupled_seg_content)

                # e.g. sub_logger: train, group: Recall, segments: R@1, R@3, plot R@1 and R@3 on the same figure
                self.graph_writer[sub_logger].make_plot(sub_logger, group, segments, per_seg_contents_all)

                for i, segment in enumerate(segments):
                    if group == segment:
                        name = sub_logger+': '+group
                    else:
                        name = sub_logger+': '+group+': '+segment
                    online_content.append((name, per_seg_contents[i]))

        if self.log_online:  # log_online
            if self.prop.online_backend == 'wandb':
                import wandb
                for i, item in enumerate(online_content):
                    if isinstance(item[1], list):
                        wandb.log({item[0]: np.mean(item[1])}, step=self.prop.epoch)
                    else:
                        wandb.log({item[0]: item[1]}, step=self.prop.epoch)
            elif self.prop.online_backend == 'comet_ml':
                for i, item in enumerate(online_content):
                    if isinstance(item[1], list):
                        self.prop.experiment.log_metric(item[0], np.mean(item[1]), self.prop.epoch)
                    else:
                        self.prop.experiment.log_metric(item[0], item[1], self.prop.epoch)
