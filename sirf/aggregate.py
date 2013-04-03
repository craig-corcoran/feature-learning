import numpy
import matplotlib.pyplot as plt
import cPickle as pickle
import pandas
from glob import glob
from itertools import izip


import sirf
import sirf.util as util

logger = sirf.get_logger(__name__)
default_colors =  ['r','g','b','k','c','m','y','Aquamarine', 'Brown', 'CadetBlue', 
                    'BurlyWood', 'Crimson', 'DarkGoldenRod']

#def plot_aggregate_data(n_samples, d_loss_data, labels):

    #x = numpy.array(n_samples, dtype = numpy.float64) if n_samples else range(len(n_samples))
    #f = plt.figure()
    #logger.info('plotting aggregate run performance data')

    #num = len(d_loss_data)
    #cols = numpy.ceil(numpy.sqrt(num))
    #rows = numpy.ceil(num/cols)
    #for i,(key,mat) in enumerate(d_loss_data.items()):

        #ax = f.add_subplot(rows,cols,i+1) 
        
        #for h,lb in enumerate(labels):                
            
            #std = numpy.std(mat[:,:,h], axis=1)
            #ste = std / numpy.sqrt(x)
            #mn = numpy.mean(mat[:,:,h], axis=1)
            
            #ax.fill_between(x, mn-ste, mn+ste, alpha=0.15, linewidth = 0)
            #ax.plot(x, mn, label = lb)
            #plt.title(key)
            #plt.axis('off')
            ##plt.legend() # lower left

#def aggregate_dicts(paths):
    
    #D = {}
    #for p in paths:
        
        #with util.openz(p) as in_file:
            #d = pickle.load(in_file)

        #for k,v in d.items():
            #if D.get(k) is None:
                #D[k] = numpy.array([])
            #D[k] = numpy.append(D[k], v)

    #return D

#def path_to_param_dict(path):
    #''' 
    #takes path of the form 'mydir/file.a:1.b:2.pdf' and returns a dictionary
    #of params {a=1, b=2}
    #'''
    #param_str = path[path.find('.')+1 : path.rfind('.')]
    #param_str = param_str.replace('.', ',')
    #return eval('dict(%s)' % param_str)

def out_string(pref, root, d_params,  suff = '', remove = " ;{;};'".split(';')):
    ''' 
    takes a prefix (directory), a root file name, a dictionary of parameters,
    and an (optional) suffix and returns the combined and formatted string
    
    ex: out_string('mydir/', 'file', dict(a=1,b=2), '.pdf') 
            = mydir/file.a:1.b:2.pdf
    '''
    keys = map(lambda x: x[:2], d_params.keys()) # shortened form of param names
    s = str(dict(izip(keys, d_params.values())))
    for r in remove:
        s = s.replace(r, '')
    s = s.replace(',','.')
    s = s.replace(':','=')
    return '%s%s.%s%s' % (pref, root, s, suff)

def reorder_columns(d_params, d_loss):
    a = numpy.argsort(d_params.keys())
    b = numpy.argsort(d_loss.keys())
    keys_array = numpy.append(numpy.array(d_params.keys())[a], numpy.array(d_loss.keys())[b])
    values_array = numpy.append(numpy.array(d_params.values())[a], numpy.array(d_loss.values())[b])
    return keys_array, values_array


def main(file_path = None, logspace = False, plot_list = None):
    
    if file_path is None:
        path_list = glob('./sirf/output/csv/*.csv.gz')
        file_path = sorted(path_list)[-1]
        
    with util.openz(file_path) as lf:
        data = pandas.read_csv(lf)
        
    time_stamp = file_path[file_path.rfind('/')+1:]
    time_stamp = '.'.join(time_stamp.split('.')[0:2])
    

    gb = data.groupby(['method','samples'], axis = 0)
    d_group_samples = dict(izip(gb.groups.keys(), [len(x) for x in gb.groups.values()]))

    mean_data = gb.mean()
    std_data = gb.std()

    md = mean_data.reset_index()
    sd = std_data.reset_index()
    methods = md.method.unique()
    losses = md.filter(regex='test-\S+|true-\S+').columns.unique()

    keys = ['alpha', 'gamma', 'k', 'lambda', 'size', 'regularization'] # min imporovement, beta, patience, maxiter?
    assert set(keys).issubset(set(data.columns))
    vals = [data.xs(0)[x] for x in keys] # get first row values for keys
    d_params = dict(izip(keys,vals)) # for output strings later

    #print 'columns in data: ', md.columns.unique()

    if plot_list is None:
        plot_list = methods.tolist()
    
    # remove extra losses
    #print plot_list
    #plot_list.remove('covariance')
    #plot_list.remove('prediction')
    #plot_list.remove('value_prediction')
    #plot_list.remove('prediction-layered')
    #plot_list.remove('value_prediction-layered')
    plot_all_losses(md, sd, losses, methods,  d_group_samples, d_params,  plot_list, 
                                    logspace = logspace)

    fig_path = out_string('./sirf/output/plots/',
        '%s.n_samples_losses%s' % (time_stamp, '' if not logspace else '_logspace'),
        d_params, 
        '.pdf')
    plt.savefig(fig_path)

    all_true_losses = ['true-fullmodel', 'true-model', 'true-bellman', 'true-lsq']
    true_losses = set(md.columns).intersection(set(all_true_losses)) # get losses present in data
    for tl in true_losses:
        plt.clf()
        plot_losses_bymethod(tl, md, sd, methods, d_group_samples, d_params, logspace = logspace)
        fig_path = out_string('./sirf/output/plots/',
        '%s.%s_bymethod%s' % (time_stamp, tl, '' if not logspace else '_logspace'),
        d_params, 
        '.pdf')
        plt.savefig(fig_path) 

def plot_losses_bymethod(loss, md, sd, methods, d_group_samples, d_params, 
                               colors = None, logspace = False):
    if colors is None:
        colors = default_colors

    base_methods = ['covariance', 'prediction', 'value_prediction', 'layered']
    
    cols = numpy.ceil(numpy.sqrt(len(base_methods)))
    rows = numpy.ceil(len(base_methods) / cols)
    rows = int(rows); cols = int(cols)
    
    plt.suptitle(loss)
    for i,bm in enumerate(base_methods):
        plt.subplot(cols, rows, i+1)
        plt.title(bm)
        cc = 0
        for j,meth in enumerate(methods):
            if bm in meth: #if the base method matches
                meth_data = md[md.method == meth]
                meth_std = sd[sd.method == meth]
                samples = meth_data['samples']
                loss_mn = meth_data[loss].values
                loss_std = meth_std[loss].values

                n_samples = [d_group_samples[(meth,s)] for s in samples.values]
                loss_err = loss_std / numpy.sqrt(n_samples)
                
                if logspace:
                    plt.semilogy(samples, loss_mn + 1e-12, label = meth)
                else:
                    plt.fill_between(samples.values, loss_mn - loss_std, loss_mn + loss_std, alpha=0.10, linewidth = 0, color = colors[cc])
                    plt.fill_between(samples.values, loss_mn - loss_err, loss_mn + loss_err, alpha=0.10, linewidth = 0, color = colors[cc])
                    plt.plot(samples, loss_mn, label = meth, color = colors[cc])
                    #plt.axis('off')
                    plt.legend(labelspacing=0.2, prop={'size':8}) 
                cc+=1
        
   


def plot_all_losses(md, sd, losses, methods, d_group_samples, d_params, 
                                plot_list, logspace = False, colors = None):

    if colors is None:
        colors = default_colors
    # plot all losses in subplots    
    cols = numpy.ceil(numpy.sqrt(len(losses)))
    rows = numpy.ceil(len(losses) / cols)
    rows = int(rows); cols = int(cols)
    for i,loss in enumerate(losses):
        plt.subplot(cols, rows, i+1)
        plt.title(loss)
        cc = 0
        for j,meth in enumerate(methods):
            if meth in plot_list:
                meth_data = md[md.method == meth]
                meth_std = sd[sd.method == meth]
                samples = meth_data['samples']
                loss_mn = meth_data[loss].values
                loss_std = meth_std[loss].values

                n_samples = [d_group_samples[(meth,s)] for s in samples.values]
                loss_err = loss_std / numpy.sqrt(n_samples)
                
                if logspace:
                    plt.semilogy(samples, loss_mn + 1e-12, label = meth)
                else:
                    plt.fill_between(samples.values, loss_mn - loss_std, loss_mn + loss_std, alpha=0.10, linewidth = 0, color = colors[cc])
                    plt.fill_between(samples.values, loss_mn - loss_err, loss_mn + loss_err, alpha=0.10, linewidth = 0, color = colors[cc])
                    plt.plot(samples, loss_mn, label = meth, color = colors[cc])
                    #plt.axis('off')
                cc+=1
        if i == 0:
            plt.legend()


if __name__ == '__main__':
    sirf.script(main)

