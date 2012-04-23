from vowpal_porpoise import VW, SimpleInstance, VPLogger
import itertools
import cPickle as pickle
import numpy
import pyfeat



class SimpleModel(object):
    def __init__(self, moniker):
        self.moniker = moniker
        self.log = VPLogger()
        self.model = VW(vw = "/usr/local/bin/vw", \
                        moniker=moniker, \
                        logger=self.log, \
                        **{'passes': 10,
                           'learning_rate': 15,
                           'power_t': 1.0, })

    def train(self, instance_stream):
        """
        Trains the model on the given data stream.
        """
        self.log.info('%s: training' % (self.moniker))
        with self.model.training():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                seen += 1
                if seen % 10000 == 0:
                    self.log.debug('streamed %d instances...' % seen)
            self.log.debug('done streaming.')
        self.log.info('%s: trained on %d data points' % (self.moniker, seen))
        return self

    def predict(self, instance_stream):
        self.log.info('%s: predicting' % self.moniker)
        instances = []
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1

        self.log.info('%s: predicted for %d data points' % (self.moniker, seen))
        predictions = list(self.model.read_predictions_())
        if seen != len(predictions):
            raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                (seen, len(predictions)))
        return itertools.izip(instances, predictions)

# TODO move this where? tools/utils?
def get_value_list(games_path,values_path):
    ''' takes a list of games and a dictionary of values and builds a list of
    (BoardState, value) pairs '''

    with pyfeat.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    with pyfeat.util.openz(values_path) as values_file:
        values = pickle.load(values_file)

    value_list = []

    for game in values.keys():
        try:
            vals = values[game]
            boards = map(specmine.go.BoardState, games[game].grids)
            value_list.extend(zip(boards,vals))
        except KeyError:
            print 'game unkown for ',game

    # remove duplicates
    value_dict = dict(value_list) 
    value_keys = value_dict.keys()
    value_list = zip(value_keys,map(value_dict.__getitem__,value_keys))

    return value_list


if __name__ == '__main__':
    num_instances = 10
    
    values_path = 'data/go_values.fuego.rollouts=256.winrate.alp.pickle'
    games_path = 'data/2010-01.pickle.gz'
    
    example_list = get_value_list(games_path, values_path)[:num_instances]

    boards = numpy.asarray(numpy.round(2*numpy.random.random((num_instances,81))-1), numpy.int8)
    instances = []
    for (board,value) in example_list:
        # make array/list to vw converter for input strings?
        feat_string = str(pyfeat.feature_maps.rect_template(board.grid, edge_max = 2))[1:-1].replace(', ', ' ') 
        instances.append(SimpleInstance(value, 1.0, feat_string))

    for (instance, prediction) in SimpleModel('myexample').train(instances).predict(instances):
        print prediction, instance

