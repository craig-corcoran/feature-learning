import numpy
import pyfeat
import hash_maps


class RectTemplateMap:

    def __init__(self, edge_max = 9, size = 9, num_bins = 2**18,
            pos_invariant = True, pos_dependent = True, return_count = False):
        
        self.edge_max = edge_max
        self.size = size
        self.num_bins = num_bins
        self.pos_invariant = pos_invariant
        self.pos_dependent = pos_dependent
        self.return_count = return_count

    def __getitem__(self, board):
        
        if board.__class__ is pyfeat.go.BoardState:
            board = board.grid
        elif board.__class is not numpy.ndarray:
            assert False

        return hash_maps.rect_template(board, self.edge_max, self.size, self.num_bins,
                self.pos_invariant, self.pos_dependent, self.return_count)

