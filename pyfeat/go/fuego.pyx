import pyfeat
import numpy

cimport numpy
cimport cython
cimport pyfeat.go.fuego_c as fuego_c

logger = pyfeat.get_logger(__name__)

cdef fuego_c.SgBlackWhite _player_to_black_white(int player):
    if player == 1:
        return fuego_c.SG_BLACK
    elif player == -1:
        return fuego_c.SG_WHITE
    else:
        assert False

cdef int _black_white_to_int(fuego_c.SgBlackWhite black_white):
    if black_white == fuego_c.SG_BLACK:
        return 1
    elif black_white == fuego_c.SG_WHITE:
        return -1
    else:
        assert False

cdef int _board_color_to_int(fuego_c.SgBoardColor color):
    if color == fuego_c.SG_EMPTY:
        return 0
    elif color == fuego_c.SG_BLACK:
        return 1
    elif color == fuego_c.SG_WHITE:
        return -1
    elif color == fuego_c.SG_BORDER:
        return 9
    else:
        assert False

cdef struct RowColumn:
    int r
    int c

def set_seed(int seed):
    """Set Fuego's internal PRNG seed."""

    fuego_c.SgRandom_SetSeed(seed)

@cython.infer_types(True)
def moves_to_grids(moves, FuegoBoard board = None):
    """
    Replay a game, returning an array of grids.

    Columns in the moves array are (player, row, column).
    """

    moves = numpy.asarray(moves, numpy.int8)

    if board is None:
        board = FuegoBoard()

    cdef int M = moves.shape[0]
    cdef int S = board.size

    cdef numpy.ndarray[numpy.int8_t, ndim = 2] moves_M3 = moves
    cdef numpy.ndarray[numpy.int8_t, ndim = 3] grids_MSS = numpy.empty((M, S, S), numpy.int8)

    for m in xrange(M):
        player = moves_M3[m, 0]
        move_r = moves_M3[m, 1]
        move_c = moves_M3[m, 2]

        if board._get_to_play() != player:
            raise ValueError("player {0} is not to play!".format(player))

        board.play(move_r, move_c)

        for r in xrange(S):
            for c in xrange(S):
                grids_MSS[m, r, c] = board._at(r, c)

    return grids_MSS

@cython.infer_types(True)
cpdef FuegoBoard replay_moves(moves, FuegoBoard board = None):
    """Replay a game, returning the resulting board."""

    moves = numpy.asarray(moves, numpy.int8)

    if board is None:
        board = FuegoBoard()

    cdef int M = moves.shape[0]
    cdef int S = board.size

    cdef numpy.ndarray[numpy.int8_t, ndim = 2] moves_M3 = moves

    for m in xrange(M):
        player = moves_M3[m, 0]
        move_r = moves_M3[m, 1]
        move_c = moves_M3[m, 2]

        if board._get_to_play() != player:
            raise ValueError("player {0} is not to play!".format(player))

        board.play(move_r, move_c)

    return board

@cython.infer_types(True)
def estimate_value(FuegoBoard board, int min_rollouts = 64, 
                FuegoPlayer player = None, FuegoPlayer opponent = None, 
                float error_thresh = 0.0002, int win_rew = 1, int lose_rew = 0):

    """
    Estimate the value of a position. play for at least min_rollouts games, 
    then play until (sample variance / (i+1)) < error_thresh
    """

    cdef int passed
    cdef double mean = 0.0
    cdef double variance = 0.0
    cdef int reward

    if player == None:
        player = FuegoAveragePlayer(board)

    if opponent == None:
        opponent = FuegoAveragePlayer(board)

    board.take_snapshot()

    cdef float error_bound = 0.0
    i = 0
    while (error_bound > error_thresh) | (i < min_rollouts):
           
        board.restore_snapshot()
        
        ## play game ##
        passed = 0
        while passed < 1:
            if board._get_to_play() == 1:
                move = player._generate_move()
            elif board._get_to_play() == -1:
                move = opponent._generate_move()                
            else:
                assert False

            if move.r == -1:        
                passed += 1
            else:
                passed = 0
                
            board.play(move.r,move.c)
        
        score = board.score_simple_endgame()
        
        reward = lose_rew if score < 0 else win_rew # should be > or < (currently: pos score good for black)?
        ####

        mean = (i*mean + reward) / float(i + 1)
        variance = ( i*variance + (reward - mean)**2 ) / float(i + 1)
    
        error_bound = variance / (i + 1)

        i += 1

    #print 'estimated value: ', mean
    #print 'estimated variance: ', variance
    #print 'v*(1-mean) ~ var : ', mean*(1-mean)
    #print 'number of games played: ', i

    return mean


cdef class FuegoBoard(object):
    cdef int _size
    cdef fuego_c.GoBoard* _board

    def __cinit__(self, int size = 9):
        self._size = size
        self._board = new fuego_c.GoBoard(size)

    def __init__(self, int size = 9):
        pass

    def __dealloc__(self):
        del self._board

    def __getitem__(self, indices):
        """Return the value at a position."""

        (row, column) = indices

        return self._at(row, column)

    def initialize(self, int size = 0):
        """Reinitialize the board, optionally altering size."""

        if size == 0:
            size = self._size

        self._board.Init(size)

    def is_legal(self, int row, int column):
        """Is the specified move legal?"""

        return self._board.IsLegal(self._row_column_to_point(row, column))

    cpdef object play(self, int row, int column):
        """Play a single move."""

        self._board.Play(self._row_column_to_point(row, column))
        if self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_SUICIDE):
            raise ValueError("move ({0}, {1}) was suicide in:\n{2}".format(row, column, self.grid))
        if self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_REPETITION):
            raise ValueError("move ({0}, {1}) was illegal repetition in:\n{2}".format(row, column, self.grid))
        if self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_ILLEGAL):
            raise ValueError("move ({0}, {1}) was illegal in:\n{2}".format(row, column, self.grid))

        
    cpdef double score_simple_endgame(self, double komi = 6.5, bint verify_endgame = True):
        """Compute score in the endgame."""

        cdef fuego_c.SgBWSet safe

        return fuego_c.ScoreSimpleEndPosition(self._board[0], komi, safe, not verify_endgame, NULL)

    cpdef object take_snapshot(self):
        self._board.TakeSnapshot()

    cpdef object restore_snapshot(self):
        self._board.RestoreSnapshot()

    @property
    def size(self):
        """The size of the board."""

        return self._board.Size()

    @property
    def to_play(self):
        """The player to play."""

        return self._get_to_play()

    @property
    def grid(self):
        """Array representation of the board state."""

        cdef numpy.ndarray[numpy.int8_t, ndim = 2] grid = numpy.empty((self._size, self._size), numpy.int8)

        for r in xrange(self._size):
            for c in xrange(self._size):
                grid[r, c] = self._at(r, c)

        return grid

    cdef int _get_to_play(self):
        """Return the player to play."""

        return _black_white_to_int(self._board.ToPlay())

    cdef int _at(self, int row, int column):
        """Return the integer board state at a position."""

        point = self._row_column_to_point(row, column)
        color = self._board.GetColor(point)

        return _board_color_to_int(color)

    cdef fuego_c.SgPoint _row_column_to_point(self, int row, int column):
        """Convert a row/column coordinate."""

        assert -1 <= row < self._size
        assert -1 <= column < self._size

        if row == -1:
            return fuego_c.SG_PASS
        else:
            return fuego_c.Pt(column + 1, self._size - row)

    cdef int _point_to_row(self, fuego_c.SgPoint point):
        """Extract the row from a point."""

        return self._size - fuego_c.Row(point)

    cdef int _point_to_column(self, fuego_c.SgPoint point):
        """Extract the column from a point."""

        return fuego_c.Col(point) - 1

cdef class FuegoPlayer(object):
    cdef FuegoBoard _board
    cdef fuego_c.GoPlayer* _player

    def __cinit__(self, FuegoBoard board):
        self._board = board
        self._player = NULL

    def __init__(self, board):
        pass

    def __dealloc__(self):
        if self._player != NULL:
            del self._player

    def generate_move(self, double seconds = 1e6, player = None):
        
        move = self._generate_move(player, seconds)

        return (move.r, move.c) # (-1,-1) returned for pass

    cdef RowColumn _generate_move(self, player = None, double seconds = 1e6):
        """ Generate a move to play. """

        cdef fuego_c.SgBlackWhite black_white
        if player is None:
            black_white = self._board._board.ToPlay()
        else:
            black_white = _player_to_black_white(player)

        cdef fuego_c.SgTimeRecord time = fuego_c.SgTimeRecord(True, seconds)
        cdef fuego_c.SgPoint point = self._player.GenMove(time, black_white)

        if point == fuego_c.SG_PASS:
            return \
                RowColumn(-1, -1)
        else:
            return \
                RowColumn(
                    self._board._point_to_row(point),
                    self._board._point_to_column(point),
                    )

cdef class FuegoRandomPlayer(FuegoPlayer):
    def __cinit__(self, FuegoBoard board):
        self._player = <fuego_c.GoPlayer*>new fuego_c.SpRandomPlayer(board._board[0])

cdef class FuegoAveragePlayer(FuegoPlayer):
    def __cinit__(self, FuegoBoard board):
        self._player = <fuego_c.GoPlayer*>new fuego_c.SpAveragePlayer(board._board[0])

cdef class FuegoCapturePlayer(FuegoPlayer):
    def __cinit__(self, FuegoBoard board):
        self._player = <fuego_c.GoPlayer*>new fuego_c.SpCapturePlayer(board._board[0])

#cdef class FuegoUctPlayer(FuegoPlayer):
    #def __cinit__(self, FuegoBoard board):
        #self._player = <fuego_c.GoPlayer*>new fuego_c.GoUctPlayer(board._board[0])

def _initialize_globals():
    """Configure defaults."""

    fuego_c.SgInit()
    fuego_c.GoInit()
    fuego_c.SgDebugToNull()

    seed = numpy.random.randint(-2e9, 2e9)

    set_seed(seed)

    logger.info("set Fuego internal PRNG seed to %i", seed)

_initialize_globals()

