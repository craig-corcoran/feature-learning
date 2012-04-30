import cPickle as pickle
import numpy
import pyfeat
import condor

logger = pyfeat.get_logger(__name__)

def find_values(game, rollouts, player, opponent, error_thresh):
    (M, _) = game.moves.shape
    values = []

    logger.info("evaluating all %i positions in game", M)

    for m in xrange(M):

        board = pyfeat.go.replay_moves(game.moves[:m + 1]) # TODO need to replay each time?
       
        value = pyfeat.go.estimate_value(board, rollouts, player, opponent, error_thresh) 

        values.append(value)

        logger.info("grid %i, has value %f", m, value)

    return numpy.array(values)

def gen_game(player = None, opponent = None, passes = 1):
    ''' plays a game with player v opponent and returns a Game object '''

    board = pyfeat.go.FuegoBoard()

    if player == None:
        player = pyfeat.go.FuegoAveragePlayer(board)

    if opponent == None:
        opponent = pyfeat.go.FuegoAveragePlayer(board)
 
    moves = numpy.zeros((0,3))
    grids = numpy.zeros((0,9,9))
    passed = 0
    while passed < passes:
        if board.to_play == 1: 
            move = player.generate_move()
        elif board.to_play == -1:
            move = opponent.generate_move()
        else:
            assert False
        
        moves = numpy.vstack((moves,(board.to_play,move[0],move[1])))

        if move[0] == -1:
            logger.info("player passing")

            passed += 1 
        else:
            passed = 0

        board.play(move[0], move[1])
      
        grids = numpy.vstack((grids,board.grid[None,:,:]))

        #logger.info("length of grids inside gen game while loop: %i", grids.shape[0])
        #logger.info("length of number of moves %i",len(moves))
        
    logger.info("final board grid : " + str(board.grid))

    score = board.score_simple_endgame()
    winner = 1 if score > 0 else -1 # black = 1, white = -1    

    return pyfeat.go.Game(moves, grids, winner)
        

# TODO want to share/combine values of rollouts on game tree?
@pyfeat.annotations(
    out_path = ("path to write game,value dictionary","positional", None, str),
    num_boards = ("number of boards to evaluate", "option", None, int),
    min_rollouts = ("min monte carlo rollouts to perform", "option", None, int),
    workers = ("number of Condor workers", "option", "w", int),
    replacement = ("boolean flag for keeping multiple copies of the same board", "option", None, bool)
    )
def main(out_path, num_boards = 1000, min_rollouts = 256, workers = 0, replacement = True,
        player = None, opponent = None, error_thresh = 0.0002):

    logger.info("generating state, value pairs using samples from given policy") 

    games = [] 
    grids = numpy.zeros((0,9,9), numpy.int8)
    boards_seen = set()
    while (grids.shape[0] < num_boards) if replacement else (len(boards_seen) < num_boards):
        new_game = gen_game()
        games.append(new_game)
        boards_seen = boards_seen.union(set(map(pyfeat.go.BoardState, new_game.grids)))
        grids = numpy.vstack((grids,new_game.grids))
        logger.info("number of boards in last game's grid: %i", new_game.grids.shape[0])
        logger.info("number of boards gathered: %i", len(boards_seen))
        

    def yield_jobs():
        logger.info("distributing jobs for %i games", len(games))

        for game in games:
            yield (find_values, [game, min_rollouts, None, None, error_thresh])

    evaluated = {}

    for (job, values) in condor.do(yield_jobs(), workers = workers):
        (game, _, _, _, _) = job.args

        evaluated[game] = values
        print 'value of empty board: ', values[0]
        print 'empty board? : ', game.grids[0]
    
    logger.info("about to pickle")
    with pyfeat.util.openz(out_path, "wb") as out_file:
        pickle.dump(evaluated, out_file, protocol = -1)


if __name__ == "__main__":
    pyfeat.script(main)
