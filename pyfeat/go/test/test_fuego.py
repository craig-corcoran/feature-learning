import nose.tools
import numpy
import pyfeat

def test_fuego_board_basic():
    print 'testing board basics'

    board = pyfeat.go.FuegoBoard()

    nose.tools.assert_equal(board.size, 9)
    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_true(numpy.all(board.grid == 0))

    board.play(0, 8)

    nose.tools.assert_equal(board.to_play, -1)
    nose.tools.assert_equal(board[0, 8], 1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 1)

    board.play(8, 1)

    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_equal(board[0, 8], 1)
    nose.tools.assert_equal(board[8, 1], -1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 2)
    nose.tools.assert_false(board.is_legal(0, 8))
    nose.tools.assert_false(board.is_legal(8, 1))

    board.initialize()

    nose.tools.assert_equal(board.size, 9)
    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_true(numpy.all(board.grid == 0))

def test_fuego_moves_to_grids():
    print 'testing moves to grids'
    moves = [
        (1, 4, 4),
        (-1, 3, 3),
        (1, 2, 2),
        ]
    grids = pyfeat.go.moves_to_grids(moves)

    nose.tools.assert_equal(grids[0, 2, 2], 0)
    nose.tools.assert_equal(grids[0, 3, 3], 0)
    nose.tools.assert_equal(grids[0, 4, 4], 1)
    nose.tools.assert_equal(grids[1, 3, 3], -1)
    nose.tools.assert_equal(grids[2, 2, 2], 1)

def test_fuego_random_player_basic():
    board = pyfeat.go.FuegoBoard()
    player = pyfeat.go.FuegoRandomPlayer(board)

    moves_a = [player.generate_move() for _ in xrange(16)]
    moves_b = [player.generate_move() for _ in xrange(16)]

    nose.tools.assert_not_equal(moves_a, moves_b)

def test_fuego_board_score_simple_endgame():
    scores = []
    
    for i in xrange(64):
        board = pyfeat.go.FuegoBoard()
        player = pyfeat.go.FuegoAveragePlayer(board)
        next_player = pyfeat.go.FuegoRandomPlayer(board)
        
        passed = 0
        num_moves = 0
        while passed < 1: 
            # TODO why aren't suicides being caught? or what is going on when 
            # you wait for two passes (scores switch sign, for example with 
            # avg v rand)
            move = player.generate_move();
            (player, next_player) = (next_player, player)

            if move == (-1,-1):
                passed += 1
            else:
                passed = 0

            if (numpy.zeros((9,9),numpy.int8) == board.grid).all() :
                nose.tools.assert_true( num_moves == 0 )

            (row, column) = move
            board.play(row, column)
            
            num_moves += 1
            if num_moves >= 1000:
                break
            
            if board.to_play == 1:
                nose.tools.assert_true(player.__class__ == pyfeat.go.FuegoAveragePlayer)
            elif board.to_play == -1:
                nose.tools.assert_true(player.__class__ == pyfeat.go.FuegoRandomPlayer)

        print 'final board: ', board.grid    
        print 'number of moves in last game: ', num_moves

        scores.append(board.score_simple_endgame())

    print 'game scores: ', scores
    nose.tools.assert_true(numpy.mean(scores) > 0.0)

if __name__ == "__main__":
    test_fuego_board_score_simple_endgame()

