import nose.tools
import numpy
import pyfeat

def test_fuego_board_basic():
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
    
    player_num = 1;
    for i in xrange(64):
        board = pyfeat.go.FuegoBoard()
        player = pyfeat.go.FuegoCapturePlayer(board)
        next_player = pyfeat.go.FuegoRandomPlayer(board)
        
        while True:
            move = player.generate_move(player = player_num);
            player_num = player_num*-1;

            (player, next_player) = (next_player, player)

            if move is None:
                board.play(-1,-1)
                break
            else:
                #print move, player
                (row, column) = move
                board.play(row, column)

        scores.append(board.score_simple_endgame())

    nose.tools.assert_true(numpy.mean(scores) < 0.0)

