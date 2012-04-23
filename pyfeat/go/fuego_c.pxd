cdef extern from "SgSystem.h":
    pass

cdef extern from "SgInit.h":
    void SgInit()

cdef extern from "SgPoint.h":
    ctypedef int SgGrid
    ctypedef int SgPoint

    enum PointConstant:
        SG_PASS

cdef extern from "SgPoint.h" namespace "SgPointUtil":
    SgPoint Pt(int column, int row)
    SgGrid Row(SgPoint point)
    SgGrid Col(SgPoint point)

cdef extern from "SgBlackWhite.h":
    ctypedef int SgBlackWhite

    enum BlackWhiteConstant:
        SG_BLACK
        SG_WHITE

cdef extern from "SgBoardColor.h":
    ctypedef int SgBoardColor

    enum BoardColorConstant:
        SG_EMPTY
        SG_BORDER

cdef extern from "SgDebug.h":
    void SgDebugToNull()

cdef extern from "SgRandom.h":
    cdef void SgRandom_SetSeed "SgRandom::SetSeed"(int seed)

cdef extern from "SgTimeRecord.h":
    cdef cppclass SgTimeRecord:
        SgTimeRecord()
        SgTimeRecord(int moves, double period, double overhead, bint lose_on_time)
        SgTimeRecord(bint one_move_only, double time_for_move)

cdef extern from "SgBWSet.h":
    cdef cppclass SgBWSet:
        SgBWSet()

cdef extern from "GoInit.h":
    void GoInit()

cdef extern from "GoBoard.h":
    enum GoMoveInfoFlag:
        GO_MOVEFLAG_REPETITION
        GO_MOVEFLAG_SUICIDE
        GO_MOVEFLAG_CAPTURING
        GO_MOVEFLAG_ILLEGAL

    cdef cppclass GoBoard:
        GoBoard(int size)
        void Init(int size)
        SgBoardColor GetColor(SgPoint point)
        SgBlackWhite ToPlay()
        SgGrid Size()
        bint IsLegal(int point)
        bint IsLegal(int point, SgBlackWhite player)
        void Play(SgPoint point)
        void Play(SgPoint point, SgBlackWhite player)
        bint LastMoveInfo(GoMoveInfoFlag flag)
        void TakeSnapshot()
        void RestoreSnapshot()

cdef extern from "GoBoardUtil.h" namespace "GoBoardUtil":
    float ScoreSimpleEndPosition(GoBoard& board, float komi, bint no_check) except +
    float ScoreSimpleEndPosition(
        GoBoard& board,
        float komi,
        SgBWSet& safe,
        bint no_check,
        void* score_board, # XXX
        ) except +

cdef extern from "GoPlayer.h":
    cdef cppclass GoPlayer:
        SgPoint GenMove(SgTimeRecord& time, SgBlackWhite player) # time is const

cdef extern from "SpRandomPlayer.h":
    cdef cppclass SpRandomPlayer:
        SpRandomPlayer(GoBoard& board)

cdef extern from "SpAveragePlayer.h":
    cdef cppclass SpAveragePlayer:
        SpAveragePlayer(GoBoard& board)

cdef extern from "SpCapturePlayer.h":
    cdef cppclass SpCapturePlayer:
        SpCapturePlayer(GoBoard& board)

cdef extern from "SpDumbTacticalPlayer.h":
    cdef cppclass SpDumbTacticalPlayer:
        SpDumbTacticalPlayer(GoBoard& board)

cdef extern from "SpGreedyPlayer.h":
    cdef cppclass SpGreedyPlayer:
        SpGreedyPlayer(GoBoard& board)

cdef extern from "SpInfluencePlayer.h":
    cdef cppclass SpInfluencePlayer:
        SpInfluencePlayer(GoBoard& board)

cdef extern from "SpLadderPlayer.h":
    cdef cppclass SpLadderPlayer:
        SpLadderPlayer(GoBoard& board)

cdef extern from "SpLibertyPlayer.h":
    cdef cppclass SpLibertyPlayer:
        SpLibertyPlayer(GoBoard& board)

cdef extern from "SpMinLibPlayer.h":
    cdef cppclass SpMinLibPlayer:
        SpMinLibPlayer(GoBoard& board)

cdef extern from "SpSafePlayer.h":
    cdef cppclass SpSafePlayer:
        SpSafePlayer(GoBoard& board)

