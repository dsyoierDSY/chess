import chess
import chess.pgn
import chess.engine
import io
import math
import numpy as np
from collections import defaultdict

# 修改全局参数
SCORE_RANGE = 250  # 进一步缩小评估范围 (Note: new calculate_accuracy doesn't use this directly)
MATE_VALUE_FOR_CALC = 800  # 降低将杀初始价值
MATE_DECAY_PER_PLY_FOR_CALC = 150 # 增加衰减速度
MIN_EFFECTIVE_CP_FOR_MATE = 300 # 长将杀的最低等效厘兵值 (remains 300 as implied)

# 棋局阶段划分参数
OPENING_MOVE_THRESHOLD = 10  # 开局阶段结束的回合数
MIDDLEGAME_MATERIAL_THRESHOLD = 2500  # 中局转残局的物质总量阈值
ENDGAME_MATERIAL_THRESHOLD = 1500     # 深度残局的物质总量阈值

# 各阶段ELO计算权重
PHASE_WEIGHTS = {
    'opening': 0.30,    # 开局权重
    'middlegame': 0.45, # 中局权重
    'endgame': 0.25     # 残局权重
}

# 采样数据阈值
MIN_SAMPLE_SIZE = 5  # 每个阶段最少需要的着法数
SAMPLE_CONFIDENCE_FACTOR = 0.8  # 样本置信度因子，样本越少权重越低

# 强制性走法判断阈值
FORCED_MOVE_THRESHOLD = 150  # 如果第二好的走法比最佳走法差150厘兵以上，则认为是强制性走法
CRITICAL_POSITION_THRESHOLD = 100  # 如果最佳走法与次佳走法差距大于此值，则认为是关键局面
EXCHANGE_DETECTION_THRESHOLD = 80  # 如果走法导致的物质变化超过此值，可能是换子

# 失误和唯一好棋对ELO的影响
BLUNDER_PENALTY = 45  # 每个严重失误的ELO惩罚
MISTAKE_PENALTY = 24  # 每个失误的ELO惩罚
GREAT_MOVE_BONUS = 30  # 每个唯一好棋的ELO奖励
BRILLIANT_MOVE_BONUS = 60  # 每个精彩着法的ELO奖励

# 调整后的参数配置
GREAT_MOVE_THRESHOLD = 80  # ↓ 降低分差要求 (原120)
BRILLIANT_MATERIAL_LOSS = 100  # 调整为至少牺牲1个兵的价值 (原99)
BRILLIANT_SCORE_GAIN = -10     # 调整为需要获得至少0个兵的补偿优势 (原50)
FORCED_MATE_DEPTH = 3          # 强制将杀检测深度
MIN_PV_LENGTH_FOR_BRILLIANT = 3  # 调整为要求更长变例 (原5)
DEEP_ANALYSIS_THRESHOLD = 100  # 触发深度分析的物质损失阈值 (用户文件已改为100)

# 新增补偿系数配置
MATERIAL_COMPENSATION_RATIO = 0.5  # 调整为要求80%的损失补偿 (原0.6)

# 调试模式开关
DEBUG_MODE = True  # 设置为True时会输出调试信息

# 棋子价值配置（用于计算物质平衡）
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # 王的价值不计入物质平衡
}

# 配置Stockfish路径
STOCKFISH_PATH = "path_to_stockfish"  # 请替换为实际的Stockfish路径

# 准确度档次划分
ACCURACY_LEVELS = [
    (0, 50, "完美"),
    (50, 100, "优秀"),
    (100, 150, "良好"),
    (150, 200, "一般"),
    (200, 250, "可接受"),
    (250, 300, "失误"),
    (300, 400, "严重失误"),
    (400, float("inf"), "灾难性失误")
]

# Lichess准确率计算方法
def calculate_win_percentage(centipawns):
    """
    根据Lichess的公式计算胜率百分比
    Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
    """
    # 防止centipawns过大或过小导致math.exp溢出
    exp_arg = -0.00368208 * centipawns
    if exp_arg > 700:  # exp(700) 已经非常大了
        return 0.0 # Win% for large negative cp
    elif exp_arg < -700: # exp(-700) 接近0
        return 100.0 # Win% for large positive cp
    return 50 + 50 * (2 / (1 + math.exp(exp_arg)) - 1)

def calculate_lichess_accuracy(win_percent_before, win_percent_after):
    """
    根据Lichess的公式计算准确率
    Accuracy% = 103.1668 * exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669
    """
    win_diff = max(0, win_percent_before - win_percent_after)  # 确保差值非负
    # 防止win_diff过大导致math.exp溢出或过小
    exp_arg = -0.04354 * win_diff
    if exp_arg > 700: # exp(large_positive) -> large value for accuracy, should be capped
        accuracy_val = 103.1668 * math.exp(700) -3.1669 # effectively infinity
    elif exp_arg < -700: # exp(large_negative) -> 0
        accuracy_val = 103.1668 * 0 - 3.1669
    else:
        accuracy_val = 103.1668 * math.exp(exp_arg) - 3.1669
    return max(0.0, min(100.0, accuracy_val))  # 确保结果在0-100之间

def get_numerical_score_for_accuracy_calc(pov_score_obj):
    """
    将PovScore对象转换为用于准确率计算的数值。
    将杀局面会转换为等效厘兵。
    """
    if pov_score_obj is None:
        return 0
    if pov_score_obj.is_mate():
        mate_in_ply = pov_score_obj.mate()
        if mate_in_ply > 0: # Positive: current player mates
            score = MATE_VALUE_FOR_CALC - (mate_in_ply - 1) * MATE_DECAY_PER_PLY_FOR_CALC
            return max(score, MIN_EFFECTIVE_CP_FOR_MATE)
        else: # Negative: current player is mated
            score = -MATE_VALUE_FOR_CALC + (abs(mate_in_ply) - 1) * MATE_DECAY_PER_PLY_FOR_CALC
            return min(score, -MIN_EFFECTIVE_CP_FOR_MATE)
    else:
        return pov_score_obj.cp if pov_score_obj and hasattr(pov_score_obj, 'cp') else 0

def calculate_accuracy(s_best, s_player):
    """
    使用Lichess的方法计算准确率
    """
    # 特殊情况处理
    if s_player <= -MATE_VALUE_FOR_CALC:  # 玩家走棋导致被将杀
        return 0.0
    if s_best <= -MATE_VALUE_FOR_CALC:    # 最佳着法也无法避免被将杀
        # If player found a better way (less negative score, or even positive) out of a losing position
        return 100.0 if s_player > s_best else 0.0
    
    # 使用Lichess的Win%和Accuracy%计算方法
    win_percent_best = calculate_win_percentage(s_best)
    win_percent_player = calculate_win_percentage(s_player)
    
    return calculate_lichess_accuracy(win_percent_best, win_percent_player)

def get_accuracy_level(accuracy):
    """
    根据准确率确定等级
    返回(等级描述, 符号表示)
    """
    if accuracy >= 98:
        return "完美", "★★★★"
    elif accuracy >= 90:
        return "极佳", "★★★☆"
    elif accuracy >= 80:
        return "优秀", "★★★"
    elif accuracy >= 70:
        return "良好", "★★½"
    elif accuracy >= 60:
        return "一般", "★★"
    elif accuracy >= 50:
        return "欠佳", "★½"
    elif accuracy >= 35:
        return "差", "★"
    else:
        return "严重失误", "☆"

def classify_time_control(clock_str):
    """
    解析PGN中的TimeControl标签示例：
    "600+5" → 10分钟快棋
    "3600+30" → 古典棋
    "180" → 3分钟超快棋
    """
    if not clock_str or clock_str == "-":
        return 'classical'  # 默认无时限
    
    parts = clock_str.split('+')
    try:
        main_time = int(parts[0])
    except ValueError:
        return 'classical' # Fallback if format is unexpected
    
    # 处理突然死亡模式（Sudden Death）
    if '+' not in clock_str: # No increment
        if main_time <= 60 : # 1 min or less often considered bullet in sudden death
            return 'bullet'
        elif main_time <= 180: # up to 3 min
             return 'bullet' # As per user provided logic for "180"
        elif main_time <= 600: # up to 10 min
             return 'blitz'
        elif main_time <= 3600: # up to 60 min
             return 'rapid'
        else:
             return 'classical'

    if main_time <= 179:    # <3分钟
        return 'bullet'
    elif main_time <= 599:  # 3-9.99分钟
        return 'blitz'
    elif main_time <= 3599: # 10-59分钟
        return 'rapid'
    else:                   # 60+分钟
        return 'classical'

def get_estimated_elo(average_accuracy, time_control, phase_accuracies=None, move_quality=None):
    """
    时制敏感等级分预测模型，考虑各阶段精确度和走法质量
    phase_accuracies: 字典，包含各阶段的平均精确度
    move_quality: 字典，包含走法质量统计，如失误数、唯一好棋数等
    """
    base_params = {
        'bullet':    (1.18, 0.85),
        'blitz':     (1.12, 0.90),
        'rapid':     (1.05, 0.95),
        'classical': (1.00, 1.00)
    }
    a, b = base_params.get(time_control, (1.0, 1.0))
    
    # 如果有各阶段精确度，使用加权平均计算
    if phase_accuracies and all(k in phase_accuracies for k in PHASE_WEIGHTS):
        weighted_accuracy = 0
        total_weight = 0
        
        for phase, weight in PHASE_WEIGHTS.items():
            if phase in phase_accuracies and phase_accuracies[phase]['moves'] > 0:
                # 根据样本数量调整权重
                sample_size = phase_accuracies[phase]['moves']
                confidence_factor = min(1.0, sample_size / MIN_SAMPLE_SIZE) * SAMPLE_CONFIDENCE_FACTOR + (1 - SAMPLE_CONFIDENCE_FACTOR)
                adjusted_weight = weight * confidence_factor
                
                weighted_accuracy += phase_accuracies[phase]['accuracy'] * adjusted_weight
                total_weight += adjusted_weight
        
        if total_weight > 0:
            x = max(20.0, min(99.0, float(weighted_accuracy / total_weight)))
        else:
            x = max(20.0, min(99.0, float(average_accuracy)))
    else:
        x = max(20.0, min(99.0, float(average_accuracy)))
    
    # tanh_term calculation
    tanh_arg = (x - 40) / 25
    # Clip tanh_arg to avoid potential overflow with math.exp in tanh if x is very small
    tanh_arg = max(-20, min(20, tanh_arg)) # tanh approaches +/-1 quickly
    try:
        tanh_val = math.tanh(tanh_arg)
    except OverflowError:
        tanh_val = 1.0 if tanh_arg > 0 else -1.0

    # exp_term_1 calculation
    exp_arg_1 = (x - 70) * 0.04
    exp_arg_1 = max(-700, min(700, exp_arg_1)) # Clip to avoid overflow
    try:
        exp_val_1 = math.exp(exp_arg_1)
    except OverflowError:
        exp_val_1 = float('inf') if exp_arg_1 > 0 else 0.0

    # exp_term_2 calculation
    exp_arg_2 = -0.1 * (x - 65)
    exp_arg_2 = max(-700, min(700, exp_arg_2)) # Clip to avoid overflow
    try:
        exp_val_2 = math.exp(exp_arg_2)
        logistic_val = 1 / (1 + exp_val_2)
    except OverflowError:
        logistic_val = 0.0 if exp_arg_2 > 0 else 1.0

    # 基础ELO计算
    base_elo = (
        1200 * tanh_val +
        500 * exp_val_1 * b +
        200 * logistic_val
    ) * a
    
    # 各阶段精确度调整
    if phase_accuracies:
        # 残局表现加成/惩罚
        endgame_bonus = 0
        if 'endgame' in phase_accuracies and phase_accuracies['endgame']['moves'] > MIN_SAMPLE_SIZE:
            endgame_acc = phase_accuracies['endgame']['accuracy']
            if endgame_acc > 85:  # 残局表现优秀
                endgame_bonus = 50
            elif endgame_acc < 60:  # 残局表现较差
                endgame_bonus = -50
        elif 'endgame' in phase_accuracies and 0 < phase_accuracies['endgame']['moves'] < MIN_SAMPLE_SIZE:
            # 残局样本太少，加成减半
            endgame_acc = phase_accuracies['endgame']['accuracy']
            sample_ratio = phase_accuracies['endgame']['moves'] / MIN_SAMPLE_SIZE
            if endgame_acc > 85:  # 残局表现优秀
                endgame_bonus = 50 * sample_ratio
            elif endgame_acc < 60:  # 残局表现较差
                endgame_bonus = -50 * sample_ratio
        
        # 开局准备加成
        opening_bonus = 0
        if 'opening' in phase_accuracies and phase_accuracies['opening']['moves'] > MIN_SAMPLE_SIZE:
            opening_acc = phase_accuracies['opening']['accuracy']
            if opening_acc > 90:  # 开局准备充分
                opening_bonus = 30
        elif 'opening' in phase_accuracies and 0 < phase_accuracies['opening']['moves'] < MIN_SAMPLE_SIZE:
            # 开局样本太少，加成按比例
            opening_acc = phase_accuracies['opening']['accuracy']
            sample_ratio = phase_accuracies['opening']['moves'] / MIN_SAMPLE_SIZE
            if opening_acc > 90:  # 开局准备充分
                opening_bonus = 30 * sample_ratio
        
        # 中局战术能力加成
        middlegame_bonus = 0
        if 'middlegame' in phase_accuracies and phase_accuracies['middlegame']['moves'] > MIN_SAMPLE_SIZE:
            middlegame_acc = phase_accuracies['middlegame']['accuracy']
            if middlegame_acc > 80:  # 中局战术能力强
                middlegame_bonus = 40
        elif 'middlegame' in phase_accuracies and 0 < phase_accuracies['middlegame']['moves'] < MIN_SAMPLE_SIZE:
            # 中局样本太少，加成按比例
            middlegame_acc = phase_accuracies['middlegame']['accuracy']
            sample_ratio = phase_accuracies['middlegame']['moves'] / MIN_SAMPLE_SIZE
            if middlegame_acc > 80:  # 中局战术能力强
                middlegame_bonus = 40 * sample_ratio
        
        # 应用阶段加成
        base_elo += endgame_bonus + opening_bonus + middlegame_bonus
    
    # 根据走法质量调整ELO
    if move_quality:
        # 失误惩罚
        blunder_count = move_quality.get('严重失误', 0) + move_quality.get('灾难性失误', 0)
        mistake_count = move_quality.get('失误', 0)
        
        # 好棋奖励
        great_move_count = move_quality.get('Great', 0)
        brilliant_move_count = move_quality.get('Brilliant', 0)
        
        # 计算总局面数和非强制性走法数
        total_moves = move_quality.get('总步数', 1)
        forced_moves = move_quality.get('强制性走法', 0)
        critical_moves = move_quality.get('关键局面', 0)
        
        # 计算有效局面数（排除强制性走法）
        effective_moves = max(1, total_moves - forced_moves)
        
        # 计算失误率和好棋率（基于有效局面数）
        blunder_rate = blunder_count / effective_moves
        mistake_rate = mistake_count / effective_moves
        great_move_rate = (great_move_count + brilliant_move_count) / max(1, critical_moves)
        
        # 计算ELO调整
        blunder_adjustment = -BLUNDER_PENALTY * blunder_count
        mistake_adjustment = -MISTAKE_PENALTY * mistake_count
        great_move_adjustment = GREAT_MOVE_BONUS * great_move_count
        brilliant_move_adjustment = BRILLIANT_MOVE_BONUS * brilliant_move_count
        
        # 应用调整
        quality_adjustment = blunder_adjustment + mistake_adjustment + great_move_adjustment + brilliant_move_adjustment
        
        # 限制调整范围
        quality_adjustment = max(-200, min(200, quality_adjustment))
        
        base_elo += quality_adjustment
    
    return min(int(base_elo), 3200)

def format_score_for_print(pov_score_obj):
    if pov_score_obj is None: return "(无评分)"
    if pov_score_obj.is_mate():
        return f"(M{abs(pov_score_obj.mate())})"
    else:
        cp_value = pov_score_obj.cp if hasattr(pov_score_obj, 'cp') else None
        if cp_value is None: return "(cp?)"
        return f"({cp_value/100:+.2f})"

# --- Tactical Detection Helpers ---
def detect_double_check_tactic(board): # board is after move
    return board.is_check() and len(board.checkers()) > 1

def detect_pin_tactic(board): # board is after move, player (not board.turn) pinned opponent (board.turn)
    player_color = not board.turn
    opponent_color = board.turn
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, opponent_color):
            if board.is_pinned(opponent_color, sq):
                # Ensure pinner is player_color
                if any(p_sq for p_sq in board.attackers(player_color, sq) if board.pin(opponent_color,sq) == p_sq):
                     return True
    return False

def detect_xray_tactic(board): # player's Q/R/B pins an enemy piece to their K/Q
    """改进的X光攻击检测"""
    player_color = not board.turn
    opponent_color = board.turn
    
    # 检查所有可能的X光攻击者（后、车、象）
    for pinner_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
        for pinner_sq in board.pieces(pinner_type, player_color):
            pinner = board.piece_at(pinner_sq)
            if not pinner:
                continue
                
            # 检查所有可能被牵制的棋子
            for pinned_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for pinned_sq in board.pieces(pinned_type, opponent_color):
                    if not board.is_pinned(opponent_color, pinned_sq):
                        continue
                        
                    # 确认牵制关系
                    if board.pin(opponent_color, pinned_sq) == pinner_sq:
                        # 检查是否牵制到国王
                        king_sq = board.king(opponent_color)
                        if king_sq and king_sq in chess.SquareSet(chess.ray(pinner_sq, pinned_sq)):
                            # 确认国王在射线上
                            if board.piece_at(king_sq) and board.piece_at(king_sq).piece_type == chess.KING:
                                # 确认牵制棋子在攻击者和国王之间
                                between = chess.SquareSet.between(pinner_sq, king_sq)
                                if pinned_sq in between:
                                    return True
                                    
                        # 检查是否牵制到后
                        for queen_sq in board.pieces(chess.QUEEN, opponent_color):
                             if queen_sq != pinned_sq and queen_sq in chess.SquareSet(chess.ray(pinner_sq, pinned_sq)):
                                # 确认牵制棋子在攻击者和后之间
                                between = chess.SquareSet.between(pinner_sq, queen_sq)
                                if pinned_sq in between:
                                 return True
    return False
    
def detect_discovered_attack_tactic(board, last_move): # board is after move
    """改进的闪击战术检测"""
    if not last_move: 
        return False
        
    # 获取移动前后的方格
    from_sq = last_move.from_square
    to_sq = last_move.to_square
    
    player_color = not board.turn  # 刚走棋的一方
    
    # 检查是否为发现将军
    if board.is_check():
        checkers = board.checkers()
        if len(checkers) == 1: # 单将军（非双将军）
            checking_piece_sq = checkers.pop()
            # 如果将军的棋子不是刚走的棋子，则可能是闪击
            if checking_piece_sq != to_sq:
                # 确保将军棋子是长距离攻击棋子（车、象、后）
                checking_piece = board.piece_at(checking_piece_sq)
                if checking_piece and checking_piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                    # 检查将军棋子是否在移动前被阻挡
                    king_sq = board.king(board.turn)
                    if king_sq:
                        # 检查移动前的棋子是否在将军棋子和国王之间
                        ray = chess.SquareSet(chess.ray(checking_piece_sq, king_sq))
                    if from_sq in ray:
                        return True
    
    # 检查其他发现攻击（不仅限于将军）
    # 获取所有可能的攻击者（长距离攻击棋子）
    attackers = board.pieces(chess.ROOK, player_color) | board.pieces(chess.BISHOP, player_color) | board.pieces(chess.QUEEN, player_color)
    
    # 检查每个攻击者是否因为棋子移动而获得了新的攻击线
    for attacker_sq in attackers:
        # 检查攻击者是否在移动前的棋子所在方格的射线上
        if from_sq in chess.SquareSet(chess.ray(attacker_sq, to_sq)):
            # 检查攻击者现在是否可以攻击到重要棋子（国王除外，已在上面检查）
            for target_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                for target_sq in board.pieces(target_type, board.turn):
                    if target_sq in chess.SquareSet(chess.ray(attacker_sq, from_sq)):
                        # 确认没有其他棋子阻挡
                        between = chess.SquareSet.between(attacker_sq, target_sq)
                        if not any(board.piece_at(sq) for sq in between):
                            return True
    
    return False

def detect_fork_tactic(board, last_move):
    """检测叉击战术"""
    if not last_move:
        return False
        
    to_sq = last_move.to_square
    player_color = not board.turn  # 刚走棋的一方
    opponent_color = board.turn
    
    # 获取移动到的棋子
    piece = board.piece_at(to_sq)
    if not piece or piece.color != player_color:
        return False
    
    # 计算该棋子能攻击到的对方重要棋子数量
    attacked_pieces = 0
    attacked_values = 0
    
    # 获取该棋子的攻击范围
    attacks = board.attacks(to_sq)
    
    # 检查攻击范围内的对方棋子
    for attack_sq in attacks:
        target = board.piece_at(attack_sq)
        if target and target.color == opponent_color:
            attacked_pieces += 1
            attacked_values += PIECE_VALUES.get(target.piece_type, 0)
    
    # 如果攻击到两个或更多棋子，或者攻击价值超过车的价值，则认为是叉击
    return attacked_pieces >= 2 or attacked_values >= PIECE_VALUES[chess.ROOK]

TACTICAL_PATTERNS = {
    "X光攻击": detect_xray_tactic, # More specific pin by Q/R/B to K/Q
    "双将检测": detect_double_check_tactic,
    "闪击战术": detect_discovered_attack_tactic, # Primarily discovered checks
    "牵制战术": detect_pin_tactic, # General pin of an opponent piece
    "叉击战术": detect_fork_tactic  # 新增叉击检测
}

def calculate_material(board, color=None):
    total = 0
    if color is None: # Calculate for all pieces
        for square, piece in board.piece_map().items():
            total += PIECE_VALUES.get(piece.piece_type, 0)
    else: # Calculate for a specific color
        for piece_type in PIECE_VALUES:
            if piece_type == chess.KING: continue
            total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total

def detect_material_loss(board_before, move, engine=None):
    """
    精确计算走子方的物质损失（返回正值表示净损失，负值表示净收益）
    
    参数：
    board_before -- 走棋前的棋盘状态
    move -- 要评估的走法
    engine -- 可选的引擎实例
    
    返回：
    走子方的净物质变化（正值为损失，负值为收益）
    """
    # 静态物质变化计算（基于棋子价值）
    def static_material_change():
        board_after = board_before.copy()
        board_after.push(move)
        
        # 计算玩家和对手的物质量变化
        player_color = board_before.turn
        player_loss = calculate_material(board_before, player_color) - calculate_material(board_after, player_color)
        opponent_loss = calculate_material(board_before, not player_color) - calculate_material(board_after, not player_color)
        
        # 净变化 = 敌方获利 - 我方损失（正数表示我方净损失）
        return opponent_loss - player_loss

    # 动态物质变化计算（基于引擎评分）
    def dynamic_material_change(eng):
        try:
            board_after = board_before.copy()
            board_after.push(move)
            
            # 深度分析走棋前局面
            analysis_before = eng.analyse(
                board_before, 
                chess.engine.Limit(depth=12), 
                info=chess.engine.INFO_SCORE
            )
            score_before = get_numerical_score_for_accuracy_calc(
                analysis_before.get("score").pov(board_before.turn)
            ) if analysis_before.get("score") else 0
            
            # 深度分析走棋后局面
            analysis_after = eng.analyse(
                board_after, 
                chess.engine.Limit(depth=12), 
                info=chess.engine.INFO_SCORE
            )
            score_after = get_numerical_score_for_accuracy_calc(
                analysis_after.get("score").pov(board_before.turn)
            ) if analysis_after.get("score") else 0
            
            # 动态损失 = 走棋后的评分变化（正数表示损失）
            return score_before - score_after
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"[DEBUG] 动态物质损失计算失败: {str(e)[:60]}...")
            return 0

    # 核心计算流程
    static_change = static_material_change()
    dynamic_change = dynamic_material_change(engine) if engine else 0
    
    # 结果选择策略（取最严重的损失值）
    if engine:
        # 比较绝对值取最大损失（保留符号）
        final_change = max([static_change, dynamic_change], key=lambda x: abs(x))
    else:
        final_change = static_change

    # 调试信息增强
    if DEBUG_MODE:
        change_type = "收益" if final_change < 0 else "损失"
        source = "静态" if final_change == static_change else "动态"
        print(f"[DEBUG] 物质变化: {abs(final_change)}厘兵({change_type}) "
              f"来源={source} (静态={static_change}, 动态={dynamic_change})")

    return final_change
def detect_mate_sequence(board, engine, current_player_color, depth=FORCED_MATE_DEPTH):
    try:
        # Analyze from the perspective of the current player
        analysis = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE)
        score = analysis.get("score")
        if score and score.pov(current_player_color).is_mate():
            mate_in_plies = score.pov(current_player_color).mate()
            if mate_in_plies > 0: # Player can mate
                return mate_in_plies
    except Exception as e:
        # print(f"Error in detect_mate_sequence: {e}")
        pass
    return 0

def analyze_brilliant_features(board_before_move, move, engine, current_player_color):
    """增强的精彩着法特征分析"""
    features = {}
    board_after_move = board_before_move.copy()
    board_after_move.push(move)

    try:
        # 提高分析深度
        # 评估走棋前的局面
        analysis_before = engine.analyse(board_before_move, chess.engine.Limit(depth=16), info=chess.engine.INFO_SCORE)
        prev_eval_obj = analysis_before.get("score")
        prev_eval = get_numerical_score_for_accuracy_calc(prev_eval_obj.pov(current_player_color) if prev_eval_obj else None)
        
        # 评估走棋后的局面（更深的分析）
        # 修复 multipv 参数传递错误
        analysis_after = engine.analyse(board_after_move, chess.engine.Limit(depth=20), multipv=1, info=chess.engine.INFO_SCORE)
        new_eval_obj = analysis_after[0].get("score") if isinstance(analysis_after, list) else analysis_after.get("score")

        new_eval = get_numerical_score_for_accuracy_calc(new_eval_obj.pov(current_player_color) if new_eval_obj else None)
        features['eval_diff'] = new_eval - prev_eval

        if isinstance(analysis_after, list) and analysis_after:
             features['main_line'] = analysis_after[0].get('pv', [])
        else:
             features['main_line'] = []

        # 更深的将杀检测
        features['mate_in'] = detect_mate_sequence(board_after_move, engine, current_player_color, FORCED_MATE_DEPTH * 3) 
        
        # 战术模式检测
        detected_patterns = []
        for pattern_name, check_func in TACTICAL_PATTERNS.items():
            try:
                if pattern_name == "闪击战术": # 需要上一步走法
                    if check_func(board_after_move, move):
                        detected_patterns.append(pattern_name)
                elif pattern_name == "叉击战术": # 需要上一步走法
                    if check_func(board_after_move, move):
                        detected_patterns.append(pattern_name)
                elif check_func(board_after_move):
                    detected_patterns.append(pattern_name)
            except Exception as e: # 捕获战术检测中的错误
                if DEBUG_MODE:
                    print(f"[DEBUG] 战术检测 {pattern_name} 失败: {e}")
                continue
        features['tactics'] = detected_patterns
        
        # 调试输出
        if DEBUG_MODE:
            print(f"[DEBUG] 精彩着法特征分析结果:")
            print(f"[DEBUG] - 评分差异: {features['eval_diff']}")
            print(f"[DEBUG] - 将杀步数: {features['mate_in']}")
            print(f"[DEBUG] - 检测到的战术: {features['tactics']}")
            print(f"[DEBUG] - 主要变例长度: {len(features['main_line'])}")
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] 精彩着法特征分析失败: {e}")
        # 初始化特征为默认值
        features.setdefault('eval_diff', 0)
        features.setdefault('main_line', [])
        features.setdefault('mate_in', 0)
        features.setdefault('tactics', [])
    return features

def is_great_move(player_move_uci, current_player_color, engine_scores_pov, multipv_results_raw):
    """改进后的唯一好棋检测"""
    if len(multipv_results_raw) < 2:
        return False
    
    best_engine_move_uci = multipv_results_raw[0]['pv'][0].uci()
    if player_move_uci != best_engine_move_uci:
        return False  # 玩家必须选择绝对最佳走法
    
    # 计算评分差异
    s_best = engine_scores_pov[0]
    s_second_best = engine_scores_pov[1]
    score_diff = s_best - s_second_best

    # 条件 1：绝对差异
    abs_diff_cond = score_diff >= GREAT_MOVE_THRESHOLD

    # 条件 2：相对差异
    relative_diff_cond = False
    if abs(s_best) > 50:  # 确保当前评分足够显著
        relative_diff_cond = (score_diff / abs(s_best)) > 0.3

    if DEBUG_MODE:
        print(f"[DEBUG] 唯一好棋检测: player_move_uci={player_move_uci}, best_move_uci={best_engine_move_uci}")
        print(f"[DEBUG] - s_best={s_best}, s_second_best={s_second_best}, score_diff={score_diff}")
        print(f"[DEBUG] - abs_diff_cond={abs_diff_cond}, relative_diff_cond={relative_diff_cond}")

    return abs_diff_cond or relative_diff_cond

def is_brilliant_move(board_before_move, move, engine, material_loss, current_player_color):
    """改进后的精彩着法检测"""
    if material_loss < BRILLIANT_MATERIAL_LOSS:
        if DEBUG_MODE:
            print(f"[DEBUG] 物质损失 {material_loss} 未达到精彩着法阈值 {BRILLIANT_MATERIAL_LOSS}")
        return False

    features = analyze_brilliant_features(board_before_move, move, engine, current_player_color)

    # 条件 1：评分差异
    score_condition = features.get('eval_diff', 0) >= BRILLIANT_SCORE_GAIN

    # 条件 2：补偿比例
    compensation_ratio = -(features.get('eval_diff', 0) / material_loss if material_loss > 0 else 0)
    ratio_condition = compensation_ratio >= MATERIAL_COMPENSATION_RATIO

    # 条件 3：战术检测
    pattern_condition = any(p in features.get('tactics', []) for p in ["叉击战术", "双将检测", "X光攻击", "闪击战术"])

    # 条件 4：变例长度
    pv_length_condition = len(features.get('main_line', [])) >= MIN_PV_LENGTH_FOR_BRILLIANT

    if DEBUG_MODE:
        print(f"[DEBUG] 精彩着法检测: move={move}")
        print(f"[DEBUG] - eval_diff={features.get('eval_diff', 0)}, mate_in={features.get('mate_in', 0)}")
        print(f"[DEBUG] - compensation_ratio={compensation_ratio}, tactics={features.get('tactics', [])}")
        print(f"[DEBUG] - score_condition={score_condition}, ratio_condition={ratio_condition}")
        print(f"[DEBUG] - pattern_condition={pattern_condition}, pv_length_condition={pv_length_condition}")

    return score_condition and (ratio_condition or pattern_condition or pv_length_condition)

def analyze_move_quality(board, move, engine, current_player_color, multipv_results_raw, material_loss, accuracy):
    """分析着法质量并返回特殊标签"""
    labels = []
    
    # 准备参数
    player_move_uci = move.uci()
    # 从玩家视角的引擎评分
    engine_scores_pov = [get_numerical_score_for_accuracy_calc(r.get("score").pov(current_player_color)) for r in multipv_results_raw if r.get("score")]

    if DEBUG_MODE:
        print(f"[DEBUG] 分析着法质量: move={player_move_uci}, material_loss={material_loss}, accuracy={accuracy}")
        print(f"[DEBUG] - engine_scores_pov={engine_scores_pov}")
        print(f"[DEBUG] - multipv_results_raw={multipv_results_raw}")

    # 精彩着法检测（优先级最高）
    # 使用动态物质损失计算
    dynamic_material_loss = detect_material_loss(board, move, engine)
    effective_material_loss = max(material_loss, dynamic_material_loss)
    
    if effective_material_loss >= BRILLIANT_MATERIAL_LOSS:
        if is_brilliant_move(board, move, engine, effective_material_loss, current_player_color):
            labels.append("Brilliant!!")
            # 添加详细信息
            features = analyze_brilliant_features(board, move, engine, current_player_color)
            if features.get('mate_in', 0) > 0:
                labels.append(f"Mate in {features.get('mate_in')}")
            if features.get('tactics', []):
                labels.append("/".join(features.get('tactics')[:2]))  # 显示最多两种战术

    # 唯一好棋检测（如果不是精彩着法）
    if not any(l.startswith("Brilliant") for l in labels):
        if len(engine_scores_pov) >= 2 and is_great_move(player_move_uci, current_player_color, engine_scores_pov, multipv_results_raw):
            labels.append("Great!")
            # 可选：添加战术标签
            features = analyze_brilliant_features(board, move, engine, current_player_color)
            if features.get('tactics', []):
                labels.append(features.get('tactics')[0])  # 添加检测到的第一种战术

    # 激励性标签（如果没有其他特殊标签）
    if not labels:
        if 70 < accuracy < 85 and effective_material_loss > 100:
            labels.append("Good effort!")
        elif accuracy > 90:
            labels.append("Precise!")

    if DEBUG_MODE:
        print(f"[DEBUG] 分析结果: labels={labels}")

    return labels

# 添加识别棋局阶段的函数
def identify_game_phase(board, move_number):
    """
    根据回合数和棋盘上的物质总量确定棋局阶段
    返回: 'opening', 'middlegame', 或 'endgame'
    """
    # 开局判断：基于回合数
    if move_number <= OPENING_MOVE_THRESHOLD:
        return 'opening'
    
    # 计算棋盘上的总物质量
    total_material = calculate_material(board)
    
    # 残局判断：基于物质总量
    if total_material <= ENDGAME_MATERIAL_THRESHOLD:
        return 'endgame'
    elif total_material <= MIDDLEGAME_MATERIAL_THRESHOLD:
        return 'middlegame_to_endgame'  # 中残局过渡阶段
    else:
        return 'middlegame'

def analyze_game(pgn_text, engine_path):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 4, "Hash": 256, "UCI_LimitStrength": False}) # 确保全功率
    
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        print("无法读取PGN对局")
        engine.quit()
        return
    
    time_control_str = game.headers.get("TimeControl", "")
    time_control = classify_time_control(time_control_str)
    
    print(f"对局信息:")
    print(f"白方: {game.headers.get('White', 'Unknown')}")
    print(f"黑方: {game.headers.get('Black', 'Unknown')}")
    print(f"结果: {game.headers.get('Result', 'Unknown')}")
    print(f"时制: {time_control_str} ({time_control})")
    print("\n走步分析:")
    print("(评分单位：厘兵，100厘兵 = 1个兵的优势)")
    print("(每步主分析时间：1.0秒，多PV，特殊着法深度分析)")
    print("(使用Lichess准确率计算方法)")
    
    board = chess.Board()
    white_total_accuracy = 0
    black_total_accuracy = 0
    white_moves = 0
    black_moves = 0

    # 各阶段精确度统计
    white_phase_stats = {
        'opening': {'accuracy': 0, 'moves': 0},
        'middlegame': {'accuracy': 0, 'moves': 0},
        'middlegame_to_endgame': {'accuracy': 0, 'moves': 0},
        'endgame': {'accuracy': 0, 'moves': 0}
    }
    black_phase_stats = {
        'opening': {'accuracy': 0, 'moves': 0},
        'middlegame': {'accuracy': 0, 'moves': 0},
        'middlegame_to_endgame': {'accuracy': 0, 'moves': 0},
        'endgame': {'accuracy': 0, 'moves': 0}
    }

    accuracy_levels_for_stats = ["完美", "极佳", "优秀", "良好", "一般", "欠佳", "差", "严重失误"]
    move_quality_stats = {
        chess.WHITE: {level: 0 for level in accuracy_levels_for_stats},
        chess.BLACK: {level: 0 for level in accuracy_levels_for_stats}
    }
    move_quality_stats[chess.WHITE]["总步数"] = 0
    move_quality_stats[chess.BLACK]["总步数"] = 0
    
    special_moves_stats = {
        chess.WHITE: {"Brilliant": 0, "Great": 0, "Good effort": 0, "Precise": 0},
        chess.BLACK: {"Brilliant": 0, "Great": 0, "Good effort": 0, "Precise": 0}
    }
    
    # 初始化走法质量统计
    white_move_quality = {'总步数': 0, '强制性走法': 0, '关键局面': 0}
    black_move_quality = {'总步数': 0, '强制性走法': 0, '关键局面': 0}
    
    # 初始化各级别失误统计
    for level in accuracy_levels_for_stats:
        white_move_quality[level] = 0
        black_move_quality[level] = 0
    
    # 初始化特殊走法统计
    for special_move in ["Brilliant", "Great", "Good effort", "Precise"]:
        white_move_quality[special_move] = 0
        black_move_quality[special_move] = 0
    
    node = game
    last_move_for_board = None # For discovered attack
    move_number = 0

    while node.variations:
        current_player_color = board.turn
        move_number = (node.variation(0).ply() + 1) // 2  # 计算当前回合数
        
        # 确定当前棋局阶段
        current_phase = identify_game_phase(board, move_number)
        
        # MultiPV analysis of the current position
        # 修复 multipv 参数传递错误
        multipv_analysis_results_raw = engine.analyse(board, chess.engine.Limit(time=1.0, nodes=2000000), multipv=3, info=chess.engine.INFO_ALL)
        
        s_best_pov = 0 # Best score from current player's POV
        if multipv_analysis_results_raw:
            best_line_score_obj = multipv_analysis_results_raw[0].get("score")
            if best_line_score_obj:
                 s_best_pov = get_numerical_score_for_accuracy_calc(best_line_score_obj.pov(current_player_color))
        else:
            print(f"警告: 无法分析局面: {board.fen()}")
            # Fallback: get a quick eval for s_best if multipv fails
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=10), info=chess.engine.INFO_SCORE)
                score_obj = info.get("score")
                if score_obj:
                    s_best_pov = get_numerical_score_for_accuracy_calc(score_obj.pov(current_player_color))
            except:
                s_best_pov = 0


        current_pgn_node = node.variation(0)
        pgn_move = current_pgn_node.move
        
        # 计算物质损失（静态）
        material_loss = detect_material_loss(board, pgn_move)
        
        board_after_player = board.copy()
        board_after_player.push(pgn_move)
        
        # Quick eval for player's move
        info_after_player_analysis = engine.analyse(board_after_player, chess.engine.Limit(time=0.1, nodes=200000), info=chess.engine.INFO_SCORE)
        s_player_pov = 0
        player_move_score_print_obj_pov = None
        
        player_score_obj_raw = info_after_player_analysis.get("score")
        if player_score_obj_raw:
            player_move_score_print_obj_pov = player_score_obj_raw.pov(current_player_color)
            s_player_pov = get_numerical_score_for_accuracy_calc(player_move_score_print_obj_pov)

        accuracy = calculate_accuracy(s_best_pov, s_player_pov)
        
        win_percent_before = calculate_win_percentage(s_best_pov)
        win_percent_after = calculate_win_percentage(s_player_pov)
        win_percent_diff = win_percent_before - win_percent_after
        
        # 检查是否为强制性走法
        engine_scores_pov = [get_numerical_score_for_accuracy_calc(r.get("score").pov(current_player_color)) for r in multipv_analysis_results_raw if r.get("score")]
        forced_move = is_forced_move(engine_scores_pov, multipv_analysis_results_raw, board, pgn_move)
        critical_position = is_critical_position(engine_scores_pov)
        
        # 分析着法质量（使用动态物质损失计算）
        quality_labels = analyze_move_quality(board, pgn_move, engine, current_player_color, multipv_analysis_results_raw, material_loss, accuracy)
        
        # 更新特殊走法统计
        for label in quality_labels:
            if "Brilliant!!" in label:
                special_moves_stats[current_player_color]["Brilliant"] += 1
                if current_player_color == chess.WHITE:
                    white_move_quality["Brilliant"] += 1
                else:
                    black_move_quality["Brilliant"] += 1
            elif "Great!" in label:
                special_moves_stats[current_player_color]["Great"] += 1
                if current_player_color == chess.WHITE:
                    white_move_quality["Great"] += 1
                else:
                    black_move_quality["Great"] += 1
            elif "Good effort!" in label:
                special_moves_stats[current_player_color]["Good effort"] += 1
                if current_player_color == chess.WHITE:
                    white_move_quality["Good effort"] += 1
                else:
                    black_move_quality["Good effort"] += 1
            elif "Precise!" in label:
                special_moves_stats[current_player_color]["Precise"] += 1
                if current_player_color == chess.WHITE:
                    white_move_quality["Precise"] += 1
                else:
                    black_move_quality["Precise"] += 1
        
        ply_count = current_pgn_node.ply()
        player_name_for_print = ""
        level, symbol = get_accuracy_level(accuracy)

        if current_player_color == chess.WHITE:
            white_total_accuracy += accuracy
            white_moves += 1
            player_name_for_print = "白方"
            move_quality_stats[chess.WHITE][level] += 1
            move_quality_stats[chess.WHITE]["总步数"] += 1
            
            # 更新白方各阶段精确度统计
            white_phase_stats[current_phase]['accuracy'] += accuracy
            white_phase_stats[current_phase]['moves'] += 1
            
            # 更新白方走法质量统计
            white_move_quality['总步数'] += 1
            white_move_quality[level] += 1
            if forced_move:
                white_move_quality['强制性走法'] += 1
            if critical_position:
                white_move_quality['关键局面'] += 1
        else:
            black_total_accuracy += accuracy
            black_moves += 1
            player_name_for_print = "黑方"
            move_quality_stats[chess.BLACK][level] += 1
            move_quality_stats[chess.BLACK]["总步数"] += 1
            
            # 更新黑方各阶段精确度统计
            black_phase_stats[current_phase]['accuracy'] += accuracy
            black_phase_stats[current_phase]['moves'] += 1
            
            # 更新黑方走法质量统计
            black_move_quality['总步数'] += 1
            black_move_quality[level] += 1
            if forced_move:
                black_move_quality['强制性走法'] += 1
            if critical_position:
                black_move_quality['关键局面'] += 1
        
        san_move_str = board.san(pgn_move)
        
        # For next iteration's discovered attack check
        last_move_for_board = pgn_move 
        board.push(pgn_move) # IMPORTANT: board is updated AFTER all analysis related to the position *before* this move
        node = current_pgn_node

        move_display_number = (ply_count + 1) // 2 if current_player_color == chess.WHITE else ply_count // 2
        
        score_str_for_print = format_score_for_print(player_move_score_print_obj_pov)
        win_percent_str = f"胜率: {win_percent_after:.1f}% (变化: {-win_percent_diff:+.1f}%)"
        phase_str = {
            'opening': '开局',
            'middlegame': '中局',
            'middlegame_to_endgame': '中残局',
            'endgame': '残局'
        }.get(current_phase, '')
        
        forced_str = " [强制]" if forced_move else ""
        critical_str = " [关键]" if critical_position else ""
        
        print(f"{move_display_number}. {player_name_for_print} {san_move_str:10} 准确率: {accuracy:5.1f}% {score_str_for_print:10} {win_percent_str} {level} {symbol} [{phase_str}]{forced_str}{critical_str}")
        
        if quality_labels:
            print(f"     {' '.join(quality_labels)}")

    # 计算各阶段平均精确度
    for phase in white_phase_stats:
        if white_phase_stats[phase]['moves'] > 0:
            white_phase_stats[phase]['accuracy'] /= white_phase_stats[phase]['moves']
    
    for phase in black_phase_stats:
        if black_phase_stats[phase]['moves'] > 0:
            black_phase_stats[phase]['accuracy'] /= black_phase_stats[phase]['moves']

    white_avg_accuracy = white_total_accuracy / white_moves if white_moves > 0 else 0
    black_avg_accuracy = black_total_accuracy / black_moves if black_moves > 0 else 0
    
    # 合并中残局到中局和残局的统计
    if white_phase_stats['middlegame_to_endgame']['moves'] > 0:
        # 将中残局数据按比例分配到中局和残局
        mid_end_moves = white_phase_stats['middlegame_to_endgame']['moves']
        mid_end_acc = white_phase_stats['middlegame_to_endgame']['accuracy']
        
        # 60%分配给中局，40%分配给残局
        white_phase_stats['middlegame']['accuracy'] = (white_phase_stats['middlegame']['accuracy'] * white_phase_stats['middlegame']['moves'] + mid_end_acc * mid_end_moves * 0.6) / (white_phase_stats['middlegame']['moves'] + mid_end_moves * 0.6) if (white_phase_stats['middlegame']['moves'] + mid_end_moves * 0.6) > 0 else 0
        white_phase_stats['middlegame']['moves'] += mid_end_moves * 0.6
        
        white_phase_stats['endgame']['accuracy'] = (white_phase_stats['endgame']['accuracy'] * white_phase_stats['endgame']['moves'] + mid_end_acc * mid_end_moves * 0.4) / (white_phase_stats['endgame']['moves'] + mid_end_moves * 0.4) if (white_phase_stats['endgame']['moves'] + mid_end_moves * 0.4) > 0 else 0
        white_phase_stats['endgame']['moves'] += mid_end_moves * 0.4
    
    if black_phase_stats['middlegame_to_endgame']['moves'] > 0:
        mid_end_moves = black_phase_stats['middlegame_to_endgame']['moves']
        mid_end_acc = black_phase_stats['middlegame_to_endgame']['accuracy']
        
        black_phase_stats['middlegame']['accuracy'] = (black_phase_stats['middlegame']['accuracy'] * black_phase_stats['middlegame']['moves'] + mid_end_acc * mid_end_moves * 0.6) / (black_phase_stats['middlegame']['moves'] + mid_end_moves * 0.6) if (black_phase_stats['middlegame']['moves'] + mid_end_moves * 0.6) > 0 else 0
        black_phase_stats['middlegame']['moves'] += mid_end_moves * 0.6
        
        black_phase_stats['endgame']['accuracy'] = (black_phase_stats['endgame']['accuracy'] * black_phase_stats['endgame']['moves'] + mid_end_acc * mid_end_moves * 0.4) / (black_phase_stats['endgame']['moves'] + mid_end_moves * 0.4) if (black_phase_stats['endgame']['moves'] + mid_end_moves * 0.4) > 0 else 0
        black_phase_stats['endgame']['moves'] += mid_end_moves * 0.4
    
    # 移除中残局统计，只保留开局、中局、残局
    white_phase_stats_final = {k: v for k, v in white_phase_stats.items() if k != 'middlegame_to_endgame'}
    black_phase_stats_final = {k: v for k, v in black_phase_stats.items() if k != 'middlegame_to_endgame'}
    
    print("\n对局总结:")
    print(f"白方平均准确率: {white_avg_accuracy:.1f}%")
    white_overall_level, white_symbol = get_accuracy_level(white_avg_accuracy)
    print(f"白方整体表现: {white_overall_level} {white_symbol}")
    
    print(f"黑方平均准确率: {black_avg_accuracy:.1f}%")
    black_overall_level, black_symbol = get_accuracy_level(black_avg_accuracy)
    print(f"黑方整体表现: {black_overall_level} {black_symbol}")

    # 打印各阶段精确度
    print("\n各阶段精确度:")
    print("白方:")
    for phase, stats in white_phase_stats_final.items():
        if stats['moves'] > 0:
            phase_name = {
                'opening': '开局',
                'middlegame': '中局',
                'endgame': '残局'
            }.get(phase, phase)
            phase_level, phase_symbol = get_accuracy_level(stats['accuracy'])
            sample_warning = f" (样本较少: {stats['moves']:.1f}步)" if stats['moves'] < MIN_SAMPLE_SIZE else ""
            print(f"  {phase_name}: {stats['accuracy']:.1f}% ({stats['moves']:.1f}步) - {phase_level} {phase_symbol}{sample_warning}")
    
    print("黑方:")
    for phase, stats in black_phase_stats_final.items():
        if stats['moves'] > 0:
            phase_name = {
                'opening': '开局',
                'middlegame': '中局',
                'endgame': '残局'
            }.get(phase, phase)
            phase_level, phase_symbol = get_accuracy_level(stats['accuracy'])
            sample_warning = f" (样本较少: {stats['moves']:.1f}步)" if stats['moves'] < MIN_SAMPLE_SIZE else ""
            print(f"  {phase_name}: {stats['accuracy']:.1f}% ({stats['moves']:.1f}步) - {phase_level} {phase_symbol}{sample_warning}")

    # 使用考虑各阶段和走法质量的ELO计算
    white_estimated_elo = get_estimated_elo(white_avg_accuracy, time_control, white_phase_stats_final, white_move_quality)
    black_estimated_elo = get_estimated_elo(black_avg_accuracy, time_control, black_phase_stats_final, black_move_quality)

    print(f"\n预估等级分 (基于{time_control}时制下的各阶段加权准确率和走法质量):")
    print(f"白方 ({game.headers.get('White', 'Unknown')}): {white_estimated_elo}")
    print(f"黑方 ({game.headers.get('Black', 'Unknown')}): {black_estimated_elo}")

    # 添加样本不足的警告
    white_low_samples = [phase for phase, stats in white_phase_stats_final.items() if 0 < stats['moves'] < MIN_SAMPLE_SIZE]
    black_low_samples = [phase for phase, stats in black_phase_stats_final.items() if 0 < stats['moves'] < MIN_SAMPLE_SIZE]
    
    if white_low_samples or black_low_samples:
        print("\n注意: 某些阶段的样本数据较少，评估结果可能不够准确。")
        if white_low_samples:
            phase_names = [{'opening': '开局', 'middlegame': '中局', 'endgame': '残局'}.get(p, p) for p in white_low_samples]
            print(f"  白方: {', '.join(phase_names)}阶段样本不足")
        if black_low_samples:
            phase_names = [{'opening': '开局', 'middlegame': '中局', 'endgame': '残局'}.get(p, p) for p in black_low_samples]
            print(f"  黑方: {', '.join(phase_names)}阶段样本不足")

    # 添加强制性走法和关键局面的统计
    print("\n走法类型统计:")
    print(f"白方:")
    white_total = white_move_quality['总步数']
    if white_total > 0:
        forced_percent = (white_move_quality['强制性走法'] / white_total) * 100
        critical_percent = (white_move_quality['关键局面'] / white_total) * 100
        effective_percent = ((white_total - white_move_quality['强制性走法']) / white_total) * 100
        print(f"  总计走法: {white_total}步")
        print(f"  强制性走法: {white_move_quality['强制性走法']}步 ({forced_percent:.1f}%)")
        print(f"  关键局面: {white_move_quality['关键局面']}步 ({critical_percent:.1f}%)")
        print(f"  有效决策: {white_total - white_move_quality['强制性走法']}步 ({effective_percent:.1f}%)")
    
    print(f"黑方:")
    black_total = black_move_quality['总步数']
    if black_total > 0:
        forced_percent = (black_move_quality['强制性走法'] / black_total) * 100
        critical_percent = (black_move_quality['关键局面'] / black_total) * 100
        effective_percent = ((black_total - black_move_quality['强制性走法']) / black_total) * 100
        print(f"  总计走法: {black_total}步")
        print(f"  强制性走法: {black_move_quality['强制性走法']}步 ({forced_percent:.1f}%)")
        print(f"  关键局面: {black_move_quality['关键局面']}步 ({critical_percent:.1f}%)")
        print(f"  有效决策: {black_total - black_move_quality['强制性走法']}步 ({effective_percent:.1f}%)")
    
    # 恢复招法评价统计
    print(f"\n招法评价统计:")
    for player_color_stats, stats_dict in move_quality_stats.items():
        player_name = "白方" if player_color_stats == chess.WHITE else "黑方"
        actual_player_name_from_header = game.headers.get("White" if player_color_stats == chess.WHITE else "Black", player_name)
        print(f"\n{player_name} ({actual_player_name_from_header}):")
        total_player_moves = stats_dict["总步数"]
        if total_player_moves == 0:
            print("  没有走棋记录")
            continue
        for level_name in accuracy_levels_for_stats:
            count = stats_dict[level_name]
            percentage = (count / total_player_moves) * 100 if total_player_moves > 0 else 0
            print(f"  {level_name:<6}: {count:3} ({percentage:5.1f}%)")
        
        special_stats = special_moves_stats[player_color_stats]
        for move_type, count in special_stats.items():
            if move_type == "Brilliant": label_print = "精彩着法"
            elif move_type == "Great": label_print = "唯一好棋"
            elif move_type == "Good effort": label_print = "积极尝试"
            elif move_type == "Precise": label_print = "精准无误"
            else: label_print = move_type
            percentage = (count / total_player_moves) * 100 if total_player_moves > 0 else 0
            print(f"  {label_print:<6}: {count:3} ({percentage:5.1f}%)")
    
    engine.quit()

# 添加检测强制性走法的函数
def is_forced_move(engine_scores_pov, multipv_results_raw, board=None, move=None):
    """
    判断当前局面是否为强制性走法
    engine_scores_pov: 从玩家视角的引擎评分列表
    multipv_results_raw: 原始的多变例分析结果
    board: 当前局面（可选）
    move: 玩家的走法（可选）
    """
    if len(engine_scores_pov) < 2:
        return True  # 只有一个合法着法时，一定是强制性走法
    
    # 最佳走法与次佳走法的分差
    score_diff = engine_scores_pov[0] - engine_scores_pov[1]
    
    # 基本判断：如果分差大于阈值，认为是强制性走法
    is_forced = score_diff > FORCED_MOVE_THRESHOLD
    
    # 如果提供了局面和走法，进行更细致的判断
    if board and move and is_forced:
        # 检查是否为换子
        is_exchange = is_exchange_move(board, move)
        
        # 如果是换子，降低强制性走法的判定标准
        if is_exchange:
            is_forced = score_diff > FORCED_MOVE_THRESHOLD * 1.5  # 对换子要求更高的分差
            if DEBUG_MODE:
                print(f"[DEBUG] 检测到换子，调整强制性走法判断: {is_forced}")
    
    return is_forced

def is_exchange_move(board, move):
    """
    判断走法是否为换子
    """
    # 复制局面并执行走法
    board_after = board.copy()
    board_after.push(move)
    
    # 计算物质变化
    material_before = calculate_material(board)
    material_after = calculate_material(board_after)
    material_change = abs(material_before - material_after)
    
    # 如果物质变化超过阈值，可能是换子
    return material_change >= EXCHANGE_DETECTION_THRESHOLD

# 添加检测关键局面的函数
def is_critical_position(engine_scores_pov):
    """
    判断当前局面是否为关键局面
    关键局面是指最佳走法与次佳走法有明显差距的局面
    """
    if len(engine_scores_pov) < 2:
        return False
    
    score_diff = engine_scores_pov[0] - engine_scores_pov[1]
    return CRITICAL_POSITION_THRESHOLD < score_diff <= FORCED_MOVE_THRESHOLD

if __name__ == "__main__":
    ENGINE_PATH = r"C:\Users\Administrator\Downloads\Compressed\stockfish-windows-x86-64-avx2_2\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    print("请输入PGN对局内容（输入完成后 Windows 上按 Ctrl+Z 然后回车, Linux/Mac 上按 Ctrl+D）:")
    pgn_text = ""
    while True:
        try:
            line = input()
            pgn_text += line + "\n"
        except EOFError:
            break
    
    if pgn_text.strip():
        analyze_game(pgn_text, ENGINE_PATH)
    else:
        print("没有输入PGN内容。")