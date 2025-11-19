import numpy as np
import time

# ==========================================
# 1. 모양 데이터베이스 (회전 불가 - 방향별 정의)
# ==========================================
shapes_db = {
    # --- 1칸 ---
    '1_dot': [[1]],

    # --- 2칸 ---
    '2_bar_h': [[1, 1]],       # 가로 ㅡ
    '2_bar_v': [[1], [1]],     # 세로 |

    # --- 3칸 ---
    '3_bar_h': [[1, 1, 1]],    # 가로 ㅡ
    '3_bar_v': [[1], [1], [1]],# 세로 |
    '3_L_ru': [[1, 0], [1, 1]],# ㄴ (우하향)
    '3_L_lu': [[0, 1], [1, 1]],# ┘ (좌상향 채움 - 실제 모양은 J 뒤집은 것)
    '3_L_rd': [[1, 1], [1, 0]],# ㄱ
    '3_L_ld': [[1, 1], [0, 1]],# ┌ (7 모양)
    
    # 사진상의 구체적인 3칸 L 모양 매핑
    '3_L_corner_bl': [[1, 0], [1, 1]], # ㄴ 모양
    '3_L_corner_tl': [[1, 1], [1, 0]], # ㄱ 모양
    '3_L_corner_tr': [[1, 1], [0, 1]], # 7 모양
    '3_L_corner_br': [[0, 1], [1, 1]], # ┘ 모양

    # --- 4칸 (에픽/슈퍼에픽) ---
    '4_square': [[1, 1], [1, 1]],     # ㅁ
    '4_bar_h': [[1, 1, 1, 1]],        # ㅡ
    '4_bar_v': [[1], [1], [1], [1]],  # |
    
    # T자
    '4_T_up':    [[0, 1, 0], [1, 1, 1]], # ㅗ
    '4_T_down':  [[1, 1, 1], [0, 1, 0]], # ㅜ
    '4_T_left':  [[0, 1], [1, 1], [0, 1]], # ㅓ
    '4_T_right': [[1, 0], [1, 1], [1, 0]], # ㅏ
    
    # L자 (테트리스)
    '4_L_normal': [[1, 0], [1, 0], [1, 1]], # ㄴ
    '4_L_flip':   [[0, 1], [0, 1], [1, 1]], # ┘ (J)
    '4_L_pair_1': [[1, 1, 1], [1, 0, 0]],   # L 누운것
    '4_L_pair_2': [[1, 0, 0], [1, 1, 1]],   # J 누운것
    '4_L_small_ang': [[1,1], [0,1], [0,1]]  # ㄱ자 길게
}

# ==========================================
# 2. 게임 맵 (잠김 칸 반영)
# ==========================================
current_map = [
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 0, 1]
]

# ==========================================
# 3. 사용자 인벤토리 (사진 분석 데이터 입력됨)
# ==========================================
# 전략: 광휘(Gwanghwi)를 메인 세트로 설정하여 보너스 획득
# 관통(Gwantong)의 슈퍼에픽은 깡점수가 높으므로 포함, 나머지는 서브
my_inventory = []

# [1] 관통 (Gwantong) - 빨간색(슈퍼에픽), 보라색(에픽)
# 사진 2번째: 슈퍼에픽 T(아래), T(왼쪽) / 에픽 T(아래), L(뒤집힘) 등
my_inventory.extend([
    {'shape': '4_T_down',  'grade': 'superepic', 'is_main_set': False}, # 🟥 슈퍼에픽 ㅜ
    {'shape': '4_T_left',  'grade': 'superepic', 'is_main_set': False}, # 🟥 슈퍼에픽 ㅓ
    {'shape': '4_T_down',  'grade': 'epic',      'is_main_set': False}, # 🟪 에픽 ㅜ
    {'shape': '4_L_flip',  'grade': 'epic',      'is_main_set': False}, # 🟪 에픽 ┘ (x2)
    {'shape': '4_L_flip',  'grade': 'epic',      'is_main_set': False}, 
    {'shape': '3_bar_h',   'grade': 'epic',      'is_main_set': False}, # 🟪 나머지 에픽들...
    {'shape': '3_L_corner_bl', 'grade': 'epic',  'is_main_set': False},
    {'shape': '3_bar_h',   'grade': 'epic',      'is_main_set': False},
    {'shape': '3_L_corner_tr', 'grade': 'epic',  'is_main_set': False}, 
    {'shape': '3_L_corner_bl', 'grade': 'epic',  'is_main_set': False},
    {'shape': '2_bar_v',   'grade': 'epic',      'is_main_set': False},
    {'shape': '2_bar_v',   'grade': 'epic',      'is_main_set': False},
])

# [2] 광휘 (Gwanghwi) - 보라색(에픽), 파란색(레어)
# 사진 1번째: 2x2 사각형이 많음. 메인 세트(True)
my_inventory.extend([
    # --- 에픽 (보라 배경) ---
    {'shape': '4_square',  'grade': 'epic', 'is_main_set': True}, # 🟪 네모 (x3)
    {'shape': '4_square',  'grade': 'epic', 'is_main_set': True},
    {'shape': '4_square',  'grade': 'epic', 'is_main_set': True},
    {'shape': '1_dot',     'grade': 'epic', 'is_main_set': True}, # 🟪 1칸 (에픽배경)
    {'shape': '1_dot',     'grade': 'epic', 'is_main_set': True},
    {'shape': '2_bar_v',   'grade': 'epic', 'is_main_set': True}, # 🟪 2칸 세로
    {'shape': '2_bar_v',   'grade': 'epic', 'is_main_set': True},
    {'shape': '3_bar_h',   'grade': 'epic', 'is_main_set': True}, # 🟪 3칸 가로
    {'shape': '3_bar_v',   'grade': 'epic', 'is_main_set': True}, # 🟪 3칸 세로
    {'shape': '3_L_corner_bl', 'grade': 'epic', 'is_main_set': True}, # 🟪 ㄴ자

    # --- 레어 (파란 배경) ---
    {'shape': '2_bar_h',   'grade': 'rare', 'is_main_set': True}, # 🟦
    {'shape': '2_bar_h',   'grade': 'rare', 'is_main_set': True},
    {'shape': '1_dot',     'grade': 'rare', 'is_main_set': True}, # 🟦 점 (x4)
    {'shape': '1_dot',     'grade': 'rare', 'is_main_set': True},
    {'shape': '1_dot',     'grade': 'rare', 'is_main_set': True},
    {'shape': '1_dot',     'grade': 'rare', 'is_main_set': True},
    {'shape': '3_bar_v',   'grade': 'rare', 'is_main_set': True}, # 🟦 3칸 세로
    {'shape': '3_bar_v',   'grade': 'rare', 'is_main_set': True},
    {'shape': '3_L_corner_br', 'grade': 'rare', 'is_main_set': True}, # 🟦 ┘ 모양 (x4)
    {'shape': '3_L_corner_br', 'grade': 'rare', 'is_main_set': True},
    {'shape': '3_L_corner_br', 'grade': 'rare', 'is_main_set': True},
    {'shape': '3_L_corner_br', 'grade': 'rare', 'is_main_set': True},
    {'shape': '3_bar_h',   'grade': 'rare', 'is_main_set': True}, # 🟦 3칸 가로 (x2)
    {'shape': '3_bar_h',   'grade': 'rare', 'is_main_set': True},
])

# ==========================================
# 4. 시뮬레이터 엔진
# ==========================================
class SugarGlassSolver:
    def __init__(self, grid_map, inventory):
        self.rows = 7
        self.cols = 7
        self.grid_map = np.array(grid_map)
        self.inventory = inventory
        # 점수: 슈퍼에픽(120), 에픽(60), 레어(30)
        self.score_table = {'superepic': 120, 'epic': 60, 'rare': 30}
        
        # 최적화: 점수 높은 순으로 정렬하되, 1칸짜리 등 작은 건 나중에 채우기 위해 뒤로
        # (단, 칸당 점수 밀도가 높은 슈퍼에픽은 무조건 앞)
        self.inventory.sort(key=lambda x: (self.score_table.get(x['grade'], 0), len(shapes_db.get(x['shape'], [[0]])[0])), reverse=True)
        
        self.best_score = -1
        self.best_grid = None

    def get_set_bonus(self, count):
        if count < 9: return 0
        capped_count = min(count, 21)
        steps = (capped_count - 9) // 3
        return 265 + (steps * 265)

    def solve(self):
        print(f"🧮 최적 배치 계산 중... (보유 조각 {len(self.inventory)}개)")
        self._backtrack(0, self.grid_map, 0, 0)
        return self.best_score, self.best_grid

    def _backtrack(self, idx, current_grid, current_base_score, main_type_count):
        # 현재 상태 점수 (기본 점수 + 세트 점수)
        total_score = current_base_score + self.get_set_bonus(main_type_count)
        
        if total_score > self.best_score:
            self.best_score = total_score
            self.best_grid = current_grid.copy()

        if idx >= len(self.inventory):
            return

        item = self.inventory[idx]
        shape_key = item['shape']
        
        if shape_key not in shapes_db: # 안전장치
            self._backtrack(idx + 1, current_grid, current_base_score, main_type_count)
            return

        shape = np.array(shapes_db[shape_key])
        grade = item['grade']
        is_main = item['is_main_set']
        
        piece_pts = self.score_table[grade] * np.sum(shape)
        piece_cells = np.sum(shape)
        
        h, w = shape.shape
        placed = False
        
        # 배치 시도
        for r in range(self.rows - h + 1):
            for c in range(self.cols - w + 1):
                # 공간 확인
                if np.all((current_grid[r:r+h, c:c+w] + shape) <= 1):
                    new_grid = current_grid.copy()
                    # 시각화 값: 슈에(8), 에픽(7), 레어(6)
                    vis_val = 8 if grade == 'superepic' else (7 if grade == 'epic' else 6)
                    
                    for i in range(h):
                        for j in range(w):
                            if shape[i][j] == 1:
                                new_grid[r+i][c+j] = vis_val
                                
                    self._backtrack(idx + 1, new_grid, current_base_score + piece_pts, 
                                    main_type_count + (piece_cells if is_main else 0))
                    placed = True
                    
                    # 가지치기: 큰 조각을 하나 배치했으면 같은 레벨의 다른 위치 탐색은 
                    # 경우의 수가 너무 많으면 줄여야 하지만, 여기선 정확도를 위해 진행
                    # (속도가 너무 느리면 break 추가 가능)
        
        # 배치하지 않고 넘어가는 경우
        # (작은 조각은 건너뛰어도 되지만, 큰 조각은 무조건 넣는게 이득이므로 로직 분리 가능)
        if not placed:
             self._backtrack(idx + 1, current_grid, current_base_score, main_type_count)

# ==========================================
# 5. 결과 출력
# ==========================================
solver = SugarGlassSolver(current_map, my_inventory)
score, final_grid = solver.solve()

print(f"\n🏆 최대 점수: {score}점")
print("\n--- [배치 결과] ---")
print("🟥:슈퍼에픽(관통)  🟪:에픽  🟦:레어  ⬛:잠김")
display_map = {0: '⬜', 1: '⬛', 8: '🟥', 7: '🟪', 6: '🟦'}

if final_grid is not None:
    for row in final_grid:
        line = ""
        for cell in row:
            line += display_map.get(cell, '⬜')
        print(line)
else:
    print("배치 실패")