"""
Autonomy challenge agent.


Explaning my appraoch
-----------------
My approach is that it firsts  1. Then the mapping section turns area into a 51x51 scan into a grid that covers the whole field.  Every cell starts of as unknown, and becomes free or
blocked once the robot has actually seen it. Then it searches grid for a route from the robot to the goal, using the A* algorithm. Then it follows the path by aiming straight at
 the next waypoint. It also slows down before sharp turns  and before the goal so the acceleration limit is never exceeded. What is missing is the path is a 
 smoothed grid-based shortest path rather than a perfectly optimal continuous-space one and there's no separate exploration search since driving to the best frontier already
covers it.

"""

import heapq
import math

UNKNOWN, FREE, BLOCKED = 0, 1, 2

# This is umber of matching readings needed before changing a cell's state.
# It uses the same value in both directions helps avoid reacting to noisy scans
CONFIRM_BLOCKED_STREAK = 4
CONFIRM_FREE_STREAK = 4

# This for how often in ticks to re-plan even when the current path still looks fine
# This is so a shortcut can be revealed to the side of the path is noticed
PERIODIC_REPLAN_TICKS = 20

# How often in ticks to retry planning while no path exists at all.
# I made it less frequent than a normal replan because a full search
# that fails still has to visit every reachable cell before giving up
STUCK_RETRY_TICKS = 15


class Agent:
  def __init__(self, cfg: dict):
    self.res = cfg["resolution"]
    self.radius = cfg["robot_radius"]
    self.v_max = cfg["v_max"]
    self.a_max = cfg["a_max"]
    self.dt = cfg["dt"]
    self.goal = cfg["goal"]
    self.goal_tol = cfg["goal_tol"]

    self.width = round(cfg["width_m"] / self.res)
    self.height = round(cfg["height_m"] / self.res)
    n = self.width * self.height

    self.grid = bytearray(n)          # Handles UNKNOWN / FREE / BLOCKED per cell
    self.pending = bytearray(n)       # For consecutive readings that disagree with grid
    self.inflated_count = [0] * n     # For how many nearby BLOCKED cells each cell has


    self._disc = self._make_disc(self.radius)

  # list of (x, y) waypoints in world coordinates
    self.path = []          
    self.path_index = 0     
    self.path_reaches_goal = False  
    self.ticks = 0
    self.next_plan_tick = 0   #
    self.est_vx, self.est_vy = 0.0, 0.0  #



  def _index(self, cx: int, cy: int) -> int:
    return cy * self.width + cx

  def _inside(self, cx: int, cy: int) -> bool:
    return 0 <= cx < self.width and 0 <= cy < self.height

  def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
    return (math.floor(x / self.res), math.floor(y / self.res))

  def _cell_to_world(self, cx: int, cy: int) -> tuple[float, float]:
    return ((cx + 0.5) * self.res, (cy + 0.5) * self.res)

  def _make_disc(self, radius_m: float) -> list[tuple[int, int]]:
    """All (dx, dy) cell offsets whose free cell could contain a point
    closer than an obstacle cell at offset (0, 0).

    Cells are not points so the closest approach between a free cell and an obstacle cell is the distance
    between their nearest edges/corners. So i tried to make this formula
    below which is the exact minimum distance between two axis-aligned squares
    of one cell each, offset by (dx, dy) cells."""
    r = math.ceil(radius_m / self.res) + 1   # a safe bounding box to search
    offsets = []
    for dy in range(-r, r + 1):
      for dx in range(-r, r + 1):
        gap_x = max(abs(dx) - 1, 0)
        gap_y = max(abs(dy) - 1, 0)
        if self.res * math.hypot(gap_x, gap_y) < radius_m:
          offsets.append((dx, dy))
    return offsets

  def _adjust_inflation(self, cx: int, cy: int, delta: int) -> None:

    for dx, dy in self._disc:
      nx, ny = cx + dx, cy + dy
      if self._inside(nx, ny):
        self.inflated_count[self._index(nx, ny)] += delta

  def _blocked_for_planning(self, cx: int, cy: int) -> bool:

    if not self._inside(cx, cy):
      return True
    idx = self._index(cx, cy)
    if self.grid[idx] != FREE:
      return True
    return self.inflated_count[idx] > 0

  def _fuse_scan(self, scan: tuple[int, int, list[str]]) -> None:
    # This will Merge one local scan into the permanent map.
    cx0, cy0, rows = scan

    for j, row in enumerate(rows):
      cy = cy0 + j
      if not self._inside(0, cy):
        continue

      for i, ch in enumerate(row):
        cx = cx0 + i
        if not self._inside(cx, cy):
          continue

        idx = self._index(cx, cy)
        state = self.grid[idx]

        if ch == "#":
          if state == BLOCKED:
            self.pending[idx] = 0 
          elif state == UNKNOWN:
            # This is first look at this cell: there is no earlier belief to
            # protect, so trust it immediately
            self.grid[idx] = BLOCKED
            self._adjust_inflation(cx, cy, +1)
          else:
            # This cell was previously confirmed to be FREE. Any single misaligned
            # scan can make a real wall look shifted by a cell (see module
            # docstring), so require a couple of readings before
            # we can overide it. Any pop-up obstacle will also get caught in a couple of ticks
            self.pending[idx] += 1
            if self.pending[idx] >= CONFIRM_BLOCKED_STREAK:
              self.grid[idx] = BLOCKED
              self._adjust_inflation(cx, cy, +1)
              self.pending[idx] = 0
        else:
          if state == BLOCKED:
            # Require several agreeing readings before trusting that a
            # previously-seen wall is actually gone.
            self.pending[idx] += 1
            if self.pending[idx] >= CONFIRM_FREE_STREAK:
              self.grid[idx] = FREE
              self._adjust_inflation(cx, cy, -1)
              self.pending[idx] = 0
          else:
            self.pending[idx] = 0
            if state == UNKNOWN:
              self.grid[idx] = FREE

  # ------------------------------------------------------------------
  # Robot Planning
  # ------------------------------------------------------------------

  def _nearest_reachable(self, cell: tuple[int, int]) -> tuple[int, int] | None:
    """Find the nearest cell that is safe for the planner to occupy.

    This is primarily used to recover from any estimation error. If the
    reported robot position falls inside an inflated obstacle, it moves the
    planning start position to the closest non-blocked cell instead.
    """
    if not self._blocked_for_planning(*cell):
      return cell

    cx0, cy0 = cell
    max_r = max(self.width, self.height)

    for r in range(1, max_r):
      # Search the perimeter of the current square ring.
      for dx in range(-r, r + 1):
        for cx, cy in ((cx0 + dx, cy0 - r), (cx0 + dx, cy0 + r), (cx0 - r, cy0 + dx), (cx0 + r, cy0 + dx)):
          if self._inside(cx, cy) and not self._blocked_for_planning(cx, cy):
            return (cx, cy)

    return None

  def _is_frontier(self, cell: tuple[int, int]) -> bool:
    """Return whether `cell` borders at least one completely unexplored cell.
    """
    cx, cy = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      nx, ny = cx + dx, cy + dy
      if self._inside(nx, ny) and self.grid[self._index(nx, ny)] == UNKNOWN:
        return True
    return False

  def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
    """Estimate the cheapest grid distance between two cells.

    Diagonal moves cost sqrt(2), while horizontal and vertical moves cost 1,
  
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    # Distance for 8-connected movement.
    return (dx + dy) - (2 - math.sqrt(2)) * min(dx, dy)

  def _neighbours(self, cell: tuple[int, int]):
    """Yield collision-free neighbouring cells and their movement costs.
    """
    cx, cy = cell

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      nx, ny = cx + dx, cy + dy
      if not self._blocked_for_planning(nx, ny):
        yield (nx, ny), 1.0

    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
      nx, ny = cx + dx, cy + dy
      # Prevent diagonal corner-cutting through two touching obstacles.
      if self._blocked_for_planning(nx, ny):
        continue
      if self._blocked_for_planning(cx + dx, cy) or self._blocked_for_planning(cx, cy + dy):
        continue
      yield (nx, ny), math.sqrt(2)

  def _reconstruct(self, came_from: dict, node: tuple[int, int]) -> list[tuple[int, int]]:
    """Reconstruct a path by following parent links back to the start."""
    path = [node]
    while node in came_from:
      node = came_from[node]
      path.append(node)
    path.reverse()
    return path

  def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Find a safe path to the goal, or a useful exploration target.

    The goal may not be reachable yet because the robot only knows the portion
    of the map it has already observed.
    The final fallback keeps the robot near the obstacle or unexplored boundary
    blocking the goal. This also helps in the future scans may reveal an
    opening that was previously unknown.
    """
    start = self._nearest_reachable(start)
    if start is None:
      return None

    heap = [(self._heuristic(start, goal), 0.0, start)]
    came_from = {}
    best_cost = {start: 0.0}
    best_frontier = None
    closest_to_goal, closest_dist = start, self._heuristic(start, goal)

    while heap:
      _, cost, current = heapq.heappop(heap)

      if current == goal:
        return self._reconstruct(came_from, current)

      # Ignore heap entries whose node has since received a cheaper path.
      if cost > best_cost.get(current, math.inf):
        continue

      # Because A* expands in increasing  order, the first path
      # encountered is the most promising path according to this search.
      if best_frontier is None and current != start and self._is_frontier(current):
        best_frontier = current

      # Track the reachable cell geometrically closest to the goal as a
      # fallback when no frontier remains.
      dist_to_goal = self._heuristic(current, goal)
      if dist_to_goal < closest_dist:
        closest_to_goal, closest_dist = current, dist_to_goal

      for nxt, step_cost in self._neighbours(current):
        new_cost = cost + step_cost
        if new_cost < best_cost.get(nxt, math.inf):
          best_cost[nxt] = new_cost
          came_from[nxt] = current
          heapq.heappush(heap, (new_cost + self._heuristic(nxt, goal), new_cost, nxt))

    if best_frontier is not None:
      return self._reconstruct(came_from, best_frontier)

    if closest_to_goal != start:
      return self._reconstruct(came_from, closest_to_goal)

    return None  # Already at the closest reachable point.

  def _line_clear(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether a straight segment between two cells is collision-free.
    """
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
      if self._blocked_for_planning(x, y):
        return False
      if (x, y) == (x1, y1):
        return True

      prev_x, prev_y = x, y
      e2 = 2 * err
      if e2 > -dy:
        err -= dy
        x += sx
      if e2 < dx:
        err += dx
        y += sy

      # A diagonal step function to clear both side cells
      if x != prev_x and y != prev_y:
        if self._blocked_for_planning(prev_x, y) or self._blocked_for_planning(x, prev_y):
          return False

  def _smooth_path(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Removes unnecessary intermediate waypoints from an A* path.
    """
    if len(cells) <= 2:
      return cells

    result = [cells[0]]
    i = 0
    while i < len(cells) - 1:
      best = i + 1

      # Prefer the farthest visible waypoint so the resulting path has as few
      # turns as possible.
      for j in range(len(cells) - 1, i, -1):
        if self._line_clear(cells[i], cells[j]):
          best = j
          break

      result.append(cells[best])
      i = best

    return result

  def _plan(self, pose: tuple[float, float]) -> None:
    """This will build a new world-coordinate path from the current robot position."""
    start = self._world_to_cell(*pose)
    goal_cell = self._world_to_cell(*self.goal)

    cells = self._astar(start, goal_cell)
    if cells is None:
      self.path = []
      self.path_index = 0
      self.path_reaches_goal = False
      return

    # This simplifies the grid path before converting it back to world coordinates.
    cells = self._smooth_path(cells)
    self.path = [self._cell_to_world(cx, cy) for cx, cy in cells]
    self.path_index = 0

    # A path can end at a frontier rather than the actual goal while the map
    # is still being explored.
    last = self.path[-1]
    self.path_reaches_goal = math.hypot(last[0] - self.goal[0], last[1] - self.goal[1]) <= self.goal_tol

  def _reached_path_end(self, pose: tuple[float, float]) -> bool:
    """Return whether the robot is close enough to the final waypoint."""
    if not self.path:
      return False
    last = self.path[-1]
    return math.hypot(last[0] - pose[0], last[1] - pose[1]) < 0.2

  # ------------------------------------------------------------------
  # following the path
  # ------------------------------------------------------------------

  def _path_blocked_ahead(self, pose: tuple[float, float]) -> bool:
    """Check whether newly discovered obstacles have blocked the path ahead.
    """
    look_ahead_m = 1.0
    travelled = 0.0
    x, y = pose

    for i in range(self.path_index, len(self.path)):
      wx, wy = self.path[i]
      travelled += math.hypot(wx - x, wy - y)
      x, y = wx, wy

      if self._blocked_for_planning(*self._world_to_cell(wx, wy)):
        return True

      if travelled > look_ahead_m:
        break

    return False

  def _advance_path_index(self, pose: tuple[float, float]) -> None:
    """
    This will help to prevent the controller from steering back toward a waypoint that is
    effectively behind the robot.
    """
    while self.path_index < len(self.path) - 1:
      wx, wy = self.path[self.path_index]
      if math.hypot(wx - pose[0], wy - pose[1]) > 0.15:
        break
      self.path_index += 1

  # Distance along the path used to inspect upcoming corners when braking.
  # A short look-ahead catches chains of nearby turns without adding much
  # computation to each control tick.
  _TURN_LOOKAHEAD_M = 1.0

  def _turns_ahead(self, pose: tuple[float, float]):
    """ This measures the  upcoming corners as distance and sharpness pairs.

    """
    i = self.path_index
    if i >= len(self.path) - 1:
      return

    prev_point = pose
    dist = 0.0

    while i < len(self.path) - 1 and dist <= self._TURN_LOOKAHEAD_M:
      bx, by = self.path[i]
      cx, cy = self.path[i + 1]

      ux, uy = bx - prev_point[0], by - prev_point[1]
      seg_len = math.hypot(ux, uy)
      dist += seg_len

      vx, vy = cx - bx, cy - by
      vn = math.hypot(vx, vy)

      if seg_len > 1e-6 and vn > 1e-6:
        cos_angle = max(-1.0, min(1.0, (ux * vx + uy * vy) / (seg_len * vn)))
        yield dist, (1.0 - cos_angle) / 2.0

      prev_point = (bx, by)
      i += 1

  def _desired_velocity(self, pose: tuple[float, float]) -> tuple[float, float]:
    """This computes a velocity command that follows the path while braking safely.
    """
    if not self.path:
      return (0.0, 0.0)

    self._advance_path_index(pose)

    # Aim directly at the current waypoint which avoids cutting across corners
    target = self.path[self.path_index]

    dx, dy = target[0] - pose[0], target[1] - pose[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
      return (0.0, 0.0)

    #This limit sideways velocity at upcoming turns so the robot has enough
    # distance to change direction without drifting outside the planned path.
    LATERAL_BUDGET = 0.35  # Maximum tolerated sideways speed at a corner.

    def speed_for_sharpness(sharpness: float, dist: float) -> float:
      """Return the maximum safe speed before a turn at `dist`."""
      sin_angle = 2.0 * math.sqrt(max(sharpness * (1.0 - sharpness), 0.0))

      # Near a full reversal, sin angle becomes small even though the turn
      # is severe, so sharpness itself provides a lower bound on the braking
      # requirement
      shed_fraction = max(sin_angle, sharpness, 1e-3)
      corner_speed = min(self.v_max, LATERAL_BUDGET / shed_fraction)

    
      return math.sqrt(max(corner_speed * corner_speed + 2.0 * self.a_max * dist, 0.0))

    turn_limit = math.inf
    for corner_dist, sharpness in self._turns_ahead(pose):
      turn_limit = min(turn_limit, speed_for_sharpness(sharpness, corner_dist))

  
    est_speed = math.hypot(self.est_vx, self.est_vy)
    if est_speed > 0.05:
      cos_angle = max(-1.0, min(1.0, (self.est_vx * dx + self.est_vy * dy) / (est_speed * distance)))
      mismatch = (1.0 - cos_angle) / 2.0
      turn_limit = min(turn_limit, speed_for_sharpness(mismatch, 0.0))

  
    end_x, end_y = self.path[-1]
    end_dist = math.hypot(end_x - pose[0], end_y - pose[1])
    stop_margin = self.goal_tol * 0.5 if self.path_reaches_goal else 0.0
    end_limit = math.sqrt(max(2.0 * self.a_max * max(end_dist - stop_margin, 0.0), 0.0))

    speed = min(self.v_max, turn_limit, end_limit)

    # Prevent a single command from carrying the robot past its current
    # target waypoint.
    speed = min(speed, distance / self.dt)

    return (dx / distance * speed, dy / distance * speed)

  # ------------------------------------------------------------------
  # main loop
  # ------------------------------------------------------------------

  @staticmethod
  def _clamp_norm(vx: float, vy: float, limit: float) -> tuple[float, float]:
    """This fixes a 2-D vector's magnitude without changing its direction.
    """
    n = math.hypot(vx, vy)
    return (vx, vy) if n <= limit else (vx * limit / n, vy * limit / n)

  def _advance_velocity_estimate(self, commanded: tuple[float, float]) -> None:
    """Predict the robot's actual velocity after applying a command.
    """
    target_vx, target_vy = self._clamp_norm(commanded[0], commanded[1], self.v_max)
    dvx, dvy = self._clamp_norm(target_vx - self.est_vx, target_vy - self.est_vy, self.a_max * self.dt)
    self.est_vx += dvx
    self.est_vy += dvy

  def step(self, pose: tuple[float, float], scan: tuple[int, int, list[str]]) -> tuple[float, float]:
    """Process one control tick and return the next velocity command."""
    self.ticks += 1
    self._fuse_scan(scan)

    need_plan = False

    if not self.path:
     
      need_plan = self.ticks >= self.next_plan_tick

    elif self._path_blocked_ahead(pose):
      # New sensor information has invalidated part of the current route.
      need_plan = True

    elif self._reached_path_end(pose) and not self.path_reaches_goal:
      need_plan = True

    elif self.path_reaches_goal and self.ticks % PERIODIC_REPLAN_TICKS == 0:
      need_plan = True

    if need_plan:
      self._plan(pose)

      
      self.next_plan_tick = self.ticks + (
        STUCK_RETRY_TICKS if not self.path else PERIODIC_REPLAN_TICKS
      )

    velocity = self._desired_velocity(pose)

    # Keep the internal velocity estimate  with the command that
    # was just issued.
    self._advance_velocity_estimate(velocity)

    return velocity

  def debug(self) -> dict:
    """Return map and path information for the terminal visualizer."""
    blocked, free = [], []

    for cy in range(self.height):
      for cx in range(self.width):
        state = self.grid[self._index(cx, cy)]

        if state == BLOCKED:
          blocked.append((cx, cy))
        elif state == FREE:
          free.append((cx, cy))

    return {"blocked": blocked, "free": free, "path": self.path}
