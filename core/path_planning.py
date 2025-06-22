"""
A* Path Planning Module for Autonomous Vehicle
Implements A* algorithm for obstacle avoidance and navigation
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum
import math

from config.settings import ModelConfig, ControlConfig
from core.logger import logger


class Direction(Enum):
    """Direction enum for path planning"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    NORTHEAST = 4
    SOUTHEAST = 5
    SOUTHWEST = 6
    NORTHWEST = 7


@dataclass
class Node:
    """Node for A* path planning"""
    x: int
    y: int
    g_cost: float = float('inf')  # Cost from start to current node
    h_cost: float = 0  # Heuristic cost from current to goal
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        """Total cost (g + h)"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        """Equality comparison"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """Hash for set operations"""
        return hash((self.x, self.y))


@dataclass
class PathPoint:
    """Point in the planned path"""
    x: float
    y: float
    direction: float  # angle in radians
    speed: float
    timestamp: float


class PathPlanner:
    """A* path planner for autonomous vehicle navigation"""
    
    def __init__(self, grid_size: Optional[int] = None, max_grid_size: Optional[int] = None):
        self.grid_size = grid_size or ModelConfig.GRID_SIZE
        self.max_grid_size = max_grid_size or ModelConfig.MAX_GRID_SIZE
        
        # Grid and obstacles
        self.grid: Optional[np.ndarray] = None
        self.grid_width = 0
        self.grid_height = 0
        
        # Path planning state
        self.current_path: List[PathPoint] = []
        self.current_goal: Optional[Tuple[float, float]] = None
        self.is_planning = False
        
        # Performance tracking
        self.planning_times: List[float] = []
        self.path_lengths: List[float] = []
        
        # Movement directions (8-connected grid)
        self.directions = [
            (0, -1),   # North
            (1, 0),    # East
            (0, 1),    # South
            (-1, 0),   # West
            (1, -1),   # Northeast
            (1, 1),    # Southeast
            (-1, 1),   # Southwest
            (-1, -1)   # Northwest
        ]
        
        # Direction costs (diagonal movement costs more)
        self.direction_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        
        logger.info("Path planner initialized")
    
    def initialize_grid(self, width: int, height: int):
        """Initialize the planning grid"""
        self.grid_width = width
        self.grid_height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        
        logger.info(f"Grid initialized: {width}x{height}")
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update obstacle map with detected objects
        
        Args:
            obstacles: List of (x, y, radius) tuples representing obstacles
        """
        if self.grid is None:
            logger.error("Grid not initialized")
            return
        
        # Clear previous obstacles
        self.grid.fill(0)
        
        # Add new obstacles
        for x, y, radius in obstacles:
            # Convert world coordinates to grid coordinates
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            grid_radius = max(1, int(radius / self.grid_size))
            
            # Mark obstacle area
            self._mark_obstacle_area(grid_x, grid_y, grid_radius)
        
        logger.debug(f"Updated obstacles: {len(obstacles)} objects")
    
    def _mark_obstacle_area(self, center_x: int, center_y: int, radius: int):
        """Mark obstacle area in the grid"""
        if self.grid is None:
            return
            
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = center_x + dx
                y = center_y + dy
                
                # Check bounds
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    # Check if point is within radius
                    if dx*dx + dy*dy <= radius*radius:
                        self.grid[y, x] = 1
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[PathPoint]:
        """Plan path from start to goal using A* algorithm"""
        if self.grid is None:
            logger.error("Grid not initialized")
            return []
        
        logger.start_timer("path_planning")
        
        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        # Validate coordinates
        if not self._is_valid_position(start_grid) or not self._is_valid_position(goal_grid):
            logger.error("Invalid start or goal position")
            return []
        
        # Check if goal is reachable
        if self.grid[goal_grid[1], goal_grid[0]] == 1:
            logger.warning("Goal position is blocked by obstacle")
            return []
        
        # Run A* algorithm
        path_grid = self._a_star_search(start_grid, goal_grid)
        
        if not path_grid:
            logger.warning("No path found to goal")
            return []
        
        # Convert grid path to world coordinates
        path_world = self._grid_to_world_path(path_grid)
        
        # Smooth path and add speed/direction information
        smooth_path = self._smooth_path(path_world)
        
        planning_time = logger.end_timer("path_planning")
        self.planning_times.append(planning_time)
        
        # Update performance metrics
        path_length = self._calculate_path_length(smooth_path)
        self.path_lengths.append(path_length)
        
        logger.info(f"Path planned: {len(smooth_path)} points, "
                   f"length: {path_length:.2f}m, time: {planning_time:.3f}s")
        
        return smooth_path
    
    def _a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* search algorithm implementation"""
        if self.grid is None:
            return []
            
        # Initialize start node
        start_node = Node(start[0], start[1], g_cost=0)
        start_node.h_cost = self._heuristic(start, goal)
        
        # Priority queue for open nodes
        open_set = [start_node]
        heapq.heapify(open_set)
        
        # Set for closed nodes
        closed_set = set()
        
        # Dictionary to track best g_costs
        g_costs: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)
            
            # Add to closed set
            closed_set.add((current.x, current.y))
            
            # Check all neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor_x = current.x + dx
                neighbor_y = current.y + dy
                
                # Check bounds and obstacles
                if not self._is_valid_position((neighbor_x, neighbor_y)):
                    continue
                
                if self.grid[neighbor_y, neighbor_x] == 1:
                    continue
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                # Calculate new g_cost
                new_g_cost = current.g_cost + self.direction_costs[i]
                
                # Check if this path is better
                if (neighbor_x, neighbor_y) not in g_costs or new_g_cost < g_costs[(neighbor_x, neighbor_y)]:
                    g_costs[(neighbor_x, neighbor_y)] = new_g_cost
                    
                    neighbor = Node(neighbor_x, neighbor_y, g_cost=new_g_cost)
                    neighbor.h_cost = self._heuristic((neighbor_x, neighbor_y), goal)
                    neighbor.parent = current
                    
                    heapq.heappush(open_set, neighbor)
        
        return []  # No path found
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate heuristic cost (Euclidean distance)"""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return math.sqrt(dx*dx + dy*dy)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return list(reversed(path))
    
    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y = world_pos
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        return grid_x, grid_y
    
    def _grid_to_world_path(self, grid_path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert grid path to world coordinates"""
        world_path = []
        for grid_x, grid_y in grid_path:
            world_x = grid_x * self.grid_size + self.grid_size / 2
            world_y = grid_y * self.grid_size + self.grid_size / 2
            world_path.append((world_x, world_y))
        return world_path
    
    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[PathPoint]:
        """Smooth path and add speed/direction information"""
        if len(path) < 2:
            return []
        
        smooth_path = []
        current_time = time.time()
        
        for i in range(len(path)):
            x, y = path[i]
            
            # Calculate direction
            if i < len(path) - 1:
                next_x, next_y = path[i + 1]
                direction = math.atan2(next_y - y, next_x - x)
            else:
                # Use previous direction for last point
                direction = smooth_path[-1].direction if smooth_path else 0
            
            # Calculate speed based on curvature
            speed = self._calculate_speed(path, i)
            
            path_point = PathPoint(
                x=x,
                y=y,
                direction=direction,
                speed=speed,
                timestamp=current_time + i * 0.1  # 100ms intervals
            )
            
            smooth_path.append(path_point)
        
        return smooth_path
    
    def _calculate_speed(self, path: List[Tuple[float, float]], index: int) -> float:
        """Calculate speed based on path curvature"""
        if len(path) < 3:
            return ControlConfig.MAX_SPEED
        
        # Calculate curvature using three points
        if index > 0 and index < len(path) - 1:
            prev = path[index - 1]
            curr = path[index]
            next_point = path[index + 1]
            
            # Calculate angle between segments
            angle1 = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            angle2 = math.atan2(next_point[1] - curr[1], next_point[0] - curr[0])
            
            angle_diff = abs(angle2 - angle1)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            
            # Reduce speed for sharp turns
            if angle_diff > math.pi/4:  # 45 degrees
                return ControlConfig.TURNING_SPEED
            elif angle_diff > math.pi/8:  # 22.5 degrees
                return ControlConfig.MAX_SPEED * 0.7
        
        return ControlConfig.MAX_SPEED
    
    def _calculate_path_length(self, path: List[PathPoint]) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            distance = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            total_length += distance
        
        return total_length
    
    def get_next_waypoint(self, current_pos: Tuple[float, float], look_ahead: float = 50.0) -> Optional[PathPoint]:
        """Get next waypoint based on current position and look-ahead distance"""
        if not self.current_path:
            return None
        
        # Find closest point on path
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.current_path):
            distance = math.sqrt((point.x - current_pos[0])**2 + (point.y - current_pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Find waypoint at look-ahead distance
        target_distance = 0
        for i in range(closest_index, len(self.current_path)):
            if i > closest_index:
                prev_point = self.current_path[i - 1]
                curr_point = self.current_path[i]
                segment_length = math.sqrt((curr_point.x - prev_point.x)**2 + (curr_point.y - prev_point.y)**2)
                target_distance += segment_length
            
            if target_distance >= look_ahead:
                return self.current_path[i]
        
        # Return last point if look-ahead distance not reached
        return self.current_path[-1] if self.current_path else None
    
    def update_path(self, new_path: List[PathPoint]):
        """Update current path"""
        self.current_path = new_path
        logger.debug(f"Path updated: {len(new_path)} waypoints")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get path planning performance metrics"""
        if not self.planning_times:
            return {}
        
        return {
            "avg_planning_time": float(np.mean(self.planning_times)),
            "max_planning_time": float(np.max(self.planning_times)),
            "avg_path_length": float(np.mean(self.path_lengths)) if self.path_lengths else 0.0,
            "total_paths_planned": len(self.planning_times)
        }
    
    def clear_performance_data(self):
        """Clear performance tracking data"""
        self.planning_times.clear()
        self.path_lengths.clear()
        logger.info("Path planning performance data cleared") 