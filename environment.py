import gymnasium as gym
import highway_env
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

class Intent(IDMVehicle):
    def act(self, action=None):
        if getattr(self, 'elapsed', 0.0) >= getattr(self, 'brake_at', float('inf')):
            self.target_speed = self.brake_speed
        super().act(action)
    
    def step(self, dt):
        self.elapsed = getattr(self, 'elapsed', 0.0) + dt
        super().step(dt)
    
    @property
    def intent(self):
        front, rear = self.road.neighbour_vehicles(self, self.lane_index)
        acceleration = self.acceleration(self, front, rear)
        return 'brake' if acceleration < -0.5 else 'accelerate' if acceleration > 0.5 else 'maintain'

class Truck(Intent):
    LENGTH = 12.0
    WIDTH = 2.5

class DrivingEnv:
    def __init__(self, bits_price=0.01, seed=11, scenario=None, render_mode=None):
            if scenario not in (None, 'lead_braking', 'truck_occlusion', 'mixed', "wide_gap", "closing_gap",):
                raise ValueError('Unknown scenario')
            self.scenario_choice = scenario
            self.bits_price = bits_price
            self.seed = seed
            self.env = gym.make('highway-v0', render_mode=render_mode, config={'lanes_count': 3,
                'vehicles_count': 0, 'duration': 20, 'initial_lane_id': 1, 'policy_frequency': 2,
                'observation': {'type': 'LidarObservation', 'cells': 16,
                                'maximum_range': 25, 'normalize': True}})
            self.requests = self.bits = 0
            self.pending = False
    
    def reset(self, seed=None):
            obs, info = self.env.reset(seed=self.seed if seed is None else seed)
            self.requests = self.bits = 0
            self.pending = False
            self.steps = 0
            self.target_lane = 2
            rng = np.random.default_rng(self.seed if seed is None else seed)
            base = self.env.unwrapped
            ego = base.vehicle
            ego.position = np.array([100.0, 4.0]); ego.speed = 25.0
            ego.target_speed = 25.0; ego.heading = 0.0
            base.road.vehicles = [ego]
            self.scenario = self.scenario_choice or str(rng.choice(['lead_braking', 'truck_occlusion', 'mixed']))
            truck_scenario = self.scenario == "truck_occlusion"
            scenarios = [
                (1, 13.0 if truck_scenario else rng.uniform(45, 60),
                rng.uniform(23, 26)),
                (2, 32.0 if truck_scenario else rng.uniform(22, 28),
                rng.uniform(23, 27)),
                (2, rng.uniform(-25, -18), rng.uniform(24, 28)),
                (2, rng.uniform(55, 65), rng.uniform(24, 27)),
            ]

            if truck_scenario:
                scenarios.append((1, 22.0, rng.uniform(22, 25)))
            for lane_index in range(3):
                for dx in (-95.0, -60.0, 95.0, 140.0):
                    if rng.random() < 0.45:
                        scenarios.append((lane_index, dx + rng.uniform(-4, 4), rng.uniform(20, 30)))
            gap_demo = self.scenario in ("wide_gap", "closing_gap")
            if gap_demo:
                rear_dx = -45.0 if self.scenario == "wide_gap" else -10.0
                scenarios = [
                    (2, 150.0, 32.0),
                    (2, rear_dx, 32.0),
                ]
            for idx, (lane_index, dx, speed) in enumerate(scenarios):
                cls = Truck if idx == 0 and self.scenario == 'truck_occlusion' else Intent
                v = cls(base.road, np.array([100.0 + dx, 4.0 * lane_index]),
                               speed=float(speed), enable_lane_change=False)
                if gap_demo:
                    v.target_speed = v.speed
                else:
                    desired_change = float(rng.choice([0.0, 5.0, -5.0]))
                    v.target_speed = max(5.0, v.speed + desired_change)
                if idx == 3 and self.scenario == 'lead_braking':
                    v.target_speed = v.speed
                    v.brake_at = float(rng.uniform(1.0, 2.0))
                    v.brake_speed = 8.0
                base.road.vehicles.append(v)
            self.episode_done = False
            info.update(scenario=self.scenario, vehicle_count=len(base.road.vehicles))
            return self.observation(self.lidar(ego, 25.0)), info

    def lidar(self, observer, maximum_range):
        """
        - split 2D scan into 16 sectors and cast 16 rays per sector
        - check each ray against vehicle rectangles using their actual orientation
        - keep only first hit on each ray, closer vehicles block farther ones.
        - closest hit in each sector report normalized distance and relative speed along the ray
        - run the scan from any vehicle's pov
        """
        angles = np.arange(256) * (2 * np.pi / 256) + observer.heading
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        distance = np.full(256, maximum_range, dtype=float)
        radial = np.zeros(256)
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is observer:
                continue
            c, s = np.cos(vehicle.heading), np.sin(vehicle.heading)
            rotation = np.array([[c, -s], [s, c]])
            origin = (observer.position - vehicle.position) @ rotation
            rays = directions @ rotation
            half = np.array([vehicle.LENGTH, vehicle.WIDTH]) / 2
            parallel = np.abs(rays) < 1e-10
            safe = np.where(parallel, 1.0, rays)
            lo = (-half - origin) / safe
            hi = (half - origin) / safe
            near = np.where(parallel, -np.inf, np.minimum(lo, hi)).max(axis=1)
            far = np.where(parallel, np.inf, np.maximum(lo, hi)).min(axis=1)
            outside = (parallel & (np.abs(origin) > half)).any(axis=1)
            hit = np.maximum(near, 0.0)
            valid = (~outside) & (far >= hit) & (hit < distance)
            distance[valid] = hit[valid]
            relative_velocity = vehicle.velocity - observer.velocity
            radial[valid] = directions[valid] @ relative_velocity
        ranges = distance.reshape(16, 16)
        indices = np.argmin(ranges, axis=1)
        speed = radial.reshape(16, 16)[np.arange(16), indices]
        return np.column_stack((ranges.min(axis=1) / maximum_range,
                                np.clip(speed / maximum_range, -1, 1))).astype(np.float32)

    def neighbors(self):
        """
        find the nearest front and back target lane cars
        """
        base = self.env.unwrapped
        ego = base.vehicle
        current = ego.lane_index
        lane_id = self.target_lane
        lane = (current[0], current[1], lane_id)
        same = [v for v in base.road.vehicles if v is not ego and v.lane_index == lane]
        front = [v for v in same if v.position[0] >= ego.position[0]]
        rear = [v for v in same if v.position[0] < ego.position[0]]
        return (min(front, key=lambda v: v.position[0], default=None),
                max(rear, key=lambda v: v.position[0], default=None))

    def request(self, action):
        if self.pending:
            raise RuntimeError('Only one request is allowed before a driving step')
        if action not in range(5):
            raise ValueError('Request action must be an integer from 0 to 4')
        self.pending = True
        if action == 0:
            return {'kind': 'none', 'neighbor': None, 'data': None, 'bits': 0}
        vehicle = self.neighbors()[0 if action in (1, 3) else 1]
        side = 'front' if action in (1, 3) else 'rear'
        if vehicle is None:
            self.requests += 1
            self.bits += 1
            return {'kind': 'unavailable', 'neighbor': side, 'data': None, 'bits': 1}
        self.requests += 1
        if action in (1, 2):
            label = vehicle.intent
            self.bits += 2
            return {'kind': 'intent', 'neighbor': side, 'data': label, 'bits': 2}
        readings = self.lidar(vehicle, 60.0)
        quantized = np.round(np.clip(readings, -1, 1) * 32767).astype(np.int16)
        bits = quantized.size * 16
        self.bits += bits
        return {'kind': 'lidar', 'neighbor': side, 'data': quantized, 'bits': bits}

    def step(self, driving_action, reply):
        raise NotImplementedError("Not implemented yet")

    def observation(self, scan):
        """
        assemble 36-value observation
        """
        v = self.env.unwrapped.vehicle
        return np.r_[np.asarray(scan, dtype=np.float32).reshape(-1), v.speed / 40,
                     v.lane_index[2] / 2, self.target_lane / 2, v.position[1] / 8].astype(np.float32)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
